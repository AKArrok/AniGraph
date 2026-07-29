"""Alias 解析工具 — 三级级联：磁盘缓存 → LLM → Web 兜底

Layer 设计:
  L1: 磁盘缓存 (data/alias_cache.json)
      运行时长出来的 {alias_lower: {full_name, confidence, source, ts}}
      seed = HARDCODED_ALIASES（常量，仅内存），首次命中零成本
  L2: simple_LLM 单次推理（结构化输出）
      LLM 知识内即可覆盖大部分 ACG 高频简称，~1s
  L3: Tavily 搜索 + simple_LLM 抽取
      仅当 L2 confidence<0.5 且允许 web 时触发，~2-4s

命中后写回 L1（内存 + 磁盘），下一次同 alias 直接零延迟。
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

from pydantic import BaseModel, Field

import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Seed: 硬编码高频别名（仅作为内存 seed，不写入磁盘缓存）
# ══════════════════════════════════════════════════════════════════════

def _build_hardcoded_alias_map() -> dict[str, str]:
    """圈内俚语/简称 → 正式名 seed 表。

    维护原则：只保留 Bangumi 官方 Alias 表**没有**登记的中文圈内简称。
    官方登记过的别名 → SQLite Alias 表（scripts/fetch_aliases.py 抓取的 10k+ 条）；
    seed 只承担官方漏收的社区俚语（如"素晴""钢炼""咒回""巨人"这类粉丝叫法）。
    """
    return {
        # ── 为美好的世界献上祝福！ ──
        "素晴": "为美好的世界献上祝福！",
        "为美": "为美好的世界献上祝福！",
        "为美好世界献上祝福": "为美好的世界献上祝福！",
        "konosuba": "为美好的世界献上祝福！",
        # ── Re:0 ──
        "re:0": "Re：从零开始的异世界生活",
        "从零开始": "Re：从零开始的异世界生活",
        # ── 进击的巨人 ──
        "巨人": "进击的巨人",
        "进击的巨人最终季": "进击的巨人 The Final Season",
        # ── 刀剑神域 ──
        "刀剑": "刀剑神域",
        # ── Fate ──
        "fate": "Fate/stay night",
        "fate zero": "Fate/Zero",
        # ── 命运石之门 ──
        "sg": "命运石之门",
        # ── 我的青春恋爱物语果然有问题。──
        "果青": "我的青春恋爱物语果然有问题",
        "大老师": "我的青春恋爱物语果然有问题",
        # ── 路人女主 ──
        "路人女主": "路人女主的养成方法",
        # ── 四月是你的谎言 ──
        "四月": "四月是你的谎言",
        "四月谎": "四月是你的谎言",
        # ── 我们仍未知道那天所看见的花的名字。──
        "花名": "我们仍未知道那天所看见的花的名字。",
        # ── 魔禁/超炮 ──
        "魔禁": "魔法禁书目录",
        "超炮": "某科学的超电磁炮",
        # ── 魔法少女小圆 ──
        "小圆": "魔法少女小圆",
        "圆神": "魔法少女小圆",
        # ── 物语系列 ──
        "物语": "物语系列",
        # ── CLANNAD ──
        "cl": "CLANNAD",
        # ── 紫罗兰永恒花园 ──
        "紫罗兰": "紫罗兰永恒花园",
        # ── Lycoris Recoil ──
        "蒜": "莉可丽丝",
        # ── 孤独摇滚！──
        "孤独摇滚": "孤独摇滚！",
        "滚": "孤独摇滚！",
        # ── 【我推的孩子】──
        "推子": "【我推的孩子】",
        "我推": "【我推的孩子】",
        # ── 葬送的芙莉莲 ──
        "芙莉莲": "葬送的芙莉莲",
        # ── 咒术回战 ──
        "咒术": "咒术回战",
        "咒回": "咒术回战",
        # ── 鬼灭之刃 ──
        "鬼灭": "鬼灭之刃",
        # ── 间谍过家家 ──
        "间谍": "间谍过家家",
        "过家家": "间谍过家家",
        # ── 无职转生 ──
        "无职": "无职转生～到了异世界就拿出真本事～",
        "无职转生": "无职转生～到了异世界就拿出真本事～",
        # ── 86 ──
        "86": "86 -不存在的战区-",
        # ── 赛博朋克：边缘行者 ──
        "边缘行者": "赛博浪客",
        # ── ONE PIECE ──
        "op": "ONE PIECE",
        "海贼": "ONE PIECE",
        # ── 火影忍者 ──
        "火影": "火影忍者",
        # ── 钢之炼金术师 FULLMETAL ALCHEMIST ──
        "钢炼": "钢之炼金术师 FULLMETAL ALCHEMIST",
    }


# 全局别名 seed（启动时构建；仅供 L1 内存查询，不写入磁盘）
HARDCODED_ALIASES: dict[str, str] = _build_hardcoded_alias_map()


# ══════════════════════════════════════════════════════════════════════
# L0: Bangumi 官方别名（SQLite）— 由 scripts/fetch_aliases.py 抓取的 Alias 表
#     10k+ 条经过官方登记的别名，是最可信的事实来源
# ══════════════════════════════════════════════════════════════════════

import sqlite3

_ALIAS_DB_INDEX: dict[str, str] | None = None  # alias_lower -> full_name (中文名优先)
_ALIAS_DB_LOCK = threading.Lock()


def _load_alias_db() -> dict[str, str]:
    """一次性把 SQLite Alias 表 + Anime.anime_title 加载到内存 dict。

    ~10k 条精确别名映射，查询 O(1)；不存在或表为空时返回空 dict。
    """
    global _ALIAS_DB_INDEX
    if _ALIAS_DB_INDEX is not None:
        return _ALIAS_DB_INDEX
    with _ALIAS_DB_LOCK:
        if _ALIAS_DB_INDEX is not None:
            return _ALIAS_DB_INDEX
        db_path = getattr(config, "ALIAS_DB_PATH", "") or ""
        idx: dict[str, str] = {}
        if not db_path or not os.path.exists(db_path):
            _ALIAS_DB_INDEX = idx
            return idx
        try:
            conn = sqlite3.connect(db_path)
            # anime_title 也当作 alias 的一种（用户直接输入中文名时也命中）
            for aid, title in conn.execute(
                "SELECT anime_id, anime_title FROM Anime WHERE anime_title IS NOT NULL"
            ):
                if title:
                    idx.setdefault(title.strip().lower(), title.strip())
            # 真正的别名表；'__none__' 哨兵跳过
            for alias, title in conn.execute(
                """
                SELECT al.alias, a.anime_title
                FROM Alias al JOIN Anime a ON al.anime_id = a.anime_id
                WHERE al.key != '__none__' AND alias IS NOT NULL
                """
            ):
                if not alias or not title:
                    continue
                key = alias.strip().lower()
                if not key:
                    continue
                # 首次命中的 title 生效（Anime.anime_title 已 setdefault，
                # 这里保留最先见到的别名 -> title 映射）
                idx.setdefault(key, title.strip())
            conn.close()
            logger.info(f"  [alias] 加载 SQLite 别名索引: {len(idx)} 条 <- {db_path}")
        except Exception as e:
            logger.warning(f"  [alias] SQLite 别名索引加载失败({e})，忽略并继续")
        _ALIAS_DB_INDEX = idx
        return idx


def _lookup_alias_db(query_lower: str) -> str | None:
    """L0 精确别名查询（不做 LIKE，避免误命中；LIKE 交给 metadata_index.fuzzy_lookup）"""
    idx = _load_alias_db()
    return idx.get(query_lower)


# ══════════════════════════════════════════════════════════════════════
# 属性词尾缀剥离：把"XX评分/XX声优/XX几集"等剥成"XX"，让 seed/LLM 提高命中率
# ══════════════════════════════════════════════════════════════════════

# 常见"番剧属性/意图"尾缀，按长度降序在正则里被合并
_ATTR_SUFFIXES = (
    # 属性问句（长优先，避免"评分9分以上"这种正当查询被误剥）
    "什么时候上映", "什么时候出", "什么时候播", "多少集完结",
    "有多少集", "多少集", "多少话", "有几集", "几集完结", "几集", "几话",
    "多少分", "评分多少", "评分是多少", "评分怎么样", "评分",
    "声优是谁", "声优", "配音是谁", "配音", "cv是谁", "cv",
    "导演是谁", "导演", "编剧是谁", "编剧",
    "制作公司", "制作方", "公司", "工作室", "出品",
    "值得看吗", "值得追吗", "好看吗", "怎么样", "如何", "咋样",
    "讲什么的", "讲的什么", "讲什么", "讲的啥", "说的是什么", "说的什么", "说了什么",
    "说的啥", "说啥", "是啥", "是什么", "介绍一下", "介绍下", "介绍",
    # 指示修饰语（"素晴这部番的评分" 剥完"评分"后还剩"这部番的"，需再剥一层）
    "这部番的", "这部番", "那部番的", "那部番", "这部动画", "那部动画", "这番", "那番",
    # 语气/助词尾巴（顺手剥掉一层，避免"素晴呢" miss）
    "了没", "了吗", "呢", "啊", "吗", "呀", "哦",
)

import re as _re_alias  # 复用文件顶部 re 也行，用别名避免和 module import 混淆
_ATTR_SUFFIX_RE = _re_alias.compile(
    r"(" + "|".join(_re_alias.escape(s) for s in _ATTR_SUFFIXES) + r")+$"
)

_MIN_STRIPPED_LEN = 2  # 剥离后短于 2 字视为退化，放弃
_MAX_STRIP_ROUNDS = 3  # 迭代剥离上限，避免死循环

# ── 定义/介绍类前缀（事实查询，剥掉后应命中确定性字典）──────────────
# 只收录"这是什么/介绍一下"这类**指向单一实体**的疑问前缀。
# 故意**不**收录"类似/推荐/求/找"这类意图词——那些由下方 _INTENT_GUARD_RE
# 显式拦截、放空交给下游语义检索，避免把"有没有类似X的番"劫持成"查 X"。
_QUERY_PREFIXES = (
    "介绍一下", "介绍下", "简单介绍", "讲一讲", "讲讲", "说一说", "说说",
    "什么是", "是什么", "谁是", "有没有", "听说过", "了解一下", "了解下",
    "问一下", "问下", "想知道", "科普一下", "科普下",
    "这个", "那个", "这部", "那部",
)
_QUERY_PREFIX_RE = _re_alias.compile(
    r"^(?:" + "|".join(_re_alias.escape(p) for p in _QUERY_PREFIXES) + r")+"
)

# ── 意图守卫：命中即放弃确定性实体锁定 ─────────────────────────────
# "推荐/类似/相似"类查询是开放语义问题，锁定单一实体会把语义检索劫持成
# 事实问答。命中这些词时 _strip_attr_suffix 直接返回 None，让链路落到 LLM/
# 语义检索，而不是硬匹配一个碰巧出现的番名。
_INTENT_GUARD_RE = _re_alias.compile(
    r"(类似|相似|差不多|一样|像.{0,6}(的|那样|这样)|推荐|求推|安利|" 
    r"找番|补番|有没有.{0,8}的番|之类|这种|那种)"
)


def _strip_attr_suffix(q: str) -> str | None:
    """把 query 尾部的属性词/语气词剥掉，返回剥离后的短查询。

    - 剥离后与原文相同 -> 返回 None（避免调用方重复查表）
    - 剥离后过短或纯数字/符号 -> 返回 None（避免退化到"评分"这种空壳）
    """
    # 意图守卫：推荐/类似类查询不做实体剥离，直接放空交给下游语义检索
    if _INTENT_GUARD_RE.search(q):
        return None
    # 先剥定义/介绍类前缀，再迭代剥属性/语气尾缀（最多 _MAX_STRIP_ROUNDS 轮），
    # 层层剥到别名核。例："素晴这部番的评分是多少" -> "素晴这部番的" -> "素晴"
    current = _QUERY_PREFIX_RE.sub("", q).strip()
    for _ in range(_MAX_STRIP_ROUNDS):
        stripped = _ATTR_SUFFIX_RE.sub("", current).strip()
        if not stripped or stripped == current:
            break
        current = stripped
    stripped = current
    if not stripped or stripped == q:
        return None
    if len(stripped) < _MIN_STRIPPED_LEN:
        return None
    # 纯数字/纯符号也算退化
    if _re_alias.fullmatch(r"[\d\s\W_]+", stripped):
        return None
    return stripped


# ══════════════════════════════════════════════════════════════════════
# L1: 持久化缓存（内存 + 磁盘）
# ══════════════════════════════════════════════════════════════════════

_LEARNED_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOADED = False
_CACHE_LOCK = threading.Lock()

_MIN_CACHE_CONFIDENCE = 0.7  # 只有高置信度结果才落盘复用


def _cache_path() -> str | None:
    p = getattr(config, "ALIAS_CACHE_PATH", "") or ""
    return p or None


def _load_learned_cache() -> None:
    """首次调用时从磁盘加载 learned cache（幂等）"""
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    with _CACHE_LOCK:
        if _CACHE_LOADED:
            return
        path = _cache_path()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    _LEARNED_CACHE.update(data)
                logger.info(f"  [alias] 加载磁盘缓存: {len(_LEARNED_CACHE)} 条 <- {path}")
            except Exception as e:
                logger.warning(f"  [alias] 磁盘缓存读取失败({e})，忽略并继续")
        _CACHE_LOADED = True


def _persist_entry(alias_lower: str, entry: dict[str, Any]) -> None:
    """把新条目追加写到磁盘缓存（整文件重写，条目量不大所以够用）"""
    path = _cache_path()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 复制一份避免序列化时被并发修改
        snapshot = dict(_LEARNED_CACHE)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"  [alias] 磁盘缓存写入失败({e})，仅进程内保留")


_MIN_SUBSTRING_ALIAS_LEN = 3   # 短于此长度的 seed 只允许精确匹配，不做子串包含
_MIN_SUBSTRING_COVERAGE = 0.5  # 命中别名长度需占查询的比例，低于此视为噪声


def _substring_match(q_lower: str) -> str | None:
    """seed 包含匹配（带安全闸）。

    两道闸避免短别名误命中：
      1. 长度闸：别名长度 >= _MIN_SUBSTRING_ALIAS_LEN 才允许子串包含
         （op/滚/86 这种只能精确命中）。
      2. 覆盖闸：命中别名长度需占查询 >= _MIN_SUBSTRING_COVERAGE，
         挡掉 2077游戏好玩吗 这种别名只占一小截的句子。
    长优先：同时命中多个时取最长的（覆盖率最高）。
    """
    qlen = len(q_lower)
    if qlen == 0:
        return None
    for alias, full_name in sorted(HARDCODED_ALIASES.items(), key=lambda x: -len(x[0])):
        if len(alias) < _MIN_SUBSTRING_ALIAS_LEN:
            continue
        if alias in q_lower and len(alias) / qlen >= _MIN_SUBSTRING_COVERAGE:
            return full_name
    return None


def _lookup_cache_raw(q_lower: str) -> str | None:
    """单趟查表：SQLite 官方别名 → seed 精确 → learned 精确 → seed 包含 → MetadataCache。"""
    if not q_lower:
        return None

    # L0: SQLite Alias 表（最权威，Bangumi 官方登记）
    hit = _lookup_alias_db(q_lower)
    if hit:
        return hit

    # seed 精确
    if q_lower in HARDCODED_ALIASES:
        return HARDCODED_ALIASES[q_lower]

    # learned 精确
    entry = _LEARNED_CACHE.get(q_lower)
    if entry and entry.get("full_name"):
        return entry["full_name"]

    # seed 包含匹配（带闸：短别名/低覆盖不允许子串命中）
    hit = _substring_match(q_lower)
    if hit:
        return hit

    # MetadataCache（外部可能预热过的动态映射）
    try:
        from agents.cache import metadata_cache
        hit = metadata_cache.resolve_alias(q_lower)
        if hit:
            return hit
    except Exception:
        pass

    return None


def _lookup_cache(query_lower: str) -> str | None:
    """L1 查表：原文 → 剥属性词尾缀后再查一遍。"""
    _load_learned_cache()

    hit = _lookup_cache_raw(query_lower)
    if hit:
        return hit

    stripped = _strip_attr_suffix(query_lower)
    if stripped:
        return _lookup_cache_raw(stripped)
    return None


def _remember(alias_lower: str, full_name: str, confidence: float, source: str) -> None:
    """把新学到的映射写回内存 + 磁盘（低置信度不落盘）"""
    if not alias_lower or not full_name:
        return
    entry = {
        "full_name": full_name,
        "confidence": round(float(confidence), 3),
        "source": source,
        "ts": int(time.time()),
    }
    with _CACHE_LOCK:
        _LEARNED_CACHE[alias_lower] = entry
        # 同步一份到 MetadataCache，让别处也能命中
        try:
            from agents.cache import metadata_cache
            metadata_cache.add_alias(alias_lower, full_name)
        except Exception:
            pass
        if confidence >= _MIN_CACHE_CONFIDENCE:
            _persist_entry(alias_lower, entry)


# ══════════════════════════════════════════════════════════════════════
# L2: LLM 单次推理（结构化输出，置信度）
# ══════════════════════════════════════════════════════════════════════

class AliasLLMOutput(BaseModel):
    """LLM 别名解析结构化输出"""
    anime: str = Field(description="正式中文番剧名，无法确认时输出空字符串")
    confidence: float = Field(description="确信度 0.0-1.0，无法确认时 0.0", ge=0.0, le=1.0)


_LLM_PROMPT = """把用户输入的字符串解析为它可能指代的 ACG 番剧正式中文名，并给出置信度。

判断标准：
- 是明确的番剧简称、缩写、罗马音、英文名、粉丝叫法：给出正式中文名，置信度反映你的把握。
- 是番剧里的角色名、梗、术语：属于"番剧简称"处理不了，把 anime 留空，confidence 设为 0。
- 是问候语、普通词语、跟番剧无关的问题：anime 留空，confidence 设为 0。
- 不确定是不是番剧：宁可置空也不要猜，避免把常见词错认成作品名。

简称: {query}
"""


def _llm_resolve_alias(query: str) -> AliasLLMOutput | None:
    """simple_LLM 单次推理，结构化输出保证字段存在"""
    try:
        from llms import simple_LLM, invoke_structured
        from langchain_core.messages import HumanMessage, SystemMessage

        return invoke_structured(simple_LLM, AliasLLMOutput, [
            SystemMessage(content=_LLM_PROMPT.format(query=query[:100])),
            HumanMessage(content=f"简称: {query}"),
        ])
    except Exception as e:
        logger.warning(f"  [alias L2] LLM 解析异常: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# L3: Web 搜索兜底（Tavily + simple_LLM 抽取）
# ══════════════════════════════════════════════════════════════════════

_WEB_EXTRACT_PROMPT = """从以下搜索结果中提取其中提到的番剧正式中文名。

搜索结果:
{results}

输出 JSON 格式: {{"anime": "正式中文番剧名", "confidence": 0.8}}
如果搜索结果中不包含任何番剧信息，输出: {{"anime": "", "confidence": 0.0}}"""


def _web_resolve_alias(query: str) -> AliasLLMOutput | None:
    """Tavily 搜索 + simple_LLM 抽取"""
    try:
        from tools.web_search import search_web
        from llms import simple_LLM, llm_invoke_with_retry
        from langchain_core.messages import HumanMessage, SystemMessage

        search_text = search_web.invoke(f"{query} 番剧 动漫 别名 简称")
        if not search_text or len(search_text) < 30:
            logger.info(f"  [alias L3] 搜索无有效结果 (query={query})")
            return None

        resp = llm_invoke_with_retry(simple_LLM, [
            SystemMessage(content=_WEB_EXTRACT_PROMPT.format(results=search_text[:2000])),
            HumanMessage(content=f"查询: {query}"),
        ])
        text = resp.content.strip()
        match = re.search(r'\{[^}]+\}', text)
        if match:
            import json as _json
            data = _json.loads(match.group())
            return AliasLLMOutput(anime=data.get("anime", ""), confidence=data.get("confidence", 0.0))
    except Exception as e:
        logger.warning(f"  [alias L3] Web 解析异常: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════
# resolve_alias_dict — 纯字典匹配（向后兼容，供 entity_resolver 等用）
# ══════════════════════════════════════════════════════════════════════

def resolve_alias_dict(query: str) -> str | None:
    """纯字典匹配：seed / 包含 / learned cache / MetadataCache 四层，零 LLM"""
    q = query.strip().lower()
    if not q:
        return None
    return _lookup_cache(q)


# ══════════════════════════════════════════════════════════════════════
# resolve_alias_ex — 三级级联主入口（新 API）
# ══════════════════════════════════════════════════════════════════════

def resolve_alias_ex(query: str, use_web: bool = False) -> dict[str, Any]:
    """三级级联别名解析，返回字典。

    Args:
        query: 用户原始查询字符串
        use_web: L2 低置信度时是否允许升级到 L3（Tavily）

    Returns:
        {
            "full_name": str | None,   # 正式番剧名，未命中则为 None
            "source": str,             # "cache" | "llm" | "web" | "miss"
            "confidence": float,       # 0.0-1.0
        }
    """
    q = query.strip().lower()
    if not q:
        return {"full_name": None, "source": "miss", "confidence": 0.0}

    # L1: 缓存命中
    hit = _lookup_cache(q)
    if hit:
        return {"full_name": hit, "source": "cache", "confidence": 1.0}

    # 尝试属性词剥离，用短查询喂 LLM，识别率更高
    stripped = _strip_attr_suffix(q)
    llm_query = stripped or query
    cache_key = stripped or q  # 归一化写回 key，避免"钢炼几集"/"钢炼多少集"两条

    # L2: LLM 推理
    llm_out = _llm_resolve_alias(llm_query)
    if llm_out and llm_out.anime and llm_out.confidence >= 0.5:
        _remember(cache_key, llm_out.anime, llm_out.confidence, "llm")
        return {"full_name": llm_out.anime, "source": "llm", "confidence": llm_out.confidence}

    # L3: Web 兜底（仅当 use_web=True 且 L2 给空或低置信度）
    if use_web and config.ENABLE_WEB_SEARCH and config.TAVILY_API_KEY:
        web_out = _web_resolve_alias(llm_query)
        if web_out and web_out.anime and web_out.confidence >= 0.5:
            _remember(cache_key, web_out.anime, web_out.confidence, "web")
            return {"full_name": web_out.anime, "source": "web", "confidence": web_out.confidence}

    return {"full_name": None, "source": "miss", "confidence": 0.0}


# ══════════════════════════════════════════════════════════════════════
# resolve_alias — 旧签名（向后兼容，包装 resolve_alias_ex）
# ══════════════════════════════════════════════════════════════════════

def resolve_alias(query: str, use_llm: bool = True) -> tuple[str, bool]:
    """旧签名兼容包装。

    Returns:
        (resolved_query, was_resolved)
    """
    # 按旧语义：use_llm=False 只走 L1，use_llm=True 走 L1+L2
    result = resolve_alias_ex(query, use_web=False)
    if result["full_name"]:
        return result["full_name"], True
    if use_llm:
        # 旧逻辑里 use_llm=True 会走一次 LLM 但不带 web
        llm_out = _llm_resolve_alias(query)
        if llm_out and llm_out.anime and llm_out.confidence >= 0.5:
            _remember(query.strip().lower(), llm_out.anime, llm_out.confidence, "llm")
            return llm_out.anime, True
    return query, False


# 启动时预热：确保 learned cache 被加载
_load_learned_cache()
