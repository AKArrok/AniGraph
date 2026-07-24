"""动漫识图工具 - trace.moe API + VLM fallback

通过 trace.moe API 识别动漫截图，返回番剧名/集数/时间戳/预览；
低置信度或 API 异常时降级到 VLM（多模态大模型）描述图片内容。
"""
import hashlib
import logging
import functools
from collections import OrderedDict

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

import config

# ══════════════════════════════════════════════════════════════════════
# 本地英中番剧名映射表（高频番，命中即免 LLM 调用）
# key 同时收录英文与罗马音写法，最大化命中率
# ══════════════════════════════════════════════════════════════════════
_EN_CN_MAP: dict[str, str] = {
    # 进击的巨人
    "Attack on Titan": "进击的巨人",
    "Shingeki no Kyojin": "进击的巨人",
    # 鬼灭之刃
    "Demon Slayer": "鬼灭之刃",
    "Kimetsu no Yaiba": "鬼灭之刃",
    # 咒术回战
    "Jujutsu Kaisen": "咒术回战",
    # 我的英雄学院
    "My Hero Academia": "我的英雄学院",
    "Boku no Hero Academia": "我的英雄学院",
    # 海贼王
    "One Piece": "海贼王",
    # 火影忍者
    "Naruto": "火影忍者",
    "Naruto: Shippuden": "火影忍者疾风传",
    # 死神
    "Bleach": "死神",
    "Bleach: Thousand-Year Blood War": "死神 千年血战篇",
    # 钢之炼金术师
    "Fullmetal Alchemist: Brotherhood": "钢之炼金术师 FULLMETAL ALCHEMIST",
    # 命运石之门
    "Steins;Gate": "命运石之门",
    # 反叛的鲁路修
    "Code Geass": "反叛的鲁路修",
    "Code Geass: Lelouch of the Rebellion": "反叛的鲁路修",
    # 死亡笔记
    "Death Note": "死亡笔记",
    # 全职猎人
    "Hunter x Hunter": "全职猎人",
    "Hunter × Hunter": "全职猎人",
    # 一拳超人
    "One Punch Man": "一拳超人",
    "One-Punch Man": "一拳超人",
    # 灵能百分百
    "Mob Psycho 100": "灵能百分百",
    # 东京喰种
    "Tokyo Ghoul": "东京喰种",
    # Re:从零开始的异世界生活
    "Re:Zero kara Hajimeru Isekai Seikatsu": "Re:从零开始的异世界生活",
    "Re:Zero - Starting Life in Another World": "Re:从零开始的异世界生活",
    # 刀剑神域
    "Sword Art Online": "刀剑神域",
    # 游戏人生
    "No Game No Life": "游戏人生",
    # 你的名字
    "Your Name": "你的名字",
    "Kimi no Na wa": "你的名字",
    # 千与千寻
    "Spirited Away": "千与千寻",
    "Sen to Chihiro no Kamikakushi": "千与千寻",
    # 来自深渊
    "Made in Abyss": "来自深渊",
    # 冰海战记
    "Vinland Saga": "冰海战记",
    # 电锯人
    "Chainsaw Man": "电锯人",
    # 间谍过家家
    "Spy x Family": "间谍过家家",
    # 葬送的芙莉莲
    "Frieren: Beyond Journey's End": "葬送的芙莉莲",
    "Sousou no Frieren": "葬送的芙莉莲",
    # 孤独摇滚
    "Bocchi the Rock!": "孤独摇滚",
    # 我推的孩子
    "Oshi no Ko": "我推的孩子",
    # 排球少年
    "Haikyu!!": "排球少年",
    "Haikyuu!!": "排球少年",
    # 妖精的尾巴
    "Fairy Tail": "妖精的尾巴",
    # 龙珠
    "Dragon Ball": "龙珠",
    "Dragon Ball Z": "龙珠Z",
}

# 大小写不敏感查找表（O(1) 命中）
_EN_CN_MAP_LOWER: dict[str, str] = {
    k.lower(): v for k, v in _EN_CN_MAP.items()
}

# ══════════════════════════════════════════════════════════════════════
# 模块级缓存
# ══════════════════════════════════════════════════════════════════════
# 标题翻译 LRU 缓存（maxsize=256）：缓存 LLM 翻译结果，避免重复调用
_TITLE_LRU_CACHE: OrderedDict[str, str] = OrderedDict()
_TITLE_CACHE_MAX = 256

# 识图结果 FIFO 缓存（maxsize=50）：key = sha256(image_b64)
_SEARCH_CACHE: dict[str, dict] = {}
_SEARCH_CACHE_MAX = 50


def _format_timestamp(seconds: float) -> str:
    """秒数转 mm:ss 格式时间戳"""
    if not seconds or seconds < 0:
        return ""
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def _llm_translate_title(title_raw: str) -> str:
    """用轻量 LLM 将番剧标题翻译为中文正式名（同步调用 simple_LLM.invoke）"""
    try:
        from llms import simple_LLM
        prompt = f"Convert this anime title to Chinese official name: {title_raw}"
        resp = simple_LLM.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception as e:
        logging.warning(f"  [识图] LLM 标题翻译失败 ({title_raw}): {e}")
        return title_raw


def normalize_to_chinese_title(title_raw: str, anilist_id: int = 0) -> str:
    """番剧标题中文化：本地映射表优先，LLM 翻译兜底。

    trace.moe 返回 AniList 英文/罗马音/原文标题，而知识库是中文键，
    必须经此函数映射为中文正式名才能命中下游检索。

    使用模块级 LRU 缓存（maxsize=256）缓存翻译结果，避免重复调用 LLM。
    """
    if not title_raw:
        return ""

    cache_key = title_raw.strip().lower()

    # 1. 命中 LRU 缓存
    if cache_key in _TITLE_LRU_CACHE:
        _TITLE_LRU_CACHE.move_to_end(cache_key)  # LRU: 访问后移到末尾
        return _TITLE_LRU_CACHE[cache_key]

    # 2. 本地英中映射表（大小写不敏感）
    cn = _EN_CN_MAP_LOWER.get(cache_key, "")

    # 3. LLM 翻译兜底
    if not cn:
        cn = _llm_translate_title(title_raw)

    # 写入 LRU 缓存
    _TITLE_LRU_CACHE[cache_key] = cn
    _TITLE_LRU_CACHE.move_to_end(cache_key)
    if len(_TITLE_LRU_CACHE) > _TITLE_CACHE_MAX:
        _TITLE_LRU_CACHE.popitem(last=False)  # 淘汰最久未用

    return cn


async def describe_image_with_vlm(image_b64: str) -> str:
    """VLM 多模态识图 fallback。

    trace.moe 低置信度/失败时，用多模态大模型描述图片内容。
    VLM_API_KEY 未配置时直接返回空字符串，跳过 fallback。
    """
    if not config.VLM_API_KEY:
        return ""

    try:
        vlm = ChatOpenAI(
            api_key=config.VLM_API_KEY,
            base_url=config.VLM_BASE_URL,
            model=config.VLM_MODEL,
            request_timeout=30,
            max_retries=1,
        )
        message = HumanMessage(content=[
            {
                "type": "text",
                "text": "这张图片是哪部动漫的截图？请用中文回答番剧名，如果不确定就描述图片内容。",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ])
        resp = await vlm.ainvoke([message])
        return resp.content.strip()
    except Exception as e:
        logging.warning(f"  [识图] VLM fallback 失败: {e}")
        return ""


async def _call_trace_moe(image_b64: str) -> dict | None:
    """调用 trace.moe API，带超时 + 1 次重试。返回最佳匹配结果或 None。"""
    payload = {"image": image_b64}
    max_attempts = 2  # 初始 + 1 次重试

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=config.TRACE_MOE_TIMEOUT) as client:
                resp = await client.post(config.TRACE_MOE_API_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()

            results = data.get("result") or []
            if not results:
                logging.info("  [识图] trace.moe 返回空结果")
                return None

            # 取相似度最高的结果（trace.moe 默认按相似度降序，取 max 更稳妥）
            best = max(results, key=lambda r: r.get("similarity", 0))
            logging.info(
                f"  [识图] trace.moe 命中: "
                f"similarity={best.get('similarity', 0):.3f}"
            )
            return best
        except Exception as e:
            logging.warning(
                f"  [识图] trace.moe 调用失败 "
                f"(attempt {attempt}/{max_attempts}): {e}"
            )
            if attempt == max_attempts:
                return None
    return None


async def search_anime_by_image(image_b64: str) -> dict:
    """动漫截图识别主入口。

    流程:
      1. 查模块级缓存（sha256 key，FIFO 50 条）
      2. 调 trace.moe API 识别截图
      3. 高置信度（>= IMAGE_CONFIDENCE_THRESHOLD）: 中文化标题，source="trace_moe"
      4. 低置信度/失败: VLM fallback 描述图片，source="vlm_fallback" 或 "failed"

    返回:
      {
        "matched": bool,        # trace.moe 高置信度命中为 True
        "anilist_id": int,      # AniList ID，未命中为 0
        "title_raw": str,       # trace.moe 原始标题（英文/罗马音/原文）
        "title_cn": str,        # 中文化正式番剧名（核心字段）/ VLM 描述
        "episode": int | None,  # 集数
        "timestamp": str,       # "mm:ss" 时间戳
        "similarity": float,    # 相似度 0.0-1.0
        "preview_url": str,     # 视频预览 URL
        "source": str,          # "trace_moe" | "vlm_fallback" | "failed"
      }
    """
    # 1. 缓存检查
    cache_key = hashlib.sha256(image_b64.encode()).hexdigest()
    if cache_key in _SEARCH_CACHE:
        return _SEARCH_CACHE[cache_key]

    # 2. 调用 trace.moe
    best = await _call_trace_moe(image_b64)

    # 3. 解析结果
    if best and best.get("similarity", 0) >= config.IMAGE_CONFIDENCE_THRESHOLD:
        # 高置信度命中
        title_obj = best.get("anilist", {}).get("title", {})
        title_raw = (
            title_obj.get("english")
            or title_obj.get("romaji")
            or title_obj.get("native")
            or ""
        )
        anilist_id = best.get("anilist_id", 0) or best.get("anilist", {}).get("id", 0)
        episode = best.get("episode")
        from_time = best.get("from", 0)
        similarity = best.get("similarity", 0)
        preview_url = best.get("video", "")

        title_cn = normalize_to_chinese_title(title_raw, anilist_id)
        result = {
            "matched": True,
            "anilist_id": anilist_id,
            "title_raw": title_raw,
            "title_cn": title_cn,
            "episode": int(episode) if episode is not None else None,
            "timestamp": _format_timestamp(from_time),
            "similarity": similarity,
            "preview_url": preview_url,
            "source": "trace_moe",
        }
    else:
        # 低置信度或 API 失败 -> VLM fallback
        vlm_desc = await describe_image_with_vlm(image_b64)
        # 保留低置信度 trace.moe 的部分信息（若有）
        sim = best.get("similarity", 0) if best else 0.0
        preview = best.get("video", "") if best else ""
        if vlm_desc:
            result = {
                "matched": False,
                "anilist_id": 0,
                "title_raw": "",
                "title_cn": vlm_desc,
                "episode": None,
                "timestamp": "",
                "similarity": sim,
                "preview_url": preview,
                "source": "vlm_fallback",
            }
        else:
            result = {
                "matched": False,
                "anilist_id": 0,
                "title_raw": "",
                "title_cn": "",
                "episode": None,
                "timestamp": "",
                "similarity": sim,
                "preview_url": preview,
                "source": "failed",
            }

    # 4. 写入缓存（FIFO 淘汰）
    if len(_SEARCH_CACHE) >= _SEARCH_CACHE_MAX:
        _SEARCH_CACHE.pop(next(iter(_SEARCH_CACHE)))
    _SEARCH_CACHE[cache_key] = result

    return result
