"""Alias Resolve — 别名/实体解析节点，从 graph.py 拆出。"""
import re
import logging
from agents.state import AgentState
import config

logger = logging.getLogger(__name__)

_SCORE_RANGE_RE = re.compile(r'(\d[\d.]*)\s*分?\s*(以上|以下|超过|高于|低于)')
_YEAR_RE = re.compile(r'(20\d{2})')


def _get_query(state: dict) -> str:
    """从 state 提取用户查询"""
    if state.get("messages"):
        last_msg = state["messages"][-1]
        q = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        if q:
            return q
    return state.get("original_query", "")


logger = logging.getLogger(__name__)


# ── 模块级常量（避免重复创建）──────────────────────────────────────


# ── 节点函数 ──────────────────────────────────────────────────────

async def alias_resolve_node(state: AgentState) -> dict:
    """别名/实体解析: 别名 → 角色/梗 → 兜底标记"""
    from agents.alias import resolve_alias_ex
    from agents.metadata_index import index
    from agents.entity_resolver import resolve_entity

    query = _get_query(state)

    # ── 1. 零 LLM 实体解析 ──
    #    先走确定性匹配（metadata_index.fuzzy_lookup），
    #    命中直接短路，不调用任何 LLM。
    index_hit = index.fuzzy_lookup(query)
    if index_hit:
        return {
            "original_query": query,
            "resolved_query": query,
            "search_keywords": [index_hit["name"]],
            "entity_type": "anime",
            "entity_name": index_hit["name"],
            "entity_anime": index_hit["name"],
            "entity_confidence": 0.95,
            "entity_source": "fuzzy_lookup",
            "metadata": [index_hit],
        }

    # ── 2. 别名解析：L1 cache -> L2 LLM -> L3 web ──
    use_web = bool(
        getattr(config, "ENABLE_ALIAS_WEB_FALLBACK", False)
        and getattr(config, "ENABLE_WEB_SEARCH", False)
        and getattr(config, "TAVILY_API_KEY", "")
    )
    alias_result = resolve_alias_ex(query, use_web=use_web)
    resolved = alias_result["full_name"] or query
    was_resolved = alias_result["full_name"] is not None
    alias_source = alias_result["source"]
    alias_conf = alias_result["confidence"]

    # ── 2. 实体解析（角色/梗）──
    entity = resolve_entity(query)

    # ── 3. 番剧别名命中 → 正常流程 ──
    # 但如果 entity resolver 已高置信度命中角色/梗，alias 路径不拦截
    entity_is_strong = (
        entity
        and entity["confidence"] >= 0.8
        and entity["type"] in ("character", "meme")
        and entity["source"] == "dict"
    )
    if was_resolved and not entity_is_strong:
        if len(query) > 15 and resolved != query:
            from agents.cache import metadata_cache
            result = {
                "original_query": query,
                "resolved_query": query,
                "search_keywords": [resolved],
                "entity_type": "alias",
                "entity_name": resolved,
                "entity_anime": resolved,
                "entity_confidence": alias_conf,
                "entity_source": f"alias_{alias_source}",
            }
            _, meta = metadata_cache.resolve(resolved)
            if meta:
                result["metadata"] = [meta]
            return result

        from agents.cache import metadata_cache
        result = {
            "original_query": query,
            "resolved_query": resolved,
            "entity_type": "alias",
            "entity_name": resolved,
            "entity_anime": resolved,
            "entity_confidence": alias_conf,
            "entity_source": f"alias_{alias_source}",
        }
        _, meta = metadata_cache.resolve(resolved)
        if meta:
            result["metadata"] = [meta]
        return result

    # ── 4. 角色/梗实体命中（高置信度）→ 记录番剧名到 search_keywords ──
    if entity and entity["confidence"] >= 0.5 and entity["anime"]:
        return {
            "original_query": query,
            "resolved_query": query,
            "search_keywords": [entity["anime"]],
            "entity_type": entity["type"],
            "entity_name": entity["entity"],
            "entity_anime": entity["anime"],
            "entity_confidence": entity["confidence"],
            "entity_source": entity["source"],
        }

    # ── 5. 角色/梗低置信度 → 标记，planner 决定是否联网 ──
    if entity:
        return {
            "original_query": query,
            "resolved_query": query,
            "entity_type": entity["type"],
            "entity_name": entity["entity"],
            "entity_anime": entity.get("anime", ""),
            "entity_confidence": entity["confidence"],
            "entity_source": entity["source"],
        }

    # ── 6. 无实体 ──
    return {"original_query": query, "resolved_query": query}


def _might_be_alias(query: str) -> bool:
    """判断查询是否可能含番剧简称（避免对明确的长句调用LLM）"""
    if len(query) <= 15:
        return True
    # 查询中包含明显的推荐/对比/闲聊意图 → 不太可能只是问番剧名
    intent_words = ["推荐", "有没有", "怎么样", "对比", "哪个好", "是什么", "有哪些", "像"]
    if any(w in query for w in intent_words):
        return False
    return True


def should_skip_alias(query: str) -> bool:
    """快速判断是否可跳过 alias_resolve 节点（按需启用）

    以下场景跳过别名/实体解析:
      - 纯闲聊/问候（零信息量）
      - 明确元数据查询不含番剧名（如"2024年有哪些热血番"——查的是标签不是具体番名）
      - 全局开关关闭
      - Embedding 预检为 chat 类别且高置信度
    """
    if not config.ENABLE_ALIAS_RESOLVE:
        return True

    q = query.strip().lower()

    # seed 里登记的已知别名/缩写（如巨人/咒术/op/86）永不跳过，
    # 否则会被下方 len<=2 / 纯英文短查询守卫误杀。
    from agents.alias import HARDCODED_ALIASES
    if q in HARDCODED_ALIASES:
        return False

    # 纯闲聊 / 英文问候
    simple_greetings = {"你好", "谢谢", "再见", "早上好", "晚上好", "晚安",
                        "hi", "hello", "hey", "help", "thanks", "bye"}
    if q in simple_greetings or len(q) <= 2:
        return True

    # 纯英文短查询（不太可能涉及中文番剧别名；seed 命中的已在上方放行）
    if len(q) <= 10 and q.isascii():
        return True

    # Embedding 预检: chat 类别高置信度 → 纯闲聊，跳过
    if config.ENABLE_EMBEDDING_PREFILTER:
        try:
            from agents.planner import _prefilter
            route, confidence, _ = _prefilter(query)
            if route == "chat" and confidence >= config.EMBEDDING_PREFILTER_THRESHOLD:
                logger.info(f"  [alias_skip] embedding预检 chat={confidence:.2f}")
                return True
        except Exception:
            pass

    # 纯元数据查询特征（年份+标签/评分，无具体番剧名）
    # 如 "2024年有哪些热血番"、"评分9分以上的番"
    from agents.retrieval import _ANIME_TAGS, _SCORE_RANGE_RE, _YEAR_RE
    has_year = _YEAR_RE.search(query) is not None
    has_score = _SCORE_RANGE_RE.search(query) is not None
    has_tag = any(t in query for t in _ANIME_TAGS)
    # 有明确的元数据过滤词但没有短名称特征
    if (has_year or has_score or has_tag) and len(query) > 15:
        # 检查是否包含可能的番剧短名（2-6个中文字符的连续词）
        # 简单启发: 如果查询以"有哪些/推荐/是什么"结尾，大概率是泛查询
        broad_patterns = ["有哪些", "推荐", "是什么", "介绍", "列表"]
        if any(p in query for p in broad_patterns):
            logger.info(f"  [alias_skip] 泛查询特征: {query[:30]}...")
            return True

    return False


async def alias_skip_node(state: AgentState) -> dict:
    """alias_resolve 被跳过时设置必需字段的默认值"""
    query = _get_query(state)
    return {
        "original_query": query,
        "resolved_query": query,
    }

