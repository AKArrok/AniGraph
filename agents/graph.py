"""多 Agent 协作图 — 基于 ExecutionPlan 动态编排

流程:
  START → [alias_resolve]? → history_extractor → context_builder → planner
    → query_processing → knowledge_retrieval
    → [metadata_reasoner || similar_expert] (parallel via Send)
    → merge → [web_fallback]? → answer_planner → answer → END
    → [simple_fact_answer] → END (快速通道)

节点分类:
  必备 (每轮必走):
    history_extractor  — 提取对话历史
    context_builder    — 构建对话上下文
    planner            — 意图分类 + 策略决策
    query_processing   — 查询优化 (direct策略零LLM)
    knowledge_retrieval— 知识检索 (按plan分流)
    merge              — 合并Expert结果
    answer_planner     — 回答结构规划 (零LLM)
    answer             — 生成最终回答

  按需 (动态加入):
    alias_resolve      — 仅在查询可能含别名/角色/梗时启用
    metadata_reasoner  — 仅 plan.experts 包含时启用
    similar_expert     — 仅 plan.experts 包含时启用
    web_fallback       — 仅 plan.need_web 或合并结果低置信时启用
    simple_fact_answer — 仅 plan.query_type == simple_fact 时走快速通道
"""
import time
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import re
import random
import logging

from agents.state import AgentState
from agents.planner import planner_node
from agents.history_extractor import history_extractor_node
from agents.context_builder import context_builder_node
from agents.metadata_reasoner import metadata_reasoner_node
from agents.similar_expert import similar_expert_node
from agents.simple_fact_answer import simple_fact_answer_node
from agents.answer import answer_node
from agents.web_fallback import web_fallback_node, should_trigger_web
from agents.merge import merge_expert_results
import config

logger = logging.getLogger(__name__)


# ── 模块级常量（避免重复创建）──────────────────────────────────────

# 番剧标签列表（_knowledge_retrieval_node 和 _extract_metadata_filters 共用）
_ANIME_TAGS: tuple[str, ...] = (
    "热血", "动作", "搞笑", "异世界", "奇幻", "科幻", "恋爱", "日常",
    "治愈", "悬疑", "推理", "战斗", "冒险", "校园", "机战", "运动",
    "魔法", "后宫", "百合", "耽美", "美食", "音乐", "竞技",
    "战争", "历史", "恐怖", "职场", "偶像", "转生", "游戏",
)

# 预编译正则（_extract_metadata_filters 用，避免每次调用重新编译）
_SCORE_RANGE_RE = re.compile(r"(\d[\d.]*)\s*分?\s*(以上|以下|超过|高于|低于)")
_YEAR_RE = re.compile(r"(20\d{2})")


# ── 节点函数 ──────────────────────────────────────────────────────

async def _alias_resolve_node(state: AgentState) -> dict:
    """别名/实体解析: 别名 → 角色/梗 → 兜底标记"""
    from agents.alias import resolve_alias
    from agents.entity_resolver import resolve_entity

    query = _get_query(state)

    # ── 1. 现有别名解析 ──
    resolved, was_resolved = resolve_alias(query, use_llm=False)

    if not was_resolved and _might_be_alias(query):
        resolved, was_resolved = resolve_alias(query, use_llm=True)

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
                "entity_confidence": 0.90,
                "entity_source": "dict",
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
            "entity_confidence": 0.90,
            "entity_source": "dict",
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


def _should_skip_alias(query: str) -> bool:
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

    # 纯闲聊 / 英文问候
    simple_greetings = {"你好", "谢谢", "再见", "早上好", "晚上好", "晚安",
                        "hi", "hello", "hey", "help", "thanks", "bye"}
    if q in simple_greetings or len(q) <= 2:
        return True

    # 纯英文短查询（不太可能涉及中文番剧别名）
    if len(q) <= 10 and q.isascii() and not any(w in q for w in ["re0", "eva", "sao"]):
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


async def _query_processing_node(state: AgentState) -> dict:
    """查询处理节点: 根据 plan.rewrite_strategy 执行 Rewrite/HyDE/Decompose/Direct"""
    plan = state.get("plan", {})
    strategy = plan.get("rewrite_strategy", "rewrite")
    query = state.get("resolved_query", "") or state.get("original_query", "")

    from tools.registry import tool_registry

    if strategy == "direct":
        queries = [query]
    elif strategy == "hyde":
        fn = tool_registry.get_callable("hyde_generate")
        queries = fn(query) if fn else [query]
    elif strategy == "decompose":
        fn = tool_registry.get_callable("decompose")
        queries = fn(query) if fn else [query]
    else:  # rewrite
        fn = tool_registry.get_callable("multi_query_rewrite")
        queries = fn(query) if fn else [query]

    return {
        "shared_context": queries,
        "optimized_queries": queries,
        "query_strategy": strategy,
    }


def _retrieve_by_keywords(keywords: list[str]) -> list[dict]:
    """别名关键词优先查 Metadata Index（番剧名 + 标签模糊匹配）"""
    results: list[dict] = []
    if not keywords:
        return results
    try:
        from agents.metadata_index import index
        for kw in keywords[:3]:
            md = index.search_by_name(kw)
            if md:
                results.extend(md)
        for kw in keywords[:3]:
            tag_kw = kw.strip("【】！! ")
            if len(tag_kw) >= 2:
                # search 支持 name 参数（名称模糊匹配），与 search_by_name 互补
                tag_hits = index.search(name=tag_kw, limit=5)
                known_ids = {str(r.get("id", "")) for r in results if r.get("id")}
                for r in tag_hits:
                    if str(r.get("id", "")) not in known_ids:
                        results.append(r)
                        known_ids.add(str(r.get("id", "")))
    except Exception as e:
        logger.warning(f"关键词 Metadata 查询失败: {e}")
    return results


def _retrieve_metadata(query: str, plan: dict, search_queries: list[str],
                        existing: list[dict]) -> list[dict]:
    """Metadata Index 查询: 结构化过滤 or 名称/标签搜索"""
    results = list(existing)
    try:
        from agents.metadata_index import index
        filters = _extract_metadata_filters(query, plan)
        if filters:
            results = index.search(**filters)
        else:
            for q in search_queries[:2]:
                md = index.search_by_name(q)
                if not results:
                    results = md
            if plan.get("query_type") in ("recommendation", "comparison"):
                matched_tags = [t for t in _ANIME_TAGS if t in query]
                if matched_tags:
                    tag_results = index.search(tag=matched_tags[0], limit=20)
                    seen_ids = {str(r.get("id", "")) for r in results}
                    for r in tag_results:
                        if str(r.get("id", "")) not in seen_ids:
                            results.append(r)
    except Exception as e:
        logger.warning(f"Metadata Index 查询失败: {e}")
    return results


def _retrieve_semantic(search_queries: list[str], state: dict) -> list[str]:
    """Pinecone + Whoosh 混合检索（向量 + 稀疏 -> Fusion + Rerank）"""
    docs: list[str] = []
    try:
        from tools.registry import tool_registry
        from tools.rag import _get_retriever
        retrieve_opt = tool_registry.get_callable("retrieve_optimized")
        retriever = _get_retriever()
        # 上游 _query_processing_node 已基于 plan.rewrite_strategy 做过决策
        # 只要 query_strategy 字段存在，就跳过 retrieve_optimized 内部的 classify 调用
        # 避免与 planner 决策冲突（如 planner 判 direct 但 classify 重判为 rewrite）
        already_optimized = bool(state.get("query_strategy")) and bool(state.get("optimized_queries"))
        for q in search_queries[:2]:
            if retrieve_opt:
                d, _ = retrieve_opt(
                    q, retriever, k_final=config.RETRIEVER_K,
                    skip_optimization=already_optimized,
                )
                docs.extend(d)
    except Exception as e:
        logger.warning(f"Pinecone/Whoosh 检索失败: {e}")
    return docs


async def _knowledge_retrieval_node(state: AgentState) -> dict:
    """知识检索节点: 根据 query_category 分流检索路径

    metadata  → Metadata Index only（结构化过滤，零 Pinecone）
    semantic  → Pinecone + Whoosh（向量检索 + 稀疏检索  → Fusion + Rerank）
    mixed     → 两者全路检索 + 融合
    """
    t0 = time.time()
    plan = state.get("plan", {})
    query_category = plan.get("query_category", "mixed")
    query = state.get("resolved_query", "") or state.get("original_query", "")
    queries = state.get("shared_context", [query])
    if isinstance(queries, str):
        queries = [queries]

    search_queries = [q for q in queries if isinstance(q, str)]
    if not search_queries:
        search_queries = [query]

    # ① 别名关键词优先查 Metadata Index
    metadata_results = _retrieve_by_keywords(state.get("search_keywords", []))

    # ② Metadata Index 查询（metadata / mixed 两类都走）
    if query_category in ("metadata", "mixed"):
        metadata_results = _retrieve_metadata(query, plan, search_queries, metadata_results)

    # ③ Pinecone + Whoosh 检索（semantic / mixed 两类才走）
    shared_context: list[str] = []
    if query_category in ("semantic", "mixed"):
        shared_context = _retrieve_semantic(search_queries, state)

    logger.info(f"知识检索完成: 返回 metadata {len(metadata_results[:30])} 条, shared_context {len(shared_context[:10])} 条 (耗时 {time.time()-t0:.1f}s)")
    return {
        "metadata": metadata_results[:30],
        "shared_context": shared_context[:10],
    }


def _extract_metadata_filters(query: str, plan: dict) -> dict | None:
    """从查询中提取结构化过滤条件，返回 MetadataIndex.search(**filters) 参数"""
    filters = {}
    q = query

    # 提取标签
    matched_tags = [t for t in _ANIME_TAGS if t in q]
    if matched_tags and plan.get("query_type") in ("recommendation", "simple_fact"):
        filters["tag"] = matched_tags[0]

    # 提取评分范围（预编译正则）
    score_match = _SCORE_RANGE_RE.search(q)
    if score_match:
        val = float(score_match.group(1))
        direction = score_match.group(2)
        if direction in ("以上", "超过", "高于"):
            filters["score_min"] = val
        else:
            filters["score_max"] = val

    # 提取年份（预编译正则）
    year_match = _YEAR_RE.search(q)
    if year_match:
        year = year_match.group(1)
        if "之前" in q or "以前" in q:
            filters["date_to"] = year
        elif "之后" in q or "以后" in q:
            filters["date_from"] = year
        else:
            filters["date_from"] = year
            filters["date_to"] = str(int(year) + 1)

    return filters if filters else None


# ── 路由函数 ──────────────────────────────────────────────────────

def _route_from_start(state: AgentState) -> str:
    """START → alias_resolve (按需) 或直接跳过到 history_extractor"""
    query = _get_query(state)
    if _should_skip_alias(query):
        logger.info(f"  [按需跳过] alias_resolve — 查询无需别名解析")
        return "alias_skip"
    return "alias_resolve"


async def _alias_skip_node(state: AgentState) -> dict:
    """alias_resolve 被跳过时设置必需字段的默认值"""
    query = _get_query(state)
    return {
        "original_query": query,
        "resolved_query": query,
    }

def _route_after_planner(state: AgentState) -> str:
    """Planner 处理完的路由"""
    plan = state.get("plan", {})
    if plan.get("query_type") == "chat":
        return "answer"
    return "query_processing"


def _route_after_retrieval(state: AgentState) -> list[Send | str]:
    """知识检索完后 → 并行分配到 Experts

    ⚠️ 重要: Send 的 arg 会作为目标节点的输入 state，不会自动携带父节点的完整 state。
    必须显式传递 Expert 需要的所有 state 字段，否则 Expert 会收到空 state。
    """
    plan = state.get("plan", {})
    experts = plan.get("experts", [])

    # simple_fact 走快速通道：跳过 Expert + Merge + Answer，一次 LLM 直接回答
    if plan.get("query_type") == "simple_fact":
        return "simple_fact_answer"

    if not experts:
        return "answer_planner"

    # 构建传递给 Expert 的 state（必须包含 Expert 需要的所有字段）
    expert_input = {
        "metadata": state.get("metadata", []),
        "shared_context": state.get("shared_context", []),
        "resolved_query": state.get("resolved_query", ""),
        "original_query": state.get("original_query", ""),
        "plan": state.get("plan", {}),
        "search_keywords": state.get("search_keywords", []),
        "context": state.get("context", {}),
    }

    # 使用 Send API 并行发送，显式传递完整输入
    sends = []
    for expert in experts[:2]:  # 最多 2 个 Expert
        sends.append(Send(expert, expert_input))

    if not sends:
        return "answer_planner"  # 无 Expert → 先规划回答结构，再生成回答

    # 单个 Expert 直接返回节点名，走正常边（state 自动继承）
    if len(sends) == 1:
        return experts[0]

    return sends


def _route_after_expert(state: AgentState) -> str:
    """Expert 处理完 → merge"""
    return "merge"


def _route_after_merge(state: AgentState) -> str:
    """Merge 后 → web_fallback 或 answer_planner"""
    if should_trigger_web(state):
        return "web_fallback"
    return "answer_planner"


def _answer_planner_node(state: AgentState) -> dict:
    """零 LLM 成本的回答结构规划器，随机选结构避免套路化"""
    plan = state.get("plan", {})
    query_type = plan.get("query_type", "recommendation")

    if query_type == "chat":
        return {"answer_plan": {"structure": "简短闲聊"}}

    structures = {
        "recommendation": [
            "top_pick — 先重点安利最推荐的1-2部，多说几句为什么喜欢，后面简略带过",
            "compare — 用对比的方式介绍，突出每部特点，让用户自己选",
            "theme — 按主题/风格归类推荐，先说共同点再展开",
            "honest — 先夸优点再说槽点，显得客观，加一句看你自己口味",
        ],
        "simple_fact": [
            "direct — 直接回答核心问题，顺带讲个相关趣事",
            "expand — 先回答核心问题，再补充1-2个相关维度",
        ],
        "comparison": [
            "vs — 逐项对比，最后一句总结谁更适合什么人",
            "narration — 先分别讲每部特点，最后说更看重X就选A看重Y就选B",
        ],
    }

    options = structures.get(query_type, structures["recommendation"])
    chosen = random.choice(options)

    return {"answer_plan": {"structure": chosen, "tone": "casual"}}


def _get_query(state: dict) -> str:
    """从 state 提取用户查询（优先 messages[-1]，跨轮最可靠）"""
    if state.get("messages"):
        last_msg = state["messages"][-1]
        q = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        if q:
            return q
    return state.get("original_query", "")


# ── 构建图 ────────────────────────────────────────────────────────

def build_graph():
    from tools.registry import register_default_tools
    register_default_tools()

    g = StateGraph(AgentState)

    # 注册节点
    g.add_node("alias_resolve", _alias_resolve_node)
    g.add_node("alias_skip", _alias_skip_node)
    g.add_node("history_extractor", history_extractor_node)
    g.add_node("context_builder", context_builder_node)
    g.add_node("planner", planner_node)
    g.add_node("query_processing", _query_processing_node)
    g.add_node("knowledge_retrieval", _knowledge_retrieval_node)
    g.add_node("metadata_reasoner", metadata_reasoner_node)
    g.add_node("similar_expert", similar_expert_node)
    g.add_node("merge", merge_expert_results)
    g.add_node("simple_fact_answer", simple_fact_answer_node)
    g.add_node("web_fallback", web_fallback_node)
    g.add_node("answer_planner", _answer_planner_node)
    g.add_node("answer", answer_node)

    # ── START 条件边: alias_resolve 按需启用 ──
    g.add_conditional_edges(START, _route_from_start, {
        "alias_resolve": "alias_resolve",
        "alias_skip": "alias_skip",
    })

    # alias → history_extractor（两条路径汇合）
    g.add_edge("alias_resolve", "history_extractor")
    g.add_edge("alias_skip", "history_extractor")

    g.add_edge("history_extractor", "context_builder")
    g.add_edge("context_builder", "planner")

    # planner → query_processing 或 answer (chat)
    g.add_conditional_edges("planner", _route_after_planner, {
        "query_processing": "query_processing",
        "answer": "answer",
    })

    g.add_edge("query_processing", "knowledge_retrieval")

    # knowledge_retrieval → experts (parallel) 或 answer_planner
    g.add_conditional_edges("knowledge_retrieval", _route_after_retrieval, {
        "metadata_reasoner": "metadata_reasoner",
        "similar_expert": "similar_expert",
        "answer_planner": "answer_planner",
        "simple_fact_answer": "simple_fact_answer",
    })

    # experts → merge
    g.add_edge("metadata_reasoner", "merge")
    g.add_edge("similar_expert", "merge")

    # merge → web_fallback 或 answer_planner
    g.add_conditional_edges("merge", _route_after_merge, {
        "web_fallback": "web_fallback",
        "answer_planner": "answer_planner",
    })

    g.add_edge("web_fallback", "answer_planner")
    g.add_edge("answer_planner", "answer")
    g.add_edge("simple_fact_answer", END)
    g.add_edge("answer", END)

    return g
