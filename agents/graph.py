"""多 Agent 协作图 — 基于 ExecutionPlan 动态编排

流程:
  START → [alias_resolve]? → history_extractor → context_builder → planner
    → query_processing → knowledge_retrieval
    → [metadata_reasoner || similar_expert] (parallel via Send or serial)
    → merge → evaluator → [replanner → query_processing]? → answer → END
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
from agents.evaluator import evaluator_node
from agents.replanner import replanner_node
from agents.image_recognition import image_recognition_node
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

    additional = state.get("replan_feedback", {}).get("additional_queries", [])
    queries = list(dict.fromkeys([
        q for q in [*queries, *additional]
        if isinstance(q, str) and q.strip()
    ]))

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
        already_optimized = bool(state.get("query_strategy")) and bool(state.get("optimized_queries"))
        plan = state.get("plan", {})
        k_final = config.RETRIEVER_K
        if plan.get("query_type") == "recommendation":
            k_final = max(k_final, 10)
        context = state.get("context", {})
        constraints = context.get("constraints", {}) if isinstance(context, dict) else {}
        query_count = 3 if constraints.get("exclude_same_series") else 2
        for q in search_queries[:query_count]:
            if retrieve_opt:
                d, _ = retrieve_opt(
                    q, retriever, k_final=k_final,
                    skip_optimization=True,
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

    # 同系列排除时，用主题标签扩展检索，避免结果全是主题自身系列
    context = state.get("context", {})
    constraints = context.get("constraints", {}) if isinstance(context, dict) else {}
    if constraints.get("exclude_same_series") and constraints.get("topic_tags"):
        tag_queries = [f"{tag} 动画 推荐" for tag in constraints["topic_tags"][:3]]
        search_queries = list(dict.fromkeys([*search_queries, *tag_queries]))

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
    experts = list(dict.fromkeys(plan.get("experts", [])))
    mode = "single" if len(experts) <= 1 else ("parallel" if plan.get("parallel") else "serial")
    return {
        "metadata": metadata_results[:30],
        "shared_context": shared_context[:10],
        "execution_mode": mode,
        "current_expert_index": 0,
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
    """START -> image_recognition (有图) 或 alias_resolve/alias_skip (无图)"""
    if config.ENABLE_IMAGE_RECOGNITION and state.get("image_data"):
        return "image_recognition"
    query = _get_query(state)
    if _should_skip_alias(query):
        logger.info(f"  [按需跳过] alias_resolve - 查询无需别名解析")
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


def _expert_input(state: AgentState, include_results: bool = False) -> dict:
    """Build an explicit Expert input for Send or serial execution."""
    payload = {
        "metadata": state.get("metadata", []),
        "shared_context": state.get("shared_context", []),
        "resolved_query": state.get("resolved_query", ""),
        "original_query": state.get("original_query", ""),
        "plan": state.get("plan", {}),
        "search_keywords": state.get("search_keywords", []),
        "context": state.get("context", {}),
        "execution_id": state.get("execution_id", ""),
        "attempt": state.get("attempt", 0),
        "recommendation_count": state.get("recommendation_count", 0),
    }
    if include_results:
        payload["expert_results"] = state.get("expert_results", [])
    return payload


def _route_after_retrieval(state: AgentState) -> list[Send] | str:
    """Route Experts according to the strict plan.parallel contract.

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

    experts = list(dict.fromkeys(experts[:2]))
    if len(experts) == 1:
        return experts[0]

    if not plan.get("parallel", False):
        return "serial_expert"

    expert_input = _expert_input(state)
    return [Send(expert, expert_input) for expert in experts]


async def _serial_expert_node(state: AgentState) -> dict:
    """Run one planned Expert and expose its result to the next Expert."""
    experts = list(dict.fromkeys(state.get("plan", {}).get("experts", [])))[:2]
    index = state.get("current_expert_index", 0)
    if index >= len(experts):
        return {}
    expert = experts[index]
    expert_state = _expert_input(state, include_results=True)
    if expert == "metadata_reasoner":
        result = await metadata_reasoner_node(expert_state)
    elif expert == "similar_expert":
        result = await similar_expert_node(expert_state)
    else:
        logger.warning("Ignoring unknown Expert in serial plan: %s", expert)
        result = {"expert_results": []}
    return {**result, "current_expert_index": index + 1}


def _route_after_serial_expert(state: AgentState) -> str:
    experts = list(dict.fromkeys(state.get("plan", {}).get("experts", [])))[:2]
    if state.get("current_expert_index", 0) < len(experts):
        return "serial_expert"
    return "merge"


def _is_recommendation_fast_path(state: AgentState) -> bool:
    plan = state.get("plan", {})
    experts = list(dict.fromkeys(plan.get("experts", [])))
    return plan.get("query_type") == "recommendation" and experts == ["similar_expert"]


def _route_after_similar_expert(state: AgentState) -> str:
    """Single-expert recommendations already have evidence; generate once and stream."""
    if _is_recommendation_fast_path(state):
        return "answer_planner"
    return "merge"


def _route_after_merge(state: AgentState) -> str:
    """Merge is always followed by evidence evaluation."""
    return "evaluator"


def _route_after_evaluator(state: AgentState) -> str:
    evaluation = state.get("evaluation", {})
    verdict = evaluation.get("verdict", "replan")
    if verdict == "pass":
        if should_trigger_web(state):
            return "web_fallback"
        return "answer_planner"
    if verdict == "fallback":
        return "web_fallback"
    if state.get("attempt", 0) < state.get("max_replans", 1):
        return "replanner"
    logger.info("replan_exhausted attempt=%s issues=%s", state.get("attempt", 0), evaluation.get("issues", []))
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

    recommendation_count = state.get("recommendation_count", 0)
    if query_type == "recommendation" and recommendation_count:
        chosen = (
            f"优先推荐约 {recommendation_count} 部作品；每部说明推荐理由；"
            "重点保证相关性，不要堆砌备选"
        )

    return {"answer_plan": {
        "structure": chosen,
        "tone": "casual",
        "recommendation_count": recommendation_count,
    }}


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
    g.add_node("image_recognition", image_recognition_node)
    g.add_node("history_extractor", history_extractor_node)
    g.add_node("context_builder", context_builder_node)
    g.add_node("planner", planner_node)
    g.add_node("query_processing", _query_processing_node)
    g.add_node("knowledge_retrieval", _knowledge_retrieval_node)
    g.add_node("metadata_reasoner", metadata_reasoner_node)
    g.add_node("similar_expert", similar_expert_node)
    g.add_node("serial_expert", _serial_expert_node)
    g.add_node("merge", merge_expert_results)
    g.add_node("evaluator", evaluator_node)
    g.add_node("replanner", replanner_node)
    g.add_node("simple_fact_answer", simple_fact_answer_node)
    g.add_node("web_fallback", web_fallback_node)
    g.add_node("answer_planner", _answer_planner_node)
    g.add_node("answer", answer_node)

    # ── START 条件边: image_recognition / alias_resolve / alias_skip ──
    g.add_conditional_edges(START, _route_from_start, {
        "image_recognition": "image_recognition",
        "alias_resolve": "alias_resolve",
        "alias_skip": "alias_skip",
    })

    # image_recognition / alias -> history_extractor（三条路径汇合）
    g.add_edge("image_recognition", "history_extractor")
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
        "serial_expert": "serial_expert",
        "answer_planner": "answer_planner",
        "simple_fact_answer": "simple_fact_answer",
    })

    # Experts normally enter the quality loop. A single recommendation candidate
    # organizer can go straight to the sole prose-generating answer call.
    g.add_edge("metadata_reasoner", "merge")
    g.add_conditional_edges("similar_expert", _route_after_similar_expert, {
        "answer_planner": "answer_planner",
        "merge": "merge",
    })

    # Serial controller loops until every planned Expert has completed.
    g.add_conditional_edges("serial_expert", _route_after_serial_expert, {
        "serial_expert": "serial_expert",
        "merge": "merge",
    })

    # merge → evaluator → replan / web fallback / answer
    g.add_conditional_edges("merge", _route_after_merge, {
        "evaluator": "evaluator",
    })
    g.add_conditional_edges("evaluator", _route_after_evaluator, {
        "replanner": "replanner",
        "web_fallback": "web_fallback",
        "answer_planner": "answer_planner",
    })
    g.add_edge("replanner", "query_processing")

    g.add_edge("web_fallback", "answer_planner")
    g.add_edge("answer_planner", "answer")
    g.add_edge("simple_fact_answer", END)
    g.add_edge("answer", END)

    return g
