"""多 Agent 协作图 — 基于 ExecutionPlan 动态编排

流程:
  START → alias_resolve → planner → query_processing → knowledge_retrieval
    → [metadata_reasoner || similar_expert] (parallel via Send)
    → merge → web_fallback? → answer_planner → answer → END
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


async def _query_processing_node(state: AgentState) -> dict:
    """查询处理节点: 根据 plan.rewrite_strategy 执行 Rewrite/HyDE/Decompose/Direct"""
    plan = state.get("plan", {})
    strategy = plan.get("rewrite_strategy", "rewrite")
    query = state.get("resolved_query", "") or state.get("original_query", "")

    from tools.query_processing import (
        classify, multi_query_rewrite, hyde_generate, decompose,
    )

    if strategy == "direct":
        queries = [query]
    elif strategy == "hyde":
        queries = hyde_generate(query)
    elif strategy == "decompose":
        queries = decompose(query)
    else:  # rewrite
        queries = multi_query_rewrite(query)

    return {
        "shared_context": queries,          # 暂存优化的 queries
        "optimized_queries": queries,       # 标记：上游已做查询优化
        "query_strategy": strategy,         # 标记：使用的优化策略
    }


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

    metadata_results = []
    shared_context = []

    # ── 别名提取的关键词优先查 Metadata Index ──
    keywords = state.get("search_keywords", [])
    if keywords:
        try:
            from agents.metadata_index import index
            for kw in keywords[:3]:
                md = index.search_by_name(kw)
                if md:
                    metadata_results.extend(md)
            # tags 也从关键词推导
            for kw in keywords[:3]:
                tag_kw = kw.strip("【】！! ")
                if len(tag_kw) >= 2:
                    tag_hits = index.search(name=tag_kw, limit=5)
                    known_ids = {str(r.get("id", "")) for r in metadata_results if r.get("id")}
                    for r in tag_hits:
                        if str(r.get("id", "")) not in known_ids:
                            metadata_results.append(r)
                            known_ids.add(str(r.get("id", "")))
        except Exception as e:
            logger.warning(f"关键词 Metadata 查询失败: {e}")

    # ── Metadata Index 查询（metadata / mixed 两类都走）──
    if query_category in ("metadata", "mixed"):
        try:
            from agents.metadata_index import index

            # 提取结构化过滤条件
            filters = _extract_metadata_filters(query, plan)

            if filters:
                # 精确多维过滤
                filter_results = index.search(**filters)
                metadata_results = filter_results
            else:
                # 按名称搜索（优先中文名）
                for q in search_queries[:2]:
                    md = index.search_by_name(q)
                    if not metadata_results:
                        metadata_results = md

                # 按标签搜索
                if plan.get("query_type") in ("recommendation", "comparison"):
                    tag_keywords = ["热血", "动作", "搞笑", "异世界", "奇幻", "科幻", "恋爱", "日常",
                                    "治愈", "悬疑", "推理", "战斗", "冒险", "校园", "机战", "运动"]
                    matched_tags = [t for t in tag_keywords if t in query]
                    if matched_tags:
                        tag_results = index.search(tag=matched_tags[0], limit=20)
                        seen_ids = {str(r.get("id", "")) for r in metadata_results}
                        for r in tag_results:
                            if str(r.get("id", "")) not in seen_ids:
                                metadata_results.append(r)
        except Exception as e:
            logger.warning(f"Metadata Index 查询失败: {e}")

    # ── Pinecone + Whoosh 检索（semantic / mixed 两类才走）──
    if query_category in ("semantic", "mixed"):
        try:
            from tools.rag_optimizer import retrieve_with_optimization
            from tools.rag import _get_retriever

            retriever = _get_retriever()
            # 检测上游是否已做查询优化，避免 rag_optimizer 二次改写
            already_optimized = state.get("query_strategy") in ("rewrite", "hyde", "decompose") \
                                and bool(state.get("optimized_queries"))
            for q in search_queries[:2]:
                docs, _ = retrieve_with_optimization(
                    q, retriever, k_final=config.RETRIEVER_K,
                    skip_optimization=already_optimized,
                )
                shared_context.extend(docs)
        except Exception as e:
            logger.warning(f"Pinecone/Whoosh 检索失败: {e}")

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
    tag_list = ["热血", "动作", "搞笑", "异世界", "奇幻", "科幻", "恋爱", "日常",
              "治愈", "悬疑", "推理", "战斗", "冒险", "校园", "机战", "运动",
              "魔法", "后宫", "百合", "耽美", "美食", "音乐", "运动", "竞技",
              "战争", "历史", "恐怖", "职场", "偶像", "转生", "游戏"]
    matched_tags = [t for t in tag_list if t in q]
    if matched_tags and plan.get("query_type") in ("recommendation", "simple_fact"):
        filters["tag"] = matched_tags[0]

    # 提取评分范围
    score_match = re.search(r"(\d[\d.]*)\s*分?\s*(以上|以下|以上|超过|高于|低于)", q)
    if score_match:
        val = float(score_match.group(1))
        direction = score_match.group(2)
        if direction in ("以上", "超过", "高于"):
            filters["score_min"] = val
        else:
            filters["score_max"] = val

    # 提取年份
    year_match = re.search(r"(20\d{2})", q)
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
    g = StateGraph(AgentState)

    # 注册节点
    g.add_node("alias_resolve", _alias_resolve_node)
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

    # 边
    g.add_edge(START, "alias_resolve")
    g.add_edge("alias_resolve", "history_extractor")
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
