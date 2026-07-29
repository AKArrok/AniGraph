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
from agents.retrieval import knowledge_retrieval_node
from agents.answer_planner import answer_planner_node
from agents.query_processor import query_processing_node
from agents.alias_resolve import alias_resolve_node, alias_skip_node, should_skip_alias
import config

logger = logging.getLogger(__name__)


# ── 路由函数 ─────────────────────────────────────────────────────


def _route_from_start(state: AgentState) -> str:
    """START -> image_recognition (有图) 或 alias_resolve/alias_skip (无图)"""
    if config.ENABLE_IMAGE_RECOGNITION and state.get("image_data"):
        return "image_recognition"
    query = get_query(state)
    if should_skip_alias(query):
        logger.info(f"  [按需跳过] alias_resolve - 查询无需别名解析")
        return "alias_skip"
    return "alias_resolve"


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
    if plan.get("query_type") == "simple_fact" and not getattr(config, "ABLATION_NO_FAST_PATH", False):
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


def get_query(state: dict) -> str:
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
    g.add_node("alias_resolve", alias_resolve_node)
    g.add_node("alias_skip", alias_skip_node)
    g.add_node("image_recognition", image_recognition_node)
    g.add_node("history_extractor", history_extractor_node)
    g.add_node("context_builder", context_builder_node)
    g.add_node("planner", planner_node)
    g.add_node("query_processing", query_processing_node)
    g.add_node("knowledge_retrieval", knowledge_retrieval_node)
    g.add_node("metadata_reasoner", metadata_reasoner_node)
    g.add_node("similar_expert", similar_expert_node)
    g.add_node("serial_expert", _serial_expert_node)
    g.add_node("merge", merge_expert_results)
    g.add_node("evaluator", evaluator_node)
    g.add_node("replanner", replanner_node)
    g.add_node("simple_fact_answer", simple_fact_answer_node)
    g.add_node("web_fallback", web_fallback_node)
    g.add_node("answer_planner", answer_planner_node)
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
