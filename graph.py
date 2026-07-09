"""Graph assembly — 动态路由工作流"""
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from state import State
from nodes import domain_router_node, dimension_expert_node, llm_tool_node, answer_node
from tools import RAG, search_web
import config

DIM_NODES = ["rating", "genre", "director", "studio", "writer", "seiyuu", "similar", "general"]


# ── 路由函数 ──────────────────────────────────────────────────────

def _route_from_router(state: State) -> str:
    """domain_router 处理完 → 路由到 primary 维度专家"""
    active = state.get("active_dimension", "general")
    return active if active in DIM_NODES else "general"


def _route_after_expert(state: State) -> str:
    """维度专家处理完 → 下一个未处理维度 或 进入工具调用"""
    dims = state.get("dimensions", ["general"])
    processed = state.get("processed_dims", [])
    for d in dims:
        if d not in processed:
            return d
    return "llm_tool"


def _tool_or_answer(s):
    """LLM 工具节点 → 需要再调用工具？还是生成回答"""
    last = s["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls and s.get("iteration_count", 0) < config.MAX_ITERATIONS:
        return "tools"
    return "answer"


# ── 图谱构建 ──────────────────────────────────────────────────────

def build_graph():
    tools = [RAG, search_web]

    g = StateGraph(State)

    # 注册节点
    g.add_node("domain_router", domain_router_node)

    for dim in DIM_NODES:
        async def _make_expert(s: State, d=dim):
            # 标记当前维度为已处理
            processed = list(s.get("processed_dims", []))
            if d not in processed:
                processed.append(d)
            result = await dimension_expert_node(s, d)
            result["processed_dims"] = processed
            return result
        g.add_node(dim, _make_expert)

    async def _llm_tool(s: State):
        return await llm_tool_node(s, tools)
    g.add_node("llm_tool", _llm_tool)
    g.add_node("tools",    ToolNode(tools))
    g.add_node("answer",   answer_node)

    # 边
    g.add_edge(START, "domain_router")
    g.add_conditional_edges("domain_router", _route_from_router, DIM_NODES)

    # 每个维度专家 → 动态决定下一个节点
    for dim in DIM_NODES:
        g.add_conditional_edges(dim, _route_after_expert, DIM_NODES + ["llm_tool"])

    g.add_conditional_edges("llm_tool", _tool_or_answer, ["tools", "answer"])
    g.add_edge("tools", "answer")
    g.add_edge("answer", END)

    return g, tools
