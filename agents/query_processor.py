"""Query Processor — 查询优化节点，从 graph.py 拆出。"""
import config
from agents.state import AgentState


async def query_processing_node(state: AgentState) -> dict:
    """查询处理节点: 根据 plan.rewrite_strategy 执行 Rewrite/HyDE/Decompose/Direct"""
    plan = state.get("plan", {})
    strategy = plan.get("rewrite_strategy", "rewrite")
    if getattr(config, "ABLATION_NO_QUERY_REWRITE", False):
        strategy = "direct"
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
