"""验证 entity_resolver 与 planner 集成的完整链路"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from agents.graph import _alias_resolve_node, _get_query
from agents.planner import planner_node

async def test():
    # 模拟最小图
    from agents.graph import build_graph
    g = build_graph()

    tests = [
        ("夏亚是谁", "character"),
        ("典明粥是什么梗", "meme"),
        ("推荐类似进击的巨人的番", "alias"),  # '巨人' 包含匹配
    ]

    for query, exp_entity_type in tests:
        # 模拟完整调用
        state = {"messages": [HumanMessage(content=query)], "original_query": query}
        result = await _alias_resolve_node(state)
        state.update(result)

        print(f"Query: {query}")
        print(f"  entity_type={state.get('entity_type','?')} entity_confidence={state.get('entity_confidence',1.0):.0%} entity_source={state.get('entity_source','?')}")
        print(f"  search_keywords={state.get('search_keywords','?')}")

        # 走 planner
        plan_result = await planner_node(state)
        plan = plan_result.get("plan", {})
        print(f"  need_web={plan.get('need_web', '?')}")
        print()

asyncio.run(test())
