"""Router node — 保留兼容（新架构使用 domain_router）"""
from langchain_core.messages import SystemMessage, HumanMessage
from state import State, DimensionDecision
from llms import router_LLM

_PROMPT = """Classify the query into one of:
- web_search : latest, news, today, current season, 新番, real-time info
- rag        : anime recommendations, 推荐番剧, genre/tag queries, director/seiyuu queries
- direct     : greetings, general knowledge, simple questions

Anime-related queries → rag. Real-time/latest info → web_search. Otherwise → direct.
Output in JSON format."""


async def router_node(state: State):
    try:
        r = await router_LLM.with_structured_output(DimensionDecision, method="function_calling").ainvoke([
            SystemMessage(content=_PROMPT),
            HumanMessage(content=state["messages"][-1].content)
        ])
        return {
            "router_decision": r.primary,
            "reasoning": r.reasoning,
            "dimensions": r.dimensions,
            "active_dimension": r.primary,
        }
    except Exception as e:
        return {
            "router_decision": "general",
            "reasoning": str(e),
            "dimensions": ["general"],
            "active_dimension": "general",
        }


def route_condition(state: State) -> str:
    return "llm_tool" if state["router_decision"] != "general" else "answer"
