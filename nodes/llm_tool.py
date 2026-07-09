"""LLM tool node — 绑定工具并生成工具调用（RAG 检索 / Web 搜索）"""
from langchain_core.messages import SystemMessage, ToolMessage
from state import State
from llms import answer_LLM

_SYSTEM = (
    "你是 ACG 番剧推荐助手。你必须调用工具来获取信息，不要直接回答。\n"
    "- 用户问番剧推荐/评分/导演/制作公司/声优/编剧/相似作品 → 调用 RAG 检索知识库\n"
    "- 用户问最新番剧/当季新番/实时信息 → 调用 search_web 联网搜索\n"
    "根据用户问题选择合适的工具调用，不要输出文字，只输出工具调用。"
)

async def llm_tool_node(state: State, tools: list):
    if isinstance(state["messages"][-1], ToolMessage):
        return {"messages": [], "iteration_count": state.get("iteration_count", 0)}

    resp = await answer_LLM.bind_tools(tools).ainvoke(
        [SystemMessage(content=_SYSTEM)] + state["messages"][-10:]
    )
    return {"messages": [resp], "iteration_count": state.get("iteration_count", 0) + 1}
