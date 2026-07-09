"""Answer node — 综合多维度推荐结果生成最终回答"""
from langchain_core.messages import SystemMessage, ToolMessage
from state import State
from llms import answer_LLM

_PROMPT = """You are an ACG anime recommendation expert.

Response rules:
- Chinese query → reply in Chinese | Roman Urdu → Roman Urdu | English → English
- Plain text only, no markdown formatting

When recommending anime from tool results:
1. List 3-5 recommendations, each with: title, rating, and why it matches
2. Mention matching tags/genres (e.g., "科幻、时间旅行、悬疑")
3. If the user asked for a specific genre/director/seiyuu, highlight that match
4. Keep each recommendation to 1-2 lines, total 5-12 lines

When no tool results are found:
- "抱歉，知识库中没有找到符合你要求的番剧，可以换个关键词试试？"

General tone: enthusiastic, knowledgeable, concise."""


def _get_tool_output(msgs):
    for m in reversed(msgs):
        if isinstance(m, ToolMessage):
            out = m.content
            if isinstance(out, list):
                return "\n".join(i.get("text", "") for i in out if i.get("type") == "text").strip()
            return out
    return None


async def answer_node(state: State):
    msgs     = state["messages"]
    tool_out = _get_tool_output(msgs)
    prompt   = _PROMPT + (f"\n\nTool result:\n{tool_out}" if tool_out else "")
    history  = [m for m in msgs[-10:] if not isinstance(m, ToolMessage)]
    return {"messages": [await answer_LLM.ainvoke([SystemMessage(content=prompt)] + history)]}
