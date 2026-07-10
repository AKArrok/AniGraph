"""Answer node — 综合多维度推荐结果生成最终回答"""
from langchain_core.messages import SystemMessage, ToolMessage
from state import State
from llms import answer_LLM, simple_LLM

_PROMPT = """你是资深二次元，像跟朋友聊天一样推荐番剧。用口语化中文，避免套路化的格式。

注意:
- 别用固定模板（不要每部都"评分：X / 推荐理由：Y"），换着花样说
- 提到类型标签时自然地融入句子，别单独列出来
- 有热门评论就引用，像"Bangumi上有人说：'xxx'"这样自然带出来
- 整体像真人在聊天，别像机器生成报告
- 篇幅随性一点，别刻意追求行数

没结果时就老实说没找到，可以问问对方想不想换关键词。

语气: 口语化、热情、像个活人。"""

_SIMPLE_PROMPT = """你是 ACG 番剧小助手，回答用户的简单查询。像漫友之间聊天一样，语气可以轻松活泼一点。基于工具返回的事实信息来回答，把细节说得有趣些——评分多少、谁配音的、哪个公司做的、有什么看点，都展开聊聊，别干巴巴一句话带过。也可以引用热门评论来丰富回答。

语气: 轻松、有料、像朋友在安利。"""


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
    is_simple = state.get("is_simple", False)

    if is_simple:
        prompt = _SIMPLE_PROMPT + (f"\n\n工具返回:\n{tool_out}" if tool_out else "")
    else:
        prompt = _PROMPT + (f"\n\nTool result:\n{tool_out}" if tool_out else "")

    llm = simple_LLM if is_simple else answer_LLM
    history = [m for m in msgs[-10:] if not isinstance(m, ToolMessage)]
    return {"messages": [await llm.ainvoke([SystemMessage(content=prompt)] + history)]}
