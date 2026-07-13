"""Entry point — 程序化 API 与交互式终端。

程序化调用:
    from main import run
    result = await run("推荐一部类似命运石之门的科幻番")

交互式终端:
    python main.py              # 默认会话
    python main.py --session my  # 指定会话 ID
"""

import asyncio
import sys
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from graph import build_graph
import config

config.validate()

# 模块级 MemorySaver 池 —— 同一个 thread_id 共享记忆
_memories: dict[str, MemorySaver] = {}


def _get_app(thread_id: str):
    """获取（或创建）thread_id 对应的已编译图实例。"""
    if thread_id not in _memories:
        _memories[thread_id] = MemorySaver()
    g = build_graph()
    return g.compile(checkpointer=_memories[thread_id])


async def run(query: str, thread_id: str = "1") -> str:
    """向 AniGraph 发送一条查询，返回回答文本。

    同一 thread_id 的多次调用共享对话历史。

    Args:
        query:     用户输入文本。
        thread_id: 会话标识符，用于隔离多轮对话。

    Returns:
        模型生成的回答字符串。
    """
    app = _get_app(thread_id)
    resp = await app.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    msgs = resp.get("messages", [])
    return msgs[-1].content if msgs else "(无回答)"


if __name__ == "__main__":
    # 使用 chat.py 的交互式终端
    from chat import main as chat_main
    chat_main()
