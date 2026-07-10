"""History Extractor — 从 messages 中提取最近 N 轮对话

职责单一: 只提取历史，不做任何业务推理。
"""
from langchain_core.messages import HumanMessage, AIMessage
import config
from agents.state import AgentState


def _extract_recent_rounds(messages: list, n: int) -> list[dict]:
    """从 messages 中提取最近 N 轮用户+助手配对"""
    if not messages:
        return []

    # 过滤出 HumanMessage 和 AIMessage，按顺序配对
    paired = []
    user_msg = None

    for m in messages:
        if isinstance(m, HumanMessage):
            user_msg = m.content
        elif isinstance(m, AIMessage) and user_msg is not None:
            paired.append({"user": user_msg, "assistant": m.content})
            user_msg = None

    # 只保留最近 N 轮
    return paired[-n:] if len(paired) > n else paired


async def history_extractor_node(state: AgentState) -> dict:
    """从 messages 中提取最近 N 轮对话，存入 context.history"""
    messages = state.get("messages", [])
    n = config.MEMORY_MAX_ROUNDS

    rounds = _extract_recent_rounds(messages, n)

    return {"context": {"history": rounds}}
