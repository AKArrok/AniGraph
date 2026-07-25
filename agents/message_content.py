"""Helpers for reading text and image blocks from LangChain messages."""

from typing import Any

from langchain_core.messages import HumanMessage


def message_text(content: Any) -> str:
    """Extract text from plain or OpenAI-compatible multimodal content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def latest_user_message(messages: list[Any]) -> HumanMessage | None:
    """Return the latest user message, ignoring trailing AI/tool messages."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None


def has_image_block(message: Any) -> bool:
    """Return whether a message contains an image URL content block."""
    content = getattr(message, "content", None)
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "image_url"
        for block in content
    )
