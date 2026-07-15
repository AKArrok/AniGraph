"""Tools — 统一通过 ToolRegistry 注册和管理。

向后兼容: RAG / search_web 仍可直接 import。
新增: tool_registry 提供 get_llm_tools() / is_enabled() 等统一接口。
"""
from tools.registry import tool_registry, ToolSpec, register_default_tools
from tools.rag import RAG
from tools.web_search import search_web

__all__ = [
    "RAG", "search_web",
    "tool_registry", "ToolSpec", "register_default_tools",
]
