"""Tools - 统一通过 ToolRegistry 注册和管理。

向后兼容: RAG / search_web 仍可直接 import，但仅在访问时加载。
新增: tool_registry 提供 get_llm_tools() / is_enabled() 等统一接口。
"""
from tools.registry import tool_registry, ToolSpec, register_default_tools

__all__ = [
    "RAG", "search_web",
    "tool_registry", "ToolSpec", "register_default_tools",
]


def __getattr__(name: str):
    """惰性加载兼容导出，避免导入 tools 时初始化模型或联网客户端。"""
    if name == "RAG":
        from tools.rag import RAG
        return RAG
    if name == "search_web":
        from tools.web_search import search_web
        return search_web
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
