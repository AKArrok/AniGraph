"""Session Store — 统一的会话记忆管理，chat.py / main.py / server.py 共用。

用法:
    store = SessionStore()
    app = store.get_app(thread_id)       # 获取或创建编译后的 graph
    store.clear(thread_id)                # 清空单会话记忆
    store.clear_all()                     # 清空全部
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from agents.graph import build_graph


class SessionStore:
    """线程安全的会话图实例池，同一 thread_id 共享 MemorySaver。"""

    def __init__(self) -> None:
        self._apps: dict[str, MemorySaver] = {}
        self._graphs: dict[str, object] = {}

    # ── public API ──────────────────────────────────────────

    def get_app(self, thread_id: str):
        """获取或创建 thread_id 对应的已编译图实例。"""
        if thread_id not in self._apps:
            self._apps[thread_id] = MemorySaver()
            g = build_graph()
            self._graphs[thread_id] = g.compile(checkpointer=self._apps[thread_id])
        return self._graphs[thread_id]

    def clear(self, thread_id: str) -> None:
        """清空单个会话的记忆，保留编译后的图结构。"""
        self._apps.pop(thread_id, None)
        self._graphs.pop(thread_id, None)

    def clear_all(self) -> None:
        """清空全部会话。"""
        self._apps.clear()
        self._graphs.clear()

    def session_count(self) -> int:
        """活跃会话数。"""
        return len(self._apps)


# 模块级默认实例（供 main.py 和 server.py 共享）
default_store = SessionStore()
