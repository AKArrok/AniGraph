"""两级 Metadata Cache — Alias Cache → Metadata Cache，命中率 > 90%"""
from __future__ import annotations
from typing import Any


class MetadataCache:
    """L1: Alias Cache → L2: Metadata Cache

    L1 存储 {alias_lower: full_name} 映射
    L2 存储 {full_name: metadata_dict} 完整元数据

    常用番剧的别名和元数据始终在内存中，零查询延迟。
    """

    def __init__(self, max_size: int = 2000):
        self._alias_cache: dict[str, str] = {}    # {alias_lower: full_name}
        self._metadata_cache: dict[str, dict] = {}  # {full_name: metadata_dict}
        self._max_size = max_size

    # ── L1: Alias Cache ──────────────────────────────────────────

    def add_alias(self, alias: str, full_name: str):
        """添加别名映射"""
        key = alias.strip().lower()
        self._alias_cache[key] = full_name
        if len(self._alias_cache) > self._max_size:
            # 简单 FIFO 淘汰
            self._alias_cache.pop(next(iter(self._alias_cache)))

    def bulk_add_alias(self, entries: list[tuple[str, str]]):
        """批量添加别名映射 [(alias, full_name), ...]"""
        for alias, full_name in entries:
            self.add_alias(alias, full_name)

    def resolve_alias(self, name: str) -> str | None:
        """别名精确匹配 → full_name，未命中返回 None"""
        return self._alias_cache.get(name.strip().lower())

    # ── L2: Metadata Cache ───────────────────────────────────────

    def add_metadata(self, full_name: str, metadata: dict):
        """缓存完整元数据"""
        key = full_name.strip()
        self._metadata_cache[key] = metadata
        if len(self._metadata_cache) > self._max_size:
            self._metadata_cache.pop(next(iter(self._metadata_cache)))

    def bulk_load_metadata(self, items: list[dict]):
        """从 metadata_index 批量加载到缓存"""
        for item in items:
            name = item.get("name_cn") or item.get("name") or ""
            if name:
                self.add_metadata(name, item)
            # 同时用日文名做 key
            jp_name = item.get("name", "")
            if jp_name and jp_name != name:
                self.add_metadata(jp_name, item)

    def get_metadata(self, full_name: str) -> dict | None:
        """精确名称 → 元数据，未命中返回 None"""
        return self._metadata_cache.get(full_name.strip())

    # ── 联合查询 ─────────────────────────────────────────────────

    def resolve(self, query: str) -> tuple[str, dict | None]:
        """两级联合查询: alias → metadata

        Returns:
            (resolved_name, metadata_dict | None)
            未命中返回 (query, None)
        """
        # L1: 别名解析
        full_name = self.resolve_alias(query)
        if not full_name:
            return query, None

        # L2: 元数据查询
        meta = self.get_metadata(full_name)
        return full_name, meta

    # ── 状态 ─────────────────────────────────────────────────────

    @property
    def alias_count(self) -> int:
        return len(self._alias_cache)

    @property
    def metadata_count(self) -> int:
        return len(self._metadata_cache)

    def get_state(self) -> dict[str, dict]:
        """序列化当前缓存状态（用于 langgraph checkpoint）"""
        return {
            "alias_cache": dict(self._alias_cache),
            "metadata_cache": dict(self._metadata_cache),
        }


# 全局单例
metadata_cache = MetadataCache()
