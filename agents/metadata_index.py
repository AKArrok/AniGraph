"""Metadata Index — SQLite/JSON 结构化索引，零 LLM 查询

支持: 标签/制作公司/导演/编剧/声优/评分范围/日期范围 过滤
查询方式: search(**filters) | get_by_alias(name) | get_by_id(sid)
"""
from __future__ import annotations
import json
import os
import sqlite3
import config
from typing import Any


class MetadataIndex:
    """ACG 番剧结构化索引，支持多维过滤查询"""

    def __init__(self, index_path: str | None = None):
        self._index_path = index_path or config.METADATA_INDEX_PATH
        self._data: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._loaded = False

    # ── 加载 ─────────────────────────────────────────────────────

    def load(self):
        """从 JSON 文件加载索引"""
        if self._loaded:
            return

        if os.path.exists(self._index_path):
            with open(self._index_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = []

        # 构建 ID 索引
        self._by_id = {}
        for item in self._data:
            sid = str(item.get("id", ""))
            if sid:
                self._by_id[sid] = item

        self._loaded = True

    def reload(self):
        """重新加载索引（知识库更新后调用）"""
        self._loaded = False
        self._data = []
        self._by_id = {}
        self.load()

    # ── 查询 ─────────────────────────────────────────────────────

    def get_by_id(self, sid: str) -> dict | None:
        """通过 bangumi ID 精确查询"""
        if not self._loaded:
            self.load()
        return self._by_id.get(str(sid))

    def get_by_alias(self, name: str) -> dict | None:
        """别名精确匹配 → 直接返回元数据"""
        if not self._loaded:
            self.load()
        n = name.strip().lower()
        for item in self._data:
            # 中文名
            if item.get("name_cn", "").lower() == n:
                return item
            # 日文名
            if item.get("name", "").lower() == n:
                return item
            # 别名列表
            for alias in item.get("alias", []):
                if alias.lower() == n:
                    return item
        return None

    def search_by_name(self, name: str) -> list[dict]:
        """名称模糊搜索（双向包含匹配）

        - 正向: 搜索词是番剧名的子串（如搜"进击"命中"进击的巨人"）
        - 反向: 番剧名是搜索词的子串（如搜"进击的巨人出了几季"命中"进击的巨人"）
        """
        if not self._loaded:
            self.load()
        n = name.strip().lower()
        results = []
        for item in self._data:
            cn = item.get("name_cn", "").lower()
            jp = item.get("name", "").lower()
            # 正向: 搜索词 ⊆ 番剧名
            if n in cn or n in jp:
                results.append(item)
                continue
            # 反向: 番剧名 ⊆ 搜索词（番剧名至少 2 字，避免单字误匹配）
            if cn and len(cn) >= 2 and cn in n:
                results.append(item)
                continue
            if jp and len(jp) >= 2 and jp in n:
                results.append(item)
                continue
            for alias in item.get("alias", []):
                al = alias.lower()
                if n in al:
                    results.append(item)
                    break
                if len(al) >= 2 and al in n:
                    results.append(item)
                    break
        return results

    def search(self, **filters) -> list[dict]:
        """多维过滤查询

        支持的过滤条件:
            tag: str          — 标签包含匹配
            tags: list[str]   — 同时包含所有标签
            studio: str       — 制作公司
            director: str     — 导演
            writer: str       — 编剧
            seiyuu: str       — 声优
            score_min: float  — 最低评分
            score_max: float  — 最高评分
            date_from: str    — 最早日期 (YYYY-MM-DD)
            date_to: str      — 最晚日期 (YYYY-MM-DD)
            kind: str         — 类型 (TV/剧场版/OVA 等)
            limit: int        — 返回数量上限
        """
        if not self._loaded:
            self.load()

        results = self._data[:]
        limit = filters.pop("limit", 50)

        # 标签过滤
        tag = filters.pop("tag", None)
        tags = filters.pop("tags", None)
        all_tags = []
        if tag:
            all_tags.append(tag)
        if tags:
            all_tags.extend(tags)
        if all_tags:
            all_tags_lower = [t.lower() for t in all_tags]
            results = [
                item for item in results
                if self._match_all_tags(item, all_tags_lower)
            ]

        # 字符串字段模糊匹配
        for field in ["studio", "director", "writer", "seiyuu", "kind"]:
            value = filters.pop(field, None)
            if value:
                v = value.lower()
                results = [
                    item for item in results
                    if self._field_contains(item, field, v)
                ]

        # 评分范围
        score_min = filters.pop("score_min", None)
        score_max = filters.pop("score_max", None)
        if score_min is not None:
            results = [item for item in results if _safe_float(item.get("score", 0)) >= score_min]
        if score_max is not None:
            results = [item for item in results if _safe_float(item.get("score", 10)) <= score_max]

        # 日期范围
        date_from = filters.pop("date_from", None)
        date_to = filters.pop("date_to", None)
        if date_from:
            results = [item for item in results if (item.get("date") or "") >= date_from]
        if date_to:
            results = [item for item in results if (item.get("date") or "") <= date_to]

        # 排序：默认按评分降序
        results.sort(key=lambda x: _safe_float(x.get("score", 0)), reverse=True)

        return results[:limit]

    # ── 辅助 ─────────────────────────────────────────────────────

    @staticmethod
    def _match_all_tags(item: dict, tags_lower: list[str]) -> bool:
        item_tags = [t.lower() for t in item.get("tags", [])]
        return all(t in item_tags for t in tags_lower)

    @staticmethod
    def _field_contains(item: dict, field: str, value: str) -> bool:
        """检查字段是否包含指定值（支持 list 和 str 两种类型）"""
        fv = item.get(field)
        if isinstance(fv, list):
            return any(value in str(x).lower() for x in fv)
        if isinstance(fv, str):
            return value in fv.lower()
        return False

    # ── 聚合 ─────────────────────────────────────────────────────

    def get_all_tags(self) -> list[tuple[str, int]]:
        """获取所有标签及出现次数"""
        if not self._loaded:
            self.load()
        counter: dict[str, int] = {}
        for item in self._data:
            for tag in item.get("tags", []):
                counter[tag] = counter.get(tag, 0) + 1
        return sorted(counter.items(), key=lambda x: -x[1])

    def get_all_studios(self) -> list[tuple[str, int]]:
        """获取所有制作公司及作品数"""
        if not self._loaded:
            self.load()
        counter: dict[str, int] = {}
        for item in self._data:
            s = item.get("studio", "")
            if s:
                counter[s] = counter.get(s, 0) + 1
        return sorted(counter.items(), key=lambda x: -x[1])

    # ── 状态 ─────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._data)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# 全局单例
index = MetadataIndex()
