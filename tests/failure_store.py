"""失败用例持久化存储

每个 FailureRecord 记录一次测试失败的完整信息，支持 JSON 持久化。
支持按场景名、错误类型、时间范围等多维度查询。
"""

import json
import os
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".failure_history.json")


@dataclass
class FailureRecord:
    """单条失败记录"""
    id: str                                    # 唯一标识
    scenario_name: str                         # 场景名，如 "simple_fact - 角色+追问"
    turn_label: str                            # 轮次标签，如 "角色身份查询(夏亚)"
    query: str                                 # 原始查询文本
    errors: list[str]                          # 错误信息列表
    plan: dict                                 # 失败时的执行计划
    resolved_query: str                        # 解析后的查询
    elapsed_seconds: float                     # 本轮耗时
    timestamp: str                             # ISO 格式失败时间
    retry_history: list[dict] = field(default_factory=list)  # 重测历史

    def add_retry_result(self, passed: bool, errors: list[str], elapsed: float):
        """追加一次重测结果"""
        self.retry_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "errors": errors,
            "elapsed_seconds": elapsed,
        })

    @property
    def retry_count(self) -> int:
        return len(self.retry_history)

    @property
    def last_fixed(self) -> bool:
        """最近一次重测是否通过"""
        if not self.retry_history:
            return False
        return self.retry_history[-1]["passed"]


class FailureStore:
    """失败用例持久化管理器"""

    def __init__(self, path: str = STORE_PATH):
        self._path = path
        self._records: list[FailureRecord] = []
        self._load()

    def _load(self):
        """从 JSON 文件加载记录"""
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._records = [FailureRecord(**r) for r in data]
            except Exception:
                self._records = []

    def _save(self):
        """保存到 JSON 文件"""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self._records], f, ensure_ascii=False, indent=2)

    def add(self, scenario_name: str, turn_label: str, query: str,
            errors: list[str], plan: dict, resolved_query: str,
            elapsed_seconds: float) -> FailureRecord:
        """添加一条失败记录"""
        record = FailureRecord(
            id=str(uuid.uuid4())[:8],
            scenario_name=scenario_name,
            turn_label=turn_label,
            query=query,
            errors=errors,
            plan=plan,
            resolved_query=resolved_query,
            elapsed_seconds=elapsed_seconds,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._records.append(record)
        self._save()
        return record

    def add_error(self, scenario_name: str, error_msg: str) -> FailureRecord:
        """添加一条异常类失败记录（非断言失败，是异常退出）"""
        record = FailureRecord(
            id=str(uuid.uuid4())[:8],
            scenario_name=scenario_name,
            turn_label="(异常退出)",
            query="",
            errors=[error_msg],
            plan={},
            resolved_query="",
            elapsed_seconds=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._records.append(record)
        self._save()
        return record

    def update_retry(self, record_id: str, passed: bool, errors: list[str], elapsed: float):
        """更新记录的重测历史"""
        for r in self._records:
            if r.id == record_id:
                r.add_retry_result(passed, errors, elapsed)
                break
        self._save()

    def get_by_id(self, record_id: str) -> Optional[FailureRecord]:
        for r in self._records:
            if r.id == record_id:
                return r
        return None

    def get_unfixed(self) -> list[FailureRecord]:
        """获取所有未修复的失败记录（最近一次重测仍失败或从未重测）"""
        return [r for r in self._records if not r.last_fixed]

    def get_all(self) -> list[FailureRecord]:
        return list(self._records)

    def query(self, *, scenario_contains: str = "", error_contains: str = "",
              since: str = "", limit: int = 0) -> list[FailureRecord]:
        """多维度查询
        
        Args:
            scenario_contains: 场景名包含此字符串
            error_contains: 错误信息包含此字符串
            since: 仅返回此时间之后的记录 (ISO格式)
            limit: 限制返回数量，0 表示不限制
        """
        results = self._records
        if scenario_contains:
            results = [r for r in results if scenario_contains.lower() in r.scenario_name.lower()]
        if error_contains:
            results = [r for r in results if any(error_contains.lower() in e.lower() for e in r.errors)]
        if since:
            results = [r for r in results if r.timestamp >= since]
        if limit > 0:
            results = results[:limit]
        return results

    def clear(self):
        """清空所有记录"""
        self._records = []
        self._save()

    def clear_fixed(self):
        """清除所有已修复的记录"""
        self._records = [r for r in self._records if not r.last_fixed]
        self._save()

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        return iter(self._records)
