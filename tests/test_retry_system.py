"""失败重测系统 — 全量单元测试 & 集成测试

覆盖场景:
  - FailureStore CRUD 操作
  - FailureRecord 生命周期
  - 多维度查询筛选
  - 重测全部通过
  - 重测仍存在失败
  - 交互式选择（模拟）
  - CLI 参数解析
  - 兼容性: 不影响原有全量测试
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

# 确保项目根目录和 tests 目录在 path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tests_dir = os.path.dirname(os.path.abspath(__file__))
for d in [_project_root, _tests_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from failure_store import (
    FailureStore, FailureRecord, STORE_PATH,
)
from retry_manager import (
    RetryRunner, retry_by_ids, retry_all_failed,
    show_failure_list, _now, _fmt_time, _render_list,
)


# ═══════════════════════════════════════════════════════════
# 单元测试: FailureRecord
# ═══════════════════════════════════════════════════════════

class TestFailureRecord(unittest.TestCase):
    """FailureRecord 数据模型测试"""

    def test_create_record(self):
        r = FailureRecord(
            id="abc123",
            scenario_name="simple_fact - 角色+追问",
            turn_label="角色身份查询(夏亚)",
            query="谁是夏亚",
            errors=["query_type: 期望 simple_fact, 实际 recommendation"],
            plan={"query_type": "recommendation"},
            resolved_query="谁是夏亚",
            elapsed_seconds=5.5,
            timestamp="2026-01-01T00:00:00Z",
        )
        self.assertEqual(r.id, "abc123")
        self.assertEqual(r.scenario_name, "simple_fact - 角色+追问")
        self.assertEqual(len(r.errors), 1)
        self.assertEqual(r.elapsed_seconds, 5.5)
        self.assertEqual(r.retry_count, 0)
        self.assertFalse(r.last_fixed)

    def test_add_retry_result_pass(self):
        r = _make_record("test1")
        self.assertEqual(r.retry_count, 0)
        r.add_retry_result(True, [], 3.0)
        self.assertEqual(r.retry_count, 1)
        self.assertTrue(r.last_fixed)

    def test_add_retry_result_fail(self):
        r = _make_record("test2")
        r.add_retry_result(False, ["still broken"], 4.0)
        self.assertEqual(r.retry_count, 1)
        self.assertFalse(r.last_fixed)

    def test_multiple_retries(self):
        r = _make_record("test3")
        r.add_retry_result(False, ["err1"], 1.0)
        r.add_retry_result(False, ["err2"], 2.0)
        r.add_retry_result(True, [], 3.0)
        self.assertEqual(r.retry_count, 3)
        self.assertTrue(r.last_fixed)

    def test_retry_history_structure(self):
        r = _make_record("test4")
        r.add_retry_result(True, [], 5.5)
        h = r.retry_history[0]
        self.assertIn("timestamp", h)
        self.assertIn("passed", h)
        self.assertIn("errors", h)
        self.assertIn("elapsed_seconds", h)
        self.assertTrue(h["passed"])
        self.assertEqual(h["elapsed_seconds"], 5.5)


def _make_record(id_str: str, scenario: str = "test_scenario") -> FailureRecord:
    return FailureRecord(
        id=id_str,
        scenario_name=scenario,
        turn_label="test_turn",
        query="test query",
        errors=["error 1"],
        plan={},
        resolved_query="test query",
        elapsed_seconds=1.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ═══════════════════════════════════════════════════════════
# 单元测试: FailureStore CRUD
# ═══════════════════════════════════════════════════════════

class TestFailureStore(unittest.TestCase):
    """FailureStore 持久化存储测试"""

    def setUp(self):
        # 使用临时文件
        self.tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.tmpfile.close()
        self.store = FailureStore(path=self.tmpfile.name)

    def tearDown(self):
        os.unlink(self.tmpfile.name)

    def test_empty_store(self):
        self.assertEqual(len(self.store), 0)
        self.assertEqual(self.store.get_unfixed(), [])

    def test_add_record(self):
        r = self.store.add(
            scenario_name="test",
            turn_label="t1",
            query="q",
            errors=["e1"],
            plan={},
            resolved_query="q",
            elapsed_seconds=2.0,
        )
        self.assertEqual(len(self.store), 1)
        self.assertEqual(r.scenario_name, "test")
        self.assertEqual(len(r.errors), 1)

    def test_add_error_record(self):
        r = self.store.add_error("test_scenario", "timeout error")
        self.assertEqual(len(self.store), 1)
        self.assertEqual(r.turn_label, "(异常退出)")
        self.assertEqual(r.errors[0], "timeout error")

    def test_get_by_id(self):
        r = self.store.add("s1", "t1", "q", ["e"], {}, "q", 1.0)
        found = self.store.get_by_id(r.id)
        self.assertIsNotNone(found)
        self.assertEqual(found.scenario_name, "s1")
        not_found = self.store.get_by_id("nonexistent")
        self.assertIsNone(not_found)

    def test_persistence(self):
        """验证 JSON 持久化：新建一个 store 实例应能加载之前的数据"""
        r = self.store.add("s1", "t1", "q", ["e"], {}, "q", 1.0)
        rid = r.id

        # 新建 store 实例加载同一文件
        store2 = FailureStore(path=self.tmpfile.name)
        self.assertEqual(len(store2), 1)
        loaded = store2.get_by_id(rid)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.scenario_name, "s1")

    def test_update_retry(self):
        r = self.store.add("s1", "t1", "q", ["e"], {}, "q", 1.0)
        self.store.update_retry(r.id, True, [], 3.0)
        # 重新加载验证持久化
        store2 = FailureStore(path=self.tmpfile.name)
        loaded = store2.get_by_id(r.id)
        self.assertEqual(loaded.retry_count, 1)
        self.assertTrue(loaded.last_fixed)

    def test_get_unfixed(self):
        self.store.add("s1", "t1", "q", ["e"], {}, "q", 1.0)
        r2 = self.store.add("s2", "t2", "q2", ["e2"], {}, "q2", 2.0)
        self.store.update_retry(r2.id, True, [], 3.0)
        # r1 未修复，r2 已修复
        unfixed = self.store.get_unfixed()
        self.assertEqual(len(unfixed), 1)  # only r1
        self.assertEqual(unfixed[0].scenario_name, "s1")

    def test_get_all(self):
        self.store.add("s1", "t1", "q", ["e"], {}, "q", 1.0)
        self.store.add("s2", "t2", "q2", ["e2"], {}, "q2", 2.0)
        self.assertEqual(len(self.store.get_all()), 2)

    def test_query_scenario_contains(self):
        self.store.add("alpha_test", "t1", "q", ["e"], {}, "q", 1.0)
        self.store.add("beta_test", "t2", "q2", ["e2"], {}, "q2", 2.0)
        results = self.store.query(scenario_contains="alpha")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].scenario_name, "alpha_test")

    def test_query_error_contains(self):
        self.store.add("s1", "t1", "q", ["timeout error"], {}, "q", 1.0)
        self.store.add("s2", "t2", "q2", ["assertion failed"], {}, "q2", 2.0)
        results = self.store.query(error_contains="timeout")
        self.assertEqual(len(results), 1)
        results2 = self.store.query(error_contains="failed")
        self.assertEqual(len(results2), 1)

    def test_query_since(self):
        old_time = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        r = self.store.add("s1", "t1", "q", ["e"], {}, "q", 1.0)
        # 用旧的时间戳应返回0条，用新的应返回
        results_old = self.store.query(since=old_time)
        self.assertEqual(len(results_old), 1)
        future_time = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        results_future = self.store.query(since=future_time)
        self.assertEqual(len(results_future), 0)

    def test_query_limit(self):
        for i in range(5):
            self.store.add(f"s{i}", f"t{i}", "q", ["e"], {}, "q", 1.0)
        results = self.store.query(limit=3)
        self.assertEqual(len(results), 3)

    def test_clear(self):
        self.store.add("s1", "t1", "q", ["e"], {}, "q", 1.0)
        self.store.clear()
        self.assertEqual(len(self.store), 0)

    def test_clear_fixed(self):
        r = self.store.add("s1", "t1", "q", ["e"], {}, "q", 1.0)
        self.store.update_retry(r.id, True, [], 3.0)
        self.store.add("s2", "t2", "q2", ["e2"], {}, "q2", 2.0)
        self.store.clear_fixed()
        self.assertEqual(len(self.store), 1)  # 只留未修复的
        self.assertEqual(self.store.get_all()[0].scenario_name, "s2")

    def test_iter(self):
        self.store.add("s1", "t1", "q", ["e"], {}, "q", 1.0)
        self.store.add("s2", "t2", "q2", ["e2"], {}, "q2", 2.0)
        names = [r.scenario_name for r in self.store]
        self.assertEqual(names, ["s1", "s2"])


# ═══════════════════════════════════════════════════════════
# 单元测试: 辅助函数
# ═══════════════════════════════════════════════════════════

class TestHelperFunctions(unittest.TestCase):
    """辅助函数测试"""

    def test_now_returns_string(self):
        result = _now()
        self.assertIsInstance(result, str)
        self.assertRegex(result, r"\d{2}:\d{2}:\d{2}")

    def test_fmt_time(self):
        iso = "2026-01-15T08:30:00+00:00"
        result = _fmt_time(iso)
        self.assertIn("01-15", result)

    def test_fmt_time_invalid(self):
        result = _fmt_time("invalid")
        self.assertEqual(result, "invalid")


# ═══════════════════════════════════════════════════════════
# 集成测试: RetryRunner
# ═══════════════════════════════════════════════════════════

class _MockSession:
    """模拟 TestSession"""
    def __init__(self, name: str, turns_pass: list[bool] = None):
        self.name = name
        self._turns_pass = turns_pass or [True]
        self.passed = sum(1 for p in self._turns_pass if p)
        self.failed = sum(1 for p in self._turns_pass if not p)
        self.results = []
        self.turn_times = []
        for i, ok in enumerate(self._turns_pass):
            status = "PASS" if ok else "FAIL"
            label = f"turn_{i}"
            elapsed = i + 1.0
            line = f"  {status} [simple_fact +metadata_reasoner] [-] {label} ({elapsed:.1f}s)"
            if not ok:
                line += " error: mock failure"
            self.results.append(line)
            self.turn_times.append((label, elapsed))

    def summary(self):
        return self.passed, self.failed

    def print_results(self):
        for line in self.results:
            print(line)


class TestRetryRunner(unittest.TestCase):
    """RetryRunner 集成测试"""

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.tmpfile.close()
        self.store = FailureStore(path=self.tmpfile.name)

    def tearDown(self):
        os.unlink(self.tmpfile.name)

    def _add_records(self, count: int) -> list[FailureRecord]:
        records = []
        for i in range(count):
            r = self.store.add(
                scenario_name=f"test_scenario_{i}",
                turn_label=f"turn_{i}",
                query=f"query_{i}",
                errors=[f"error_{i}"],
                plan={},
                resolved_query=f"query_{i}",
                elapsed_seconds=float(i),
            )
            records.append(r)
        return records

    @patch("retry_manager._SCENARIO_MAP", {})
    def test_retry_by_ids_empty(self):
        """重测不存在的 ID — 应返回空"""
        async def _run():
            return await retry_by_ids(self.store, ["nonexistent"])
        report = asyncio.run(_run())
        self.assertEqual(report["total_retried"], 0)

    @patch("retry_manager._SCENARIO_MAP", {})
    def test_retry_all_failed_empty(self):
        """没有失败用例时重测全部"""
        async def _run():
            return await retry_all_failed(self.store)
        report = asyncio.run(_run())
        self.assertEqual(report["total_retried"], 0)

    def test_retry_runner_basic(self):
        """RetryRunner 基本流程 — 模拟场景函数"""
        records = self._add_records(2)

        async def mock_fn():
            return _MockSession("test", [True, True])

        with patch.dict("retry_manager._SCENARIO_MAP",
                        {"test_scenario_0": mock_fn, "test_scenario_1": mock_fn}):
            runner = RetryRunner(self.store)
            async def _run():
                return await runner.run_selected(records)
            report = asyncio.run(_run())

        self.assertEqual(report["total_retried"], 2)
        self.assertEqual(report["total_fixed"], 2)
        self.assertEqual(report["total_still_fail"], 0)

    def test_retry_runner_still_failing(self):
        """重测后仍然失败"""
        r = self.store.add(
            scenario_name="failing_test", turn_label="t1", query="q",
            errors=["e"], plan={}, resolved_query="q", elapsed_seconds=1.0,
        )

        async def mock_fn():
            return _MockSession("failing_test", [False])

        with patch.dict("retry_manager._SCENARIO_MAP", {"failing_test": mock_fn}):
            runner = RetryRunner(self.store)
            async def _run():
                return await runner.run_selected([r])
            report = asyncio.run(_run())

        self.assertEqual(report["total_retried"], 1)
        self.assertEqual(report["total_fixed"], 0)
        self.assertEqual(report["total_still_fail"], 1)

    def test_retry_runner_mixed(self):
        """混合：部分修复、部分仍然失败"""
        records = self._add_records(2)

        call_count = [0]
        async def mock_fn_0():
            call_count[0] += 1
            return _MockSession("test0", [False])  # 始终失败

        async def mock_fn_1():
            return _MockSession("test1", [True])   # 始终通过

        with patch.dict("retry_manager._SCENARIO_MAP",
                        {"test_scenario_0": mock_fn_0, "test_scenario_1": mock_fn_1}):
            runner = RetryRunner(self.store)
            async def _run():
                return await runner.run_selected(records)
            report = asyncio.run(_run())

        self.assertEqual(report["total_retried"], 2)
        self.assertEqual(report["total_fixed"], 1)
        self.assertEqual(report["total_still_fail"], 1)

    def test_report_structure(self):
        """验证对比报告结构"""
        records = self._add_records(1)
        async def mock_fn():
            return _MockSession("test", [True])

        with patch.dict("retry_manager._SCENARIO_MAP", {"test_scenario_0": mock_fn}):
            runner = RetryRunner(self.store)
            async def _run():
                return await runner.run_selected(records)
            report = asyncio.run(_run())

        self.assertIn("total_retried", report)
        self.assertIn("total_fixed", report)
        self.assertIn("total_still_fail", report)
        self.assertIn("elapsed_seconds", report)
        self.assertIn("details", report)
        self.assertIsInstance(report["details"], list)

    def test_update_retry_persistence(self):
        """重测后更新记录并持久化"""
        r = self.store.add(
            scenario_name="persist_test", turn_label="t1", query="q",
            errors=["e"], plan={}, resolved_query="q", elapsed_seconds=1.0,
        )
        self.store.update_retry(r.id, True, [], 3.0)

        store2 = FailureStore(path=self.tmpfile.name)
        loaded = store2.get_by_id(r.id)
        self.assertEqual(loaded.retry_count, 1)
        self.assertTrue(loaded.last_fixed)


# ═══════════════════════════════════════════════════════════
# 集成测试: CLI 参数解析 + 兼容性
# ═══════════════════════════════════════════════════════════

class TestCLICompatibility(unittest.TestCase):
    """CLI 参数解析和兼容性测试"""

    def test_parser_default_is_full_test(self):
        """无参数时默认走全量测试"""
        from test_agent import _build_parser
        parser = _build_parser()
        args = parser.parse_args([])
        self.assertFalse(args.retry)
        self.assertFalse(args.retry_all_failed)
        self.assertFalse(args.show_failures)
        self.assertEqual(args.retry_id, [])

    def test_parser_retry_flags(self):
        from test_agent import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--retry"])
        self.assertTrue(args.retry)

    def test_parser_retry_id_single(self):
        from test_agent import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--retry-id", "abc123"])
        self.assertEqual(args.retry_id, ["abc123"])

    def test_parser_retry_id_multiple(self):
        from test_agent import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--retry-id", "abc", "def", "ghi"])
        self.assertEqual(args.retry_id, ["abc", "def", "ghi"])

    def test_parser_retry_all_failed(self):
        from test_agent import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--retry-all-failed"])
        self.assertTrue(args.retry_all_failed)

    def test_parser_retry_last(self):
        from test_agent import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--retry-last"])
        self.assertTrue(args.retry_last)

    def test_parser_show_failures(self):
        from test_agent import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--show-failures"])
        self.assertTrue(args.show_failures)

    def test_parser_clear_failures(self):
        from test_agent import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--clear-failures"])
        self.assertTrue(args.clear_failures)

    def test_is_retry_cmd_detection(self):
        """验证重测命令检测逻辑"""
        from test_agent import _build_parser
        parser = _build_parser()

        # 全量测试不应是重测命令
        args = parser.parse_args([])
        self.assertFalse(any([args.retry, args.retry_id, args.retry_all_failed,
                              args.retry_last, args.show_failures, args.clear_failures]))

        # 重测命令
        for flag in ["--retry", "--retry-all-failed", "--retry-last",
                     "--show-failures", "--clear-failures"]:
            args = parser.parse_args([flag])
            is_retry = any([args.retry, bool(args.retry_id), args.retry_all_failed,
                           args.retry_last, args.show_failures, args.clear_failures])
            self.assertTrue(is_retry, f"{flag} 应被识别为重测命令")


# ═══════════════════════════════════════════════════════════
# 集成测试: 全量测试兼容性
# ═══════════════════════════════════════════════════════════

class TestFullTestCompatibility(unittest.TestCase):
    """确保新增功能不影响原有全量测试"""

    def test_ALL_SCENARIOS_structure(self):
        """ALL_SCENARIOS 结构不变"""
        from test_agent import ALL_SCENARIOS
        self.assertIsInstance(ALL_SCENARIOS, list)
        self.assertGreater(len(ALL_SCENARIOS), 0)
        for item in ALL_SCENARIOS:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertTrue(callable(item[1]))

    def test_TestSession_structure(self):
        """TestSession 接口不变"""
        from test_agent import TestSession
        s = TestSession("test_name", "test_id")
        self.assertTrue(hasattr(s, "turn"))
        self.assertTrue(hasattr(s, "summary"))
        self.assertTrue(hasattr(s, "print_results"))
        self.assertTrue(hasattr(s, "results"))
        self.assertTrue(hasattr(s, "passed"))
        self.assertTrue(hasattr(s, "failed"))
        # 新增字段
        self.assertTrue(hasattr(s, "failures"))
        self.assertTrue(hasattr(s, "turn_times"))

    def test_helper_functions_intact(self):
        """工具函数不变"""
        from test_agent import _now, _plan, _ctx, _get_entity_names
        self.assertTrue(callable(_now))
        self.assertTrue(callable(_plan))
        self.assertTrue(callable(_ctx))
        self.assertTrue(callable(_get_entity_names))

    def test_main_function_exists(self):
        """main() 函数存在且可调用"""
        from test_agent import main
        self.assertTrue(callable(main))
        self.assertTrue(asyncio.iscoroutinefunction(main))

    def test_store_path_in_tests_dir(self):
        """验证存储路径在 tests/ 目录下"""
        self.assertIn("tests", STORE_PATH.replace("\\", "/"))
        self.assertTrue(STORE_PATH.endswith(".json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
