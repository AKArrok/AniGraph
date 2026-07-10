"""失败用例重测管理器

功能:
  1. 交互式筛选界面 — 按名称/错误类型/时间筛选，单选/批量选中
  2. 重测执行 — 实时进度、当前状态
  3. 对比报告 — 重测结果 vs 历史失败
  4. 命令行调用 — 支持 --retry-id / --retry-all-failed / --retry-last
"""

import asyncio
import sys
import os
import time
from datetime import datetime, timezone
from typing import Callable

# 确保 tests 目录在 path 中
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from failure_store import FailureStore, FailureRecord

# 场景映射: 场景名 → 测试函数
# 延迟导入，避免循环依赖
_SCENARIO_MAP: dict[str, Callable] = {}

def _init_scenario_map():
    """延迟初始化场景映射"""
    if _SCENARIO_MAP:
        return
    from test_agent import ALL_SCENARIOS
    for name, fn in ALL_SCENARIOS:
        _SCENARIO_MAP[name] = fn


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_time(iso: str) -> str:
    """ISO 时间转本地可读格式"""
    try:
        dt = datetime.fromisoformat(iso)
        local = dt.astimezone()
        return local.strftime("%m-%d %H:%M:%S")
    except Exception:
        return iso[:19]


# ═══════════════════════════════════════════════════════════
# 交互式筛选界面
# ═══════════════════════════════════════════════════════════

def _render_list(records: list[FailureRecord], selected: set[int]):
    """渲染失败用例列表"""
    print(f"\n{'='*70}")
    print(f"  失败用例列表 ({len(records)} 条)")
    print(f"{'='*70}")
    print(f"  {'#':<4} {'Sel':<5} {'场景':<35} {'标签':<30} {'时间':<16}")
    print(f"  {'-'*4} {'-'*5} {'-'*35} {'-'*30} {'-'*16}")

    for i, r in enumerate(records):
        sel_mark = "[x]" if i in selected else "[ ]"
        scenario = r.scenario_name[:33] + ".." if len(r.scenario_name) > 35 else r.scenario_name
        label = r.turn_label[:28] + ".." if len(r.turn_label) > 30 else r.turn_label
        ts = _fmt_time(r.timestamp)
        retry_note = f" (重测{r.retry_count}次)" if r.retry_count > 0 else ""
        print(f"  {i:<4} {sel_mark:<5} {scenario:<35} {label:<30} {ts:<16}{retry_note}")

    print(f"  {'-'*70}")


def _render_detail(record: FailureRecord):
    """展示单条记录详情"""
    print(f"\n  ── 详情 ──")
    print(f"  ID:       {record.id}")
    print(f"  场景:     {record.scenario_name}")
    print(f"  标签:     {record.turn_label}")
    print(f"  查询:     {record.query}")
    print(f"  解析后:   {record.resolved_query}")
    print(f"  耗时:     {record.elapsed_seconds:.1f}s")
    print(f"  错误:")
    for e in record.errors:
        print(f"    - {e}")
    if record.retry_history:
        print(f"  重测历史:")
        for h in record.retry_history:
            status = "PASS" if h["passed"] else "FAIL"
            ts = _fmt_time(h["timestamp"])
            print(f"    [{ts}] {status} ({h['elapsed_seconds']:.1f}s)")


def _interactive_select(store: FailureStore) -> list[FailureRecord]:
    """交互式筛选并选择待重测的失败用例"""
    records = store.get_unfixed()
    if not records:
        print("\n  没有未修复的失败用例。")
        return []

    selected: set[int] = set()
    current_filter = ""
    view_mode = "list"  # list | detail

    while True:
        # 应用筛选
        if current_filter:
            filtered = [r for i, r in enumerate(records)
                       if current_filter.lower() in r.scenario_name.lower()
                       or current_filter.lower() in r.turn_label.lower()
                       or any(current_filter.lower() in e.lower() for e in r.errors)]
        else:
            filtered = records

        if view_mode == "list":
            _render_list(filtered, selected)
        else:
            # detail 模式只展示最后选中的
            pass

        print(f"\n  命令: [a]全选  [c]清除  [f]筛选  [d]详情  [0-9]选中/取消")
        print(f"         [r]开始重测  [q]退出  当前筛选: {current_filter!r}")
        cmd = input("  > ").strip().lower()

        if cmd == "q":
            return []
        elif cmd == "a":
            selected = set(range(len(records)))
        elif cmd == "c":
            selected.clear()
        elif cmd == "f":
            current_filter = input("  筛选关键词 (回车取消): ").strip()
        elif cmd == "d":
            idx_str = input("  输入序号查看详情: ").strip()
            try:
                idx = int(idx_str)
                if 0 <= idx < len(records):
                    _render_detail(records[idx])
            except ValueError:
                pass
        elif cmd == "r":
            if not selected:
                print("  未选中任何用例！")
                continue
            return [records[i] for i in sorted(selected)]
        else:
            # 尝试解析数字（支持逗号分隔和范围）
            try:
                idx = int(cmd)
                if 0 <= idx < len(records):
                    if idx in selected:
                        selected.discard(idx)
                    else:
                        selected.add(idx)
            except ValueError:
                print(f"  未知命令: {cmd}")


# ═══════════════════════════════════════════════════════════
# 重测执行
# ═══════════════════════════════════════════════════════════

class RetryRunner:
    """重测执行器"""

    def __init__(self, store: FailureStore):
        self.store = store
        self.results: list[dict] = []
        _init_scenario_map()

    async def run_selected(self, records: list[FailureRecord]) -> dict:
        """执行选定用例的重测，返回汇总报告"""
        print(f"\n{'='*70}")
        print(f"  开始重测 {len(records)} 个失败用例")
        print(f"{'='*70}\n")

        # 按场景分组，同一场景只跑一次（因为场景函数本身就是多轮的）
        scenario_records: dict[str, list[FailureRecord]] = {}
        for r in records:
            scenario_records.setdefault(r.scenario_name, []).append(r)

        total_retried = 0
        total_fixed = 0
        total_still_fail = 0
        t_start = time.time()
        all_details: list[dict] = []

        for idx, (scenario_name, recs) in enumerate(scenario_records.items()):
            fn = _SCENARIO_MAP.get(scenario_name)
            if fn is None:
                print(f"  [{_now()}] [{idx+1}/{len(scenario_records)}] {scenario_name} — SKIP (未找到函数)")
                continue

            # ── 实时进度 ──
            print(f"  [{_now()}] [{idx+1}/{len(scenario_records)}] {scenario_name} ...",
                  end=" ", flush=True)

            try:
                session = await fn()
                p, f = session.summary()

                # 判断该场景的哪些失败轮次修复了
                for rec in recs:
                    still_failing = False
                    new_errors = []
                    for line in session.results:
                        if line.strip().startswith("FAIL"):
                            still_failing = True
                            # 提取错误信息
                            if ";" in line:
                                new_errors = [e.strip() for e in line.split(";", 1)[1:]]
                            break

                    elapsed = sum(float(r.get("elapsed_seconds", 0)) for r in all_details if r["record_id"] == rec.id)
                    self.store.update_retry(rec.id, not still_failing, new_errors, elapsed)
                    total_retried += 1
                    if still_failing:
                        total_still_fail += 1
                    else:
                        total_fixed += 1

                    all_details.append({
                        "record_id": rec.id,
                        "scenario_name": scenario_name,
                        "turn_label": rec.turn_label,
                        "was_failing": rec.errors,
                        "now_passing": not still_failing,
                        "new_errors": new_errors,
                    })

                if f == 0:
                    print(f"OK (场景 {p} pass, 该场景所有失败已修复)")
                else:
                    print(f"STILL FAIL ({p} pass, {f} fail)")

            except Exception as e:
                print(f"ERROR: {e}")
                for rec in recs:
                    self.store.update_retry(rec.id, False, [str(e)], 0)
                    total_retried += 1
                    total_still_fail += 1
                    all_details.append({
                        "record_id": rec.id,
                        "scenario_name": scenario_name,
                        "turn_label": rec.turn_label,
                        "was_failing": rec.errors,
                        "now_passing": False,
                        "new_errors": [str(e)],
                    })

        elapsed = time.time() - t_start
        report = {
            "total_retried": total_retried,
            "total_fixed": total_fixed,
            "total_still_fail": total_still_fail,
            "elapsed_seconds": elapsed,
            "details": all_details,
        }

        self._print_report(report)
        return report

    def _print_report(self, report: dict):
        """打印对比报告"""
        total = report["total_retried"]
        fixed = report["total_fixed"]
        still = report["total_still_fail"]
        pct = (fixed / total * 100) if total > 0 else 0

        print(f"\n{'='*70}")
        print(f"  重测对比报告")
        print(f"{'='*70}")
        print(f"  重测用例数:   {total}")
        print(f"  已修复:       {fixed} ({pct:.0f}%)")
        print(f"  仍然失败:     {still}")
        print(f"  总耗时:       {report['elapsed_seconds']:.0f}s")
        print(f"{'='*70}")

        # 逐条对比
        if report["details"]:
            print(f"\n  {'状态':<6} {'场景':<35} {'标签':<25} {'历史错误':<40}")
            print(f"  {'-'*6} {'-'*35} {'-'*25} {'-'*40}")
            for d in report["details"]:
                status = "FIXED" if d["now_passing"] else "FAIL"
                status_icon = "✓" if d["now_passing"] else "✗"
                scenario = d["scenario_name"][:33] + ".." if len(d["scenario_name"]) > 35 else d["scenario_name"]
                label = d["turn_label"][:23] + ".." if len(d["turn_label"]) > 25 else d["turn_label"]
                hist_err = "; ".join(d["was_failing"])[:38] + ".." if len("; ".join(d["was_failing"])) > 40 else "; ".join(d["was_failing"])
                print(f"  {status_icon} {status:<4} {scenario:<35} {label:<25} {hist_err:<40}")
                if not d["now_passing"] and d["new_errors"]:
                    for ne in d["new_errors"]:
                        print(f"         └─ 当前错误: {ne}")

        # 建议
        if still > 0:
            print(f"\n  💡 仍有 {still} 个用例失败，可再次运行: python tests/test_agent.py --retry-last")
        else:
            print(f"\n  ✓ 所有重测用例均已通过！")
            # 自动清除已修复的记录
            self.store.clear_fixed()
            print(f"  已自动清除已修复记录。")


# ═══════════════════════════════════════════════════════════
# 命令行模式入口
# ═══════════════════════════════════════════════════════════

async def retry_by_ids(store: FailureStore, ids: list[str]) -> dict:
    """通过 ID 列表指定重测"""
    records = [store.get_by_id(rid) for rid in ids]
    records = [r for r in records if r is not None]
    if not records:
        print("  未找到匹配的记录。")
        return {"total_retried": 0, "total_fixed": 0, "total_still_fail": 0, "elapsed_seconds": 0, "details": []}
    runner = RetryRunner(store)
    return await runner.run_selected(records)


async def retry_all_failed(store: FailureStore) -> dict:
    """重测所有未修复的失败用例"""
    records = store.get_unfixed()
    if not records:
        print("  没有未修复的失败用例。")
        return {"total_retried": 0, "total_fixed": 0, "total_still_fail": 0, "elapsed_seconds": 0, "details": []}
    runner = RetryRunner(store)
    return await runner.run_selected(records)


async def retry_interactive(store: FailureStore) -> dict:
    """交互式选择并重测"""
    if not store.get_unfixed():
        print("  没有未修复的失败用例。")
        return {"total_retried": 0, "total_fixed": 0, "total_still_fail": 0, "elapsed_seconds": 0, "details": []}
    records = _interactive_select(store)
    if not records:
        return {"total_retried": 0, "total_fixed": 0, "total_still_fail": 0, "elapsed_seconds": 0, "details": []}
    runner = RetryRunner(store)
    return await runner.run_selected(records)


def show_failure_list(store: FailureStore):
    """展示失败列表（非交互）"""
    records = store.get_unfixed()
    if not records:
        print("  没有未修复的失败用例。")
        return
    _render_list(records, set())
    for r in records:
        _render_detail(r)
