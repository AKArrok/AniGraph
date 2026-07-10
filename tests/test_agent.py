"""AniGraph 多 Agent 协作 — 全功能多轮问答回归测试

覆盖功能：
  query_type:   simple_fact / recommendation / comparison / chat
  检索路径:     metadata / semantic / mixed
  Expert:       metadata_reasoner / similar_expert
  Entity:       character / alias / meme
  多轮上下文:    追问检测 / 指代解析 / 话题连续 / recent_entities 传递
  别名解析:     字典匹配 / 包含匹配
  快速通道:     simple_fact_answer / chat 直达

结构: 每个测试会话 (thread) 包含 1-3 轮对话，按功能场景分组。

命令行:
  python tests/test_agent.py                    完整回归测试
  python tests/test_agent.py --retry            交互式选择失败用例重测
  python tests/test_agent.py --retry-id <id>    重测指定 ID 的失败用例
  python tests/test_agent.py --retry-all-failed 重测所有未修复失败用例
  python tests/test_agent.py --retry-last       重测最近失败的用例
  python tests/test_agent.py --show-failures    查看失败历史
"""

import argparse
import asyncio
import sys
import os
import logging
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)
logging.getLogger("agents").setLevel(logging.WARNING)

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from graph import build_graph

# ═══════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════


def _final_answer(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content.strip():
            return m.content.strip()
    return ""


def _get_entity_names(state: dict) -> list[str]:
    """从 state 提取当前讨论的实体名"""
    names = []
    if state.get("entity_name"):
        names.append(state["entity_name"])
    for e in state.get("recent_entities", []):
        if e.get("name") and e.get("name") != state.get("entity_name", ""):
            names.append(e["name"])
    return names


async def _query(app, query: str, thread_id: str) -> dict:
    """发送一条消息，返回最终 state"""
    resp = await app.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        {"configurable": {"thread_id": thread_id}},
    )
    return resp


def _plan(state: dict) -> dict:
    return state.get("plan", {})


def _ctx(state: dict) -> dict:
    c = state.get("context", {})
    return c if isinstance(c, dict) else {}


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


# ═══════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════

class TestSession:
    """一个测试会话，封装 thread_id + graph + 断言方法"""

    def __init__(self, name: str, thread_id: str):
        self.name = name
        self.thread_id = thread_id
        self._memory = MemorySaver()
        self.passed = 0
        self.failed = 0
        self.results: list[str] = []
        self.failures: list[dict] = []        # 结构化失败数据
        self.turn_times: list[tuple[str, float]] = []  # (label, elapsed)

    def _build(self):
        g = build_graph()
        return g.compile(checkpointer=self._memory)

    async def turn(self, query: str, **expects):
        """发送一条消息并断言。expects 可包含:
          query_type: 期望的查询分类
          is_followup: 是否为追问
          resolved_query_contains: resolved_query 应包含此子串
          entity_type: 实体类型
          entity_name: 实体名 (None 表示期望为空)
          entity_in_recent: 实体名应在 recent_entities 中
          experts: 期望的 Expert 列表
          answer_not_empty: 应答非空 (默认 True)
          label: 本轮描述
        """
        label = expects.pop("label", query)
        t0 = time.time()
        app = self._build()
        state = await _query(app, query, self.thread_id)
        elapsed = time.time() - t0
        answer = _final_answer(state.get("messages", []))
        plan = _plan(state)
        ctx = _ctx(state)

        errors: list[str] = []
        qt = plan.get("query_type", "?")

        # ── 必检项 ──
        if "query_type" in expects and qt != expects["query_type"]:
            errors.append(f"query_type: 期望 {expects['query_type']}, 实际 {qt}")
        if "is_followup" in expects and ctx.get("is_followup") != expects["is_followup"]:
            errors.append(f"is_followup: 期望 {expects['is_followup']}, 实际 {ctx.get('is_followup')}")
        if "resolved_query_contains" in expects and expects["resolved_query_contains"] not in ctx.get("resolved_query", ""):
            errors.append(f"resolved_query 不含 '{expects['resolved_query_contains']}', 实际: {ctx.get('resolved_query', '')}")
        if "entity_type" in expects and state.get("entity_type") != expects["entity_type"]:
            errors.append(f"entity_type: 期望 {expects['entity_type']!r}, 实际 {state.get('entity_type')!r}")
        if "entity_name" in expects:
            en = expects["entity_name"]
            actual = state.get("entity_name", "")
            if en is None and actual:
                errors.append(f"entity_name: 期望为空, 实际 {actual!r}")
            elif en is not None and en not in actual:
                errors.append(f"entity_name: 期望含 {en!r}, 实际 {actual!r}")
        if "entity_in_recent" in expects:
            recent_names = [e.get("name") for e in state.get("recent_entities", [])]
            if expects["entity_in_recent"] not in recent_names:
                errors.append(f"recent_entities 不含 '{expects['entity_in_recent']}', 实际: {recent_names}")
        if expects.get("answer_not_empty", True) and not answer:
            errors.append("answer 为空")
        if "experts" in expects:
            actual_experts = plan.get("experts", [])
            if sorted(actual_experts) != sorted(expects["experts"]):
                errors.append(f"experts: 期望 {expects['experts']}, 实际 {actual_experts}")

        # ── 输出 ──
        ent_str = ",".join(_get_entity_names(state)) or "-"
        plan_str = f"{qt}"
        if plan.get("experts"):
            plan_str += f" +{','.join(plan['experts'])}"
        fup = "F" if ctx.get("is_followup") else "-"
        rq = ctx.get("resolved_query", query)
        status = "PASS" if not errors else "FAIL"
        timing_str = f"{elapsed:.1f}s".rjust(7)
        self.results.append(
            f"  {status} [{plan_str}] [{fup}] {label} ({elapsed:.1f}s) {'; '.join(errors)}"
        )
        if status == "PASS":
            self.passed += 1
        else:
            self.failed += 1
            # 失败时打印更多上下文
            self.results.append(f"       query={query!r} resolved={rq!r} entity={ent_str}")
            self.results.append(f"       plan={plan}")
            # 结构化失败数据（供持久化存储）
            self.failures.append({
                "scenario_name": self.name,
                "turn_label": label,
                "query": query,
                "errors": list(errors),
                "plan": dict(plan),
                "resolved_query": rq,
                "elapsed_seconds": elapsed,
            })

        # 记录每轮耗时
        self.turn_times.append((label, elapsed))

        return state

    def summary(self) -> tuple[int, int]:
        return self.passed, self.failed

    def print_results(self):
        """打印本轮所有结果"""
        for line in self.results:
            print(line)


# ═══════════════════════════════════════════════════════════
# 测试场景
# ═══════════════════════════════════════════════════════════


async def test_simple_fact_character():
    """simple_fact: 角色身份查询 + 追问声优"""
    s = TestSession("角色事实查询", "test_char_fact")

    # 轮1: 谁是夏亚 → simple_fact, 实体解析
    await s.turn("谁是夏亚",
        query_type="simple_fact",
        entity_type="character", entity_name="夏亚",
        is_followup=False,
        label="角色身份查询(夏亚)")

    # 轮2: 追问声优 → simple_fact, 指代解析 "他→夏亚"
    await s.turn("他的声优是谁",
        query_type="simple_fact",
        is_followup=True,
        resolved_query_contains="夏亚",
        entity_in_recent="夏亚",
        label="追问声优(他→夏亚)")

    return s


async def test_simple_fact_score():
    """simple_fact: 评分查询"""
    s = TestSession("评分查询", "test_score")

    await s.turn("命运石之门的评分是多少",
        query_type="simple_fact",
        label="评分查询(石头门)")

    # 追问: 排名
    await s.turn("它在Bangumi排第几",
        query_type="simple_fact",
        is_followup=True,
        resolved_query_contains="命运石之门",
        label="追问排名(它→石头门)")

    return s


async def test_simple_fact_studio():
    """simple_fact: 制作公司查询（metadata 路径）"""
    s = TestSession("制作查询", "test_studio")

    await s.turn("进击的巨人制作公司是哪家",
        query_type="simple_fact",
        label="制作公司查询")

    return s


async def test_simple_fact_seiyuu():
    """simple_fact: 声优查询 + 角色切换追问"""
    s = TestSession("声优查询", "test_seiyuu")

    await s.turn("雷姆的声优是谁",
        query_type="simple_fact",
        entity_type="character", entity_name="雷姆",
        label="声优查询(雷姆)")

    # 追问同作品内另一个角色
    await s.turn("那拉姆呢",
        query_type="simple_fact",
        is_followup=True,
        entity_in_recent="雷姆",
        label="角色切换追问(拉姆)")

    return s


async def test_simple_fact_director():
    """simple_fact: 导演查询"""
    s = TestSession("导演查询", "test_director")

    await s.turn("EVA的导演是谁",
        query_type="simple_fact",
        label="导演查询(EVA别名)")

    return s


async def test_recommendation_similar():
    """recommendation: 相似推荐 (semantic → similar_expert)"""
    s = TestSession("相似推荐", "test_similar")

    await s.turn("有没有类似进击的巨人的番",
        query_type="recommendation",
        label="相似推荐(巨人)")

    # 追问更多推荐
    await s.turn("还有吗",
        is_followup=True,
        label="追问更多推荐")

    return s


async def test_recommendation_tag():
    """recommendation: 标签推荐 (metadata + semantic → mixed)"""
    s = TestSession("标签推荐", "test_tag")

    await s.turn("推荐几部好看的异世界番",
        query_type="recommendation",
        label="标签推荐(异世界)")

    return s


async def test_recommendation_discovery():
    """recommendation: 发现类查询"""
    s = TestSession("发现推荐", "test_discover")

    await s.turn("有哪些评分高的科幻动漫",
        label="发现查询(高分科幻)")

    return s


async def test_comparison():
    """comparison: 对比查询"""
    s = TestSession("对比查询", "test_compare")

    await s.turn("对比一下鬼灭之刃和咒术回战",
        query_type="comparison",
        label="对比查询")

    return s


async def test_chat():
    """chat: 闲聊直达"""
    s = TestSession("闲聊", "test_chat")

    # 问候 → chat, 直接返回
    await s.turn("你好呀",
        query_type="chat",
        experts=[],  # chat 不调 Expert
        label="问候")

    # 能力询问
    await s.turn("你能做什么",
        query_type="chat",
        label="能力询问")

    return s


async def test_alias_resolution():
    """别名解析: 字典匹配 / 包含匹配"""
    s = TestSession("别名解析", "test_alias")

    # 精确别名 → 可能是 simple_fact 或 recommendation（"好看吗" 带评价语义）
    await s.turn("石头门好看吗",
        entity_type="alias",
        label="别名解析(石头门→命运石之门)")

    return s


async def test_meme_entity():
    """梗实体解析"""
    s = TestSession("梗实体", "test_meme")

    await s.turn("典明粥是什么梗",
        entity_type="meme",
        label="梗实体(典明粥)")

    return s


async def test_multi_entity_conversation():
    """多实体切换 + 指代正确性"""
    s = TestSession("多实体切换", "test_multi_ent")

    # 轮1: 第一个角色
    await s.turn("谁是炭治郎",
        query_type="simple_fact",
        entity_type="character", entity_name="炭治郎",
        label="角色1(炭治郎)")

    # 轮2: 切换到新角色（"那...呢" 是追问模式）
    await s.turn("那祢豆子呢",
        is_followup=True,
        entity_in_recent="炭治郎",
        label="角色2(祢豆子)")

    # 轮3: "他" 应指代最近讨论的祢豆子
    await s.turn("她的能力是什么",
        query_type="simple_fact",
        is_followup=True,
        resolved_query_contains="祢豆子",
        label="指代最近实体(她→祢豆子)")

    return s


async def test_mixed_tag_semantic():
    """mixed 路径: 标签过滤 + 语义推荐"""
    s = TestSession("混合路径", "test_mixed")

    await s.turn("推荐类似命运石之门的科幻悬疑番",
        query_type="recommendation",
        label="混合(语义+标签)")

    return s


async def test_edge_empty_entity():
    """边界: 无实体匹配的短查询"""
    s = TestSession("无实体查询", "test_noent")

    await s.turn("有什么好看的番",
        query_type="recommendation",
        label="无实体发现查询")

    return s


async def test_edge_short_ambiguous():
    """边界: 短查询可能歧义"""
    s = TestSession("短歧义查询", "test_short")

    await s.turn("巨人的评分",
        query_type="simple_fact",
        label="短查询(含别名'巨人')")

    return s


async def test_followup_complex_chain():
    """长链路追问: 事实→推荐→追问→切换话题"""
    s = TestSession("长链路追问", "test_chain")

    # 轮1: 事实
    await s.turn("夏亚是谁",
        query_type="simple_fact", entity_name="夏亚",
        label="轮1-事实")

    # 轮2: 追问对手
    await s.turn("他的对手是谁",
        query_type="simple_fact",
        is_followup=True, resolved_query_contains="夏亚",
        label="轮2-对手")

    # 轮3: 推荐类似作品
    await s.turn("推荐类似高达的机战番",
        query_type="recommendation",
        label="轮3-推荐(话题切换)")

    return s


async def test_topic_persistence():
    """话题持久: recent_entities 跨话题传递"""
    s = TestSession("话题持久", "test_topic")

    await s.turn("谁是艾伦",
        query_type="simple_fact", entity_name="艾伦",
        label="话题A-艾伦")

    # 切换到新话题
    await s.turn("惠惠是谁",
        query_type="simple_fact", entity_name="惠惠",
        label="话题B-惠惠")

    # "她" 应指惠惠（最近实体），不是艾伦
    await s.turn("她的技能是什么",
        is_followup=True,
        resolved_query_contains="惠惠",
        label="指代惠惠(她→惠惠)")

    return s


# ═══════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════

ALL_SCENARIOS = [
    ("simple_fact - 角色+追问", test_simple_fact_character),
    ("simple_fact - 评分+追问", test_simple_fact_score),
    ("simple_fact - 制作公司", test_simple_fact_studio),
    ("simple_fact - 声优+角色切换", test_simple_fact_seiyuu),
    ("simple_fact - 导演(别名)", test_simple_fact_director),
    ("recommendation - 相似推荐+追问", test_recommendation_similar),
    ("recommendation - 标签推荐", test_recommendation_tag),
    ("recommendation - 发现查询", test_recommendation_discovery),
    ("comparison - 对比查询", test_comparison),
    ("chat - 闲聊直达", test_chat),
    ("alias - 别名解析", test_alias_resolution),
    ("meme - 梗实体", test_meme_entity),
    ("multi-entity - 多实体切换+指代", test_multi_entity_conversation),
    ("mixed - 标签+语义融合", test_mixed_tag_semantic),
    ("edge - 无实体查询", test_edge_empty_entity),
    ("edge - 短歧义查询(别名)", test_edge_short_ambiguous),
    ("followup - 长链路追问", test_followup_complex_chain),
    ("topic - 话题持久+指代", test_topic_persistence),
]


async def main():
    """完整回归测试运行入口"""
    print(f"\n{'='*60}")
    print("  AniGraph 全功能多轮回归测试")
    print(f"  {len(ALL_SCENARIOS)} 个场景, 启动中...")
    print(f"{'='*60}\n")

    from failure_store import FailureStore
    store = FailureStore()

    total_pass = 0
    total_fail = 0
    scenario_fail = 0
    scenario_times: list[tuple[str, float, int, int]] = []  # (name, elapsed, pass, fail)
    t_start = time.time()

    for name, fn in ALL_SCENARIOS:
        sc_t0 = time.time()
        print(f"[{_now()}] {name} ...", end=" ", flush=True)
        try:
            s = await fn()
            p, f = s.summary()
            sc_elapsed = time.time() - sc_t0
            scenario_times.append((name, sc_elapsed, p, f))
            total_pass += p
            total_fail += f
            if f > 0:
                scenario_fail += 1
                print(f"FAIL ({p} pass, {f} fail) — {sc_elapsed:.0f}s")
                s.print_results()
                # 持久化失败用例
                for fail_data in s.failures:
                    store.add(
                        scenario_name=fail_data["scenario_name"],
                        turn_label=fail_data["turn_label"],
                        query=fail_data["query"],
                        errors=fail_data["errors"],
                        plan=fail_data["plan"],
                        resolved_query=fail_data["resolved_query"],
                        elapsed_seconds=fail_data["elapsed_seconds"],
                    )
            else:
                print(f"OK ({p} pass) — {sc_elapsed:.0f}s")
        except Exception as e:
            scenario_fail += 1
            sc_elapsed = time.time() - sc_t0
            scenario_times.append((name, sc_elapsed, 0, 1))
            error_msg = str(e)[:200]
            print(f"ERROR: {error_msg}")
            store.add_error(name, error_msg)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  总计: {total_pass} pass, {total_fail} fail")
    print(f"  场景: {len(ALL_SCENARIOS)-scenario_fail}/{len(ALL_SCENARIOS)} 通过")
    print(f"  总耗时: {elapsed:.0f}s")
    print(f"{'='*60}")

    # ── 各场景耗时明细 ──
    if scenario_times:
        print(f"\n{'='*60}")
        print(f"  各场景耗时明细")
        print(f"{'='*60}")
        print(f"  {'场景':<40} {'耗时':>8} {'结果':>10}")
        print(f"  {'-'*40} {'-'*8} {'-'*10}")
        for name, sc_elapsed, p, f in scenario_times:
            result = f"{p}P/{f}F" if f > 0 else "OK"
            # 耗时分级着色（用符号标记）
            if sc_elapsed < 30:
                mark = "  "
            elif sc_elapsed < 90:
                mark = " *"
            else:
                mark = "**"
            bar = "█" * min(int(sc_elapsed / 10), 15)
            print(f"  {name:<40} {sc_elapsed:>5.0f}s{mark} {bar} {result:>10}")
        print(f"  {'-'*40} {'-'*8} {'-'*10}")
        print(f"  {'总计':<40} {elapsed:>5.0f}s")

    # ── 失败提示 ──
    if total_fail > 0:
        print(f"\n  ⚠ {total_fail} 个断言失败。重测命令:")
        print(f"    python tests/test_agent.py --retry              # 交互式选择")
        print(f"    python tests/test_agent.py --retry-all-failed   # 重测全部失败")
        print(f"    python tests/test_agent.py --show-failures      # 查看失败详情")
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    p = argparse.ArgumentParser(
        description="AniGraph 全功能多轮回归测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tests/test_agent.py                       完整回归测试
  python tests/test_agent.py --retry               交互式选择失败用例重测
  python tests/test_agent.py --retry-id abc123     重测指定 ID
  python tests/test_agent.py --retry-all-failed    重测所有未修复失败用例
  python tests/test_agent.py --retry-last          重测最近失败的用例
  python tests/test_agent.py --show-failures       查看失败历史
  python tests/test_agent.py --clear-failures      清空失败历史记录
        """,
    )
    p.add_argument("--retry", action="store_true",
                   help="交互式选择失败用例重测")
    p.add_argument("--retry-id", type=str, nargs="+", default=[],
                   help="重测指定 ID 的失败用例（可多个）")
    p.add_argument("--retry-all-failed", action="store_true",
                   help="重测所有未修复的失败用例")
    p.add_argument("--retry-last", action="store_true",
                   help="重测最近失败的用例（按时间取最近 10 个）")
    p.add_argument("--show-failures", action="store_true",
                   help="查看失败历史记录")
    p.add_argument("--clear-failures", action="store_true",
                   help="清空所有失败历史记录")
    return p


async def _handle_retry_cmds(args):
    """处理重测相关命令"""
    from failure_store import FailureStore
    from retry_manager import (
        retry_interactive, retry_by_ids, retry_all_failed, show_failure_list
    )

    store = FailureStore()

    if args.clear_failures:
        store.clear()
        print("  已清空所有失败历史记录。")
        return

    if args.show_failures:
        show_failure_list(store)
        return

    if args.retry_id:
        report = await retry_by_ids(store, args.retry_id)
        if report["total_still_fail"] > 0:
            sys.exit(1)
        return

    if args.retry_all_failed:
        report = await retry_all_failed(store)
        if report["total_still_fail"] > 0:
            sys.exit(1)
        return

    if args.retry_last:
        records = store.get_unfixed()
        if not records:
            print("  没有未修复的失败用例。")
            return
        # 按时间取最近 10 个
        records.sort(key=lambda r: r.timestamp, reverse=True)
        records = records[:10]
        from retry_manager import RetryRunner
        runner = RetryRunner(store)
        report = await runner.run_selected(records)
        if report["total_still_fail"] > 0:
            sys.exit(1)
        return

    if args.retry:
        report = await retry_interactive(store)
        if report["total_still_fail"] > 0:
            sys.exit(1)
        return


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    # 判断是否为重测命令
    is_retry_cmd = any([
        args.retry, args.retry_id, args.retry_all_failed,
        args.retry_last, args.show_failures, args.clear_failures,
    ])

    if is_retry_cmd:
        asyncio.run(_handle_retry_cmds(args))
    else:
        asyncio.run(main())
