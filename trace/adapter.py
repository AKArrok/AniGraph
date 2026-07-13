"""EventAdapter — 将 LangGraph astream_events 事件转为前端 TraceEvent。

职责：
    - 接收 astream_events(version="v2") 的 on_chain_start / on_chain_end 事件
    - 提取节点名、时间戳、状态变化
    - 生成统一的 TraceEvent dict

解耦 LangGraph 版本：以后 LangGraph API 变化只需改本文件。
"""

import time
from trace.models import NODE_DISPLAY, NodeInfo, NodeRuntime, LLMTrace, StateDiff, TraceEvent
from trace.pricing import TokenProvider


class EventAdapter:
    """LangGraph astream_events → TraceEvent 适配器。"""

    def __init__(self):
        self._nodes: dict[str, dict] = {}       # node_name -> {start, state_before}
        self._llm_events: dict[str, list] = {}  # node_name -> [LLMTrace]
        self._global_start: float = time.time()
        self._graph_path: list[str] = []        # 按执行顺序记录节点名

    # ── 节点事件（从 astream_events 的 on_chain_start/end）──

    def adapt_task_start_from_event(self, event: dict) -> TraceEvent:
        """on_chain_start 事件 → node_start TraceEvent。"""
        name = event.get("name", "")
        display = NODE_DISPLAY.get(name, name)
        task_input = event.get("data", {}).get("input", {})

        self._nodes[name] = {
            "start": time.time(),
            "state_before": _shallow_state(task_input),
        }
        self._graph_path.append(name)

        return TraceEvent(
            type="node_start",
            node=NodeInfo(name=name, display=display),
            runtime=NodeRuntime(
                start=self._nodes[name]["start"],
                end=0,
                state_diff=None,
                llm_calls=[],
                error="",
            ),
            answer_text="",
            graph_path=[],
            summary={},
        )

    def adapt_task_end_from_event(self, event: dict) -> TraceEvent:
        """on_chain_end 事件 → node_end TraceEvent。"""
        name = event.get("name", "")
        display = NODE_DISPLAY.get(name, name)
        output_data = event.get("data", {}).get("output", {})
        error = event.get("data", {}).get("error") or ""

        node_data = self._nodes.get(name, {})
        start_ts = node_data.get("start", time.time())
        state_before = node_data.get("state_before", {})
        state_after = _shallow_state(output_data)
        llm_calls = self._llm_events.pop(name, [])

        state_diff = _compute_state_diff(state_before, state_after)
        end_ts = time.time()

        return TraceEvent(
            type="node_end",
            node=NodeInfo(name=name, display=display),
            runtime=NodeRuntime(
                start=start_ts,
                end=end_ts,
                state_diff=state_diff,
                llm_calls=llm_calls,
                error=error,
            ),
            answer_text="",
            graph_path=[],
            summary={},
        )

    # ── LLM 事件 ───────────────────────────────────────

    def record_llm_call(self, node_name: str, model: str, input_tokens: int,
                         output_tokens: int, elapsed: float,
                         system_prompt: str = "", user_prompt: str = ""):
        """记录一次 LLM 调用（从 astream_events 的 on_chat_model_end 收集）。"""
        cost = TokenProvider.calculate(model, input_tokens, output_tokens)
        self._llm_events.setdefault(node_name, []).append(LLMTrace(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            elapsed=elapsed,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ))

    # ── 回答流 ─────────────────────────────────────────

    def build_answer_chunk(self, text: str) -> TraceEvent:
        """增量文本 → answer_chunk TraceEvent。"""
        return TraceEvent(
            type="answer_chunk",
            node=None,
            runtime=None,
            answer_text=text,
            graph_path=[],
            summary={},
        )

    # ── 汇总 ───────────────────────────────────────────

    def build_summary(self, nodes: list[dict]) -> TraceEvent:
        """所有节点完成后，生成 done TraceEvent。"""
        total_tokens_in = 0
        total_tokens_out = 0

        for n in nodes:
            runtime = n.get("runtime", {})
            for llm in runtime.get("llm_calls", []):
                total_tokens_in += llm.get("input_tokens", 0)
                total_tokens_out += llm.get("output_tokens", 0)

        total_elapsed = time.time() - self._global_start

        return TraceEvent(
            type="done",
            node=None,
            runtime=None,
            answer_text="",
            graph_path=self._graph_path.copy(),
            summary={
                "total_elapsed": round(total_elapsed, 2),
                "total_tokens_in": total_tokens_in,
                "total_tokens_out": total_tokens_out,
            },
        )


# ============================================================
# 内部工具函数
# ============================================================

def _shallow_state(d: dict) -> dict:
    """浅拷贝 state dict，跳过 messages 等大字段。"""
    SKIP = {"messages"}
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items() if k not in SKIP}


def _compute_state_diff(before: dict, after: dict) -> StateDiff:
    """计算两个浅层 state 之间的差异。"""
    added = {}
    changed: list[str] = []
    all_keys = set(before.keys()) | set(after.keys())
    for key in all_keys:
        bv = before.get(key)
        av = after.get(key)
        if bv != av:
            changed.append(key)
            added[key] = _safe_value(av)
    return StateDiff(added=added, changed=changed)


def _safe_value(v) -> str:
    """将复杂值转为可 JSON 序列化的字符串摘要。"""
    if v is None:
        return "[None]"
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        return f"[list, {len(v)} items]"
    if isinstance(v, dict):
        return f"[dict, {len(v)} keys: {', '.join(list(v.keys())[:5])}]"
    return f"[{type(v).__name__}]"
