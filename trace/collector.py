"""TraceCollector — 用单一 astream_events 流捕获节点事件 + LLM token 用量。

核心流程:
    1. 调用 compiled_graph.astream_events(version="v2")
    2. on_chain_start → adapter.adapt_task_start → yield node_start event
    3. on_chain_end   → adapter.adapt_task_end   → yield node_end event
    4. on_chat_model_end → adapter.record_llm_call（token 用量）
    5. on_chat_model_stream → 累积并发送 answer_chunk（打字机流）
    6. 完成 → adapter.build_summary → yield done event

⚠️ 关键: 只用一个 astream_events 流，避免 astream + astream_events 双流导致图执行两次。
"""
import time
from typing import AsyncIterator, Any
from trace.models import TraceEvent
from trace.adapter import EventAdapter


# ── 需要追踪的图节点（过滤掉 LangGraph 内部链和路由函数）──
_GRAPH_NODES = frozenset({
    "alias_resolve", "history_extractor", "context_builder",
    "planner", "query_processing", "knowledge_retrieval",
    "metadata_reasoner", "similar_expert", "merge",
    "simple_fact_answer", "web_fallback", "answer_planner", "answer",
})

# ── 回答节点（这些节点的 on_chat_model_stream 流式输出即最终回答）──
_ANSWER_NODES = frozenset({"simple_fact_answer", "answer"})


class TraceCollector:
    """收集一次 agent.astream_events 调用中所有节点事件 + LLM token 用量。"""

    def __init__(self):
        self.adapter = EventAdapter()

    async def collect(self, app: Any, input_state: dict, config: dict) -> AsyncIterator[TraceEvent]:
        """主收集循环（单一 astream_events 流）。

        Args:
            app: 编译后的 CompiledStateGraph
            input_state: {"messages": [HumanMessage(...)]}
            config: {"configurable": {"thread_id": "..."}}

        Yields:
            TraceEvent，按时间顺序发送给前端。
        """
        collected_nodes: list[dict] = []
        answer_text = ""           # 累积回答文本
        answer_streaming = False   # 是否正在流式输出回答

        # 单流: astream_events(version="v2") 包含所有事件
        async for event in app.astream_events(input_state, config, version="v2"):
            evt_type = event.get("event", "")
            name = event.get("name", "")
            langgraph_node = event.get("metadata", {}).get("langgraph_node", "")

            # ── 节点开始 ──
            if evt_type == "on_chain_start" and name in _GRAPH_NODES:
                evt = self.adapter.adapt_task_start_from_event(event)
                collected_nodes.append({"evt": evt, "runtime": None})
                yield evt

            # ── 节点结束 ──
            elif evt_type == "on_chain_end" and name in _GRAPH_NODES:
                evt = self.adapter.adapt_task_end_from_event(event)
                for n in collected_nodes:
                    if n["evt"]["node"]["name"] == name and n["runtime"] is None:
                        n["evt"] = evt
                        n["runtime"] = evt["runtime"]
                        break
                else:
                    collected_nodes.append({"evt": evt, "runtime": evt["runtime"]})
                yield evt

            # ── LLM Token 收集 ──
            elif evt_type == "on_chat_model_end":
                data = event.get("data", {})
                output = data.get("output")
                if not output:
                    continue
                usage = getattr(output, "usage_metadata", None)
                if not usage:
                    continue
                node_name = langgraph_node or name
                model_name = getattr(output, "response_metadata", {}).get("model_name", "unknown")
                self.adapter.record_llm_call(
                    node_name=node_name,
                    model=model_name,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    elapsed=0,
                )

            # ── 回答流式输出（on_chat_model_stream）──
            elif evt_type == "on_chat_model_stream":
                # 只追踪回答节点的流式输出
                if langgraph_node not in _ANSWER_NODES:
                    continue
                answer_streaming = True
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    content = chunk.content
                    if isinstance(content, str):
                        answer_text += content
                        yield self.adapter.build_answer_chunk(answer_text)

        # ── 如果回答了但没流式（如 non-streaming LLM），从节点输出中提取 ──
        if not answer_streaming and collected_nodes:
            for n in reversed(collected_nodes):
                name = n["evt"]["node"]["name"]
                if name in _ANSWER_NODES:
                    # 从 on_chain_end 的输出中提取回答
                    runtime = n.get("runtime") or n["evt"].get("runtime", {})
                    state_diff = runtime.get("state_diff", {}) if isinstance(runtime, dict) else {}
                    added = state_diff.get("added", {}) if isinstance(state_diff, dict) else {}
                    for val in added.values():
                        if isinstance(val, str) and len(val) > 10:
                            answer_text = val
                            break
                    if answer_text:
                        yield self.adapter.build_answer_chunk(answer_text)
                    break

        # ── 汇总 ───────────────────────────────────────
        nodes_for_summary = [
            {
                "node": n["evt"]["node"],
                "runtime": n.get("runtime") or n["evt"]["runtime"],
            }
            for n in collected_nodes
        ]
        yield self.adapter.build_summary(nodes_for_summary)
