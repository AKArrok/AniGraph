from types import SimpleNamespace

import asyncio

from trace import TraceCollector


class _FakeApp:
    async def astream_events(self, input_state, config, version):
        for content in ("第一", "段", "回答"):
            yield {
                "event": "on_chat_model_stream",
                "name": "fake_model",
                "metadata": {"langgraph_node": "answer"},
                "data": {"chunk": SimpleNamespace(content=content)},
            }


def test_trace_collector_emits_incremental_answer_chunks():
    async def collect():
        collector = TraceCollector()
        return [event async for event in collector.collect(_FakeApp(), {}, {})]

    events = asyncio.run(collect())
    chunks = [event["answer_text"] for event in events if event["type"] == "answer_chunk"]

    assert chunks == ["第一", "段", "回答"]
    assert "".join(chunks) == "第一段回答"
    assert sum(event["type"] == "done" for event in events) == 1
