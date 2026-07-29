"""Run a real SSE chat request and report streaming latency and output."""

from __future__ import annotations

import json
import sys
import time

import requests


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "推荐几部类似命运石之门的悬疑科幻动画，排除同系列作品"
    thread_id = sys.argv[2] if len(sys.argv) > 2 else f"stream-probe-{time.time_ns()}"
    started = time.perf_counter()
    first_chunk_at: float | None = None
    chunks: list[str] = []
    done_count = 0
    event_counts: dict[str, int] = {}
    node_timings: list[tuple[str, float]] = []

    with requests.post(
        "http://127.0.0.1:9527/chat/stream",
        json={"query": query, "thread_id": thread_id},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=(10, 180),
    ) as response:
        response.raise_for_status()
        event_type = "message"
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                event_type = "message"
                continue
            if raw_line.startswith("event: "):
                event_type = raw_line[7:]
                continue
            if not raw_line.startswith("data: "):
                continue

            event = json.loads(raw_line[6:])
            actual_type = event.get("type", event_type)
            event_counts[actual_type] = event_counts.get(actual_type, 0) + 1
            if actual_type == "answer_chunk":
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                chunks.append(event.get("answer_text", ""))
            elif actual_type == "done":
                done_count += 1
            elif actual_type == "node_end":
                node = event.get("node") or {}
                runtime = event.get("runtime") or {}
                duration = max(0, runtime.get("end", 0) - runtime.get("start", 0))
                node_timings.append((node.get("name", "unknown"), round(duration, 3)))
            elif actual_type == "error":
                print("ERROR:", event, flush=True)

    ended = time.perf_counter()
    print(f"QUERY: {query}")
    print(f"THREAD_ID: {thread_id}")
    print(f"TTFT_SECONDS: {(first_chunk_at - started) if first_chunk_at else -1:.3f}")
    print(f"TOTAL_SECONDS: {ended - started:.3f}")
    print(f"ANSWER_CHUNKS: {len(chunks)}")
    print(f"DONE_COUNT: {done_count}")
    print(f"EVENT_COUNTS: {json.dumps(event_counts, ensure_ascii=False)}")
    print(f"NODE_TIMINGS: {json.dumps(node_timings, ensure_ascii=False)}")
    print("ANSWER_BEGIN")
    print("".join(chunks))
    print("ANSWER_END")
    return 0 if chunks and done_count == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
