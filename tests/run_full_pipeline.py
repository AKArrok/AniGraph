"""Full-pipeline (B5) ablation on tests/eval_hard_final.json.

Runs the real agents/graph.build_graph() pipeline (alias_resolve, planner,
metadata_reasoner, retrieval, experts, evaluator, replan, web_fallback,
answer) so the components changed today actually get exercised. Scored with
the SAME tests/deterministic_scorer.score_case as H0/H1/H2, so B5 is directly
comparable to the dense-only baselines.

Records whether web_fallback fired so KB-only vs web-assisted can be split.
Saves after each case to tests/full_pipeline_results.json (resumable).

Usage:
  $env:PYTHONPATH="."; python tests/run_full_pipeline.py [--limit=N]
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agents.graph import build_graph
from tests.deterministic_scorer import score_case

EVAL = ROOT / "tests" / "eval_hard_final.json"
OUT = ROOT / "tests" / "full_pipeline_results.json"

WEB_MARK = "[" + "联网搜索结果" + "]"


def web_was_used(state: dict) -> bool:
    merged = state.get("merged_results", "") or ""
    if WEB_MARK in merged:
        return True
    return state.get("termination_reason") == "web_fallback" and WEB_MARK in merged


async def run_one(query: str, idx: int) -> dict:
    memory = MemorySaver()
    app = build_graph().compile(checkpointer=memory)
    t0 = time.time()
    resp = await app.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        {"configurable": {"thread_id": "b5_" + str(idx)}},
    )
    latency = time.time() - t0
    msgs = resp.get("messages", [])
    answer = msgs[-1].content if msgs else ""
    return {
        "answer": answer,
        "latency": latency,
        "web_used": web_was_used(resp),
        "termination_reason": resp.get("termination_reason", ""),
    }


def _summary(results: list) -> dict:
    if not results:
        return {}
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    kb_only = [r for r in results if not r["web_used"]]
    web_rows = [r for r in results if r["web_used"]]
    by_type: dict = {}
    for r in results:
        b = by_type.setdefault(r["type"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(r["correct"])
    for b in by_type.values():
        b["acc"] = round(b["correct"] / b["n"], 3)
    lat = [r["latency"] for r in results if r["latency"]]
    return {
        "name": "B5_full",
        "n": n,
        "strict_accuracy": correct / n,
        "kb_only_n": len(kb_only),
        "kb_only_correct": sum(1 for r in kb_only if r["correct"]),
        "web_used_n": len(web_rows),
        "web_used_correct": sum(1 for r in web_rows if r["correct"]),
        "avg_latency": round(statistics.mean(lat), 3) if lat else 0,
        "by_type": by_type,
    }


def _dump(results: list) -> None:
    payload = {"summary": _summary(results), "details": results}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _report(results: list) -> None:
    s = _summary(results)
    print("\n===== B5 Full Pipeline =====")
    print("Overall: {:.1%} ({} cases)".format(s["strict_accuracy"], s["n"]))
    print("  KB-only subset:  {}/{}".format(s["kb_only_correct"], s["kb_only_n"]))
    print("  Web-assisted:    {}/{}".format(s["web_used_correct"], s["web_used_n"]))
    print("By type:")
    for t, b in s["by_type"].items():
        print("  {}: {}/{} = {:.0%}".format(t, b["correct"], b["n"], b["acc"]))


async def main() -> None:
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=")[1])

    cases = json.loads(EVAL.read_text(encoding="utf-8"))
    if limit:
        cases = cases[:limit]

    results: list = []
    done: set = set()
    if OUT.exists():
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        results = prev.get("details", [])
        done = {r["id"] for r in results}

    for i, c in enumerate(cases):
        if c["id"] in done:
            print("[{}/{}] {} SKIP (cached)".format(i + 1, len(cases), c["id"]))
            continue
        print("[{}/{}] {} {} ...".format(i + 1, len(cases), c["id"], c["type"]), flush=True)
        try:
            r = await run_one(c["query"], i)
        except Exception as e:
            r = {"answer": "", "latency": 0.0, "web_used": False,
                 "termination_reason": "ERROR: " + str(e)}
            print("    ERROR:", e)
        score = score_case(c, r["answer"] or "")
        row = {
            "id": c["id"], "type": c["type"], "query": c["query"],
            "answer": r["answer"], "correct": score["correct"],
            "partial_score": score.get("partial_score", 0.0),
            "score_detail": score, "latency": round(r["latency"], 3),
            "web_used": r["web_used"], "termination_reason": r["termination_reason"],
        }
        results.append(row)
        _dump(results)
        mark = "PASS" if score["correct"] else "FAIL"
        web = " [WEB]" if r["web_used"] else ""
        print("    {}{}  {:.1f}s  {}".format(mark, web, r["latency"], repr((r["answer"] or "")[:60])))

    _dump(results)
    _report(results)


if __name__ == "__main__":
    asyncio.run(main())
