"""Component ablation on tests/eval_hard_final.json.

Runs the full 15-node pipeline with a single component disabled per variant:
  A1  no fast-path            simple_fact still goes through planner->experts->answer
  A2  no evaluator conflict   evaluator skips the LLM-based conflict judgement
  A3  no query rewrite        query_processor forces strategy=direct

Switching is done via env vars (see config.ABLATION_*). Each variant lives in
its own results file so B5 baseline stays untouched.

Usage:
  $env:PYTHONPATH="."; python tests/run_component_ablation.py A1 A2 A3
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
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

VARIANTS: dict = {
    "A1": {
        "env": {"ABLATION_NO_FAST_PATH": "true"},
        "out": ROOT / "tests" / "component_ablation_A1_results.json",
        "desc": "no simple_fact fast path",
    },
    "A2": {
        "env": {"ABLATION_NO_EVALUATOR_CONFLICT": "true"},
        "out": ROOT / "tests" / "component_ablation_A2_results.json",
        "desc": "no evaluator conflict LLM",
    },
    "A3": {
        "env": {"ABLATION_NO_QUERY_REWRITE": "true"},
        "out": ROOT / "tests" / "component_ablation_A3_results.json",
        "desc": "no query rewrite (force direct)",
    },
}

EVAL = ROOT / "tests" / "eval_hard_final.json"
WEB_MARK = "[" + "联网搜索结果" + "]"


def web_was_used(state: dict) -> bool:
    merged = state.get("merged_results", "") or ""
    if WEB_MARK in merged:
        return True
    return state.get("termination_reason") == "web_fallback" and WEB_MARK in merged


def reload_pipeline():
    import config as cfg
    importlib.reload(cfg)
    for mod in (
        "agents.evaluator",
        "agents.query_processor",
        "agents.simple_fact_answer",
        "agents.planner",
        "agents.retrieval",
        "agents.graph",
    ):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    from agents.graph import build_graph
    return build_graph


async def run_one(build_graph, query: str, idx: int) -> dict:
    memory = MemorySaver()
    app = build_graph().compile(checkpointer=memory)
    t0 = time.time()
    resp = await app.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        {"configurable": {"thread_id": f"comp_{idx}"}},
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


def summary(rows):
    if not rows:
        return {}
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    by_type = {}
    for r in rows:
        b = by_type.setdefault(r["type"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(r["correct"])
    for b in by_type.values():
        b["acc"] = round(b["correct"] / b["n"], 3)
    lat = [r["latency"] for r in rows if r["latency"]]
    return {
        "n": n,
        "strict_accuracy": correct / n,
        "avg_latency": round(statistics.mean(lat), 3) if lat else 0,
        "by_type": by_type,
    }


def dump(out, rows):
    payload = {"summary": summary(rows), "details": rows}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_variant(tag):
    v = VARIANTS[tag]
    # Reset all ablation env vars first, then flip this variant on
    for other_v in VARIANTS.values():
        for k in other_v["env"]:
            os.environ[k] = "false"
    for k, val in v["env"].items():
        os.environ[k] = val

    build_graph = reload_pipeline()
    from tests.deterministic_scorer import score_case

    cases = json.loads(EVAL.read_text(encoding="utf-8"))
    limit_arg = os.environ.get("COMPONENT_LIMIT")
    if limit_arg:
        try:
            cases = cases[: int(limit_arg)]
        except Exception:
            pass
    out = v["out"]
    rows = []
    done = set()
    if out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
            rows = prev.get("details", [])
            done = {r["id"] for r in rows}
        except Exception:
            pass

    print(f"\n===== {tag} ({v['desc']}) =====")
    for i, c in enumerate(cases):
        if c["id"] in done:
            print(f"  [{i+1:>2}/{len(cases)}] {c['id']} SKIP (cached)")
            continue
        print(f"  [{i+1:>2}/{len(cases)}] {c['id']} {c['type']} ...", flush=True)
        try:
            r = await run_one(build_graph, c["query"], i)
        except Exception as e:
            print(f"    ERROR: {e}")
            r = {"answer": "", "latency": 0.0, "web_used": False,
                 "termination_reason": f"ERROR: {e}"}
        score = score_case(c, r["answer"] or "")
        row = {
            "id": c["id"], "type": c["type"], "query": c["query"],
            "answer": r["answer"], "correct": score["correct"],
            "partial_score": score.get("partial_score", 0.0),
            "score_detail": score, "latency": round(r["latency"], 3),
            "web_used": r["web_used"], "termination_reason": r["termination_reason"],
        }
        rows.append(row)
        dump(out, rows)
        mark = "PASS" if score["correct"] else "FAIL"
        preview = (r["answer"] or "")[:60]
        print(f"    {mark}  {r['latency']:.1f}s  {preview!r}")

    dump(out, rows)
    s = summary(rows)
    print(f"\n{tag} summary: acc={s['strict_accuracy']:.1%}  "
          f"lat={s['avg_latency']:.1f}s  by_type={s['by_type']}")


async def main():
    tags = [a for a in sys.argv[1:] if a in VARIANTS]
    if not tags:
        tags = list(VARIANTS.keys())
    for tag in tags:
        await run_variant(tag)


if __name__ == "__main__":
    asyncio.run(main())
