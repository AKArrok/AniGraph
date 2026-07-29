"""Rescore B5 full_pipeline_results.json similar_recommendation with the SAME
widened pool used by tests/rescore_similar.py, so B5 is directly comparable to
the rescored H1. No LLM/embedding calls; DB + saved answers only.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tests.deterministic_scorer import _contains

DB = ROOT / "data" / "anime_data.db"
RESULTS = ROOT / "tests" / "full_pipeline_results.json"
EVAL = ROOT / "tests" / "eval_hard_final.json"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row


def expand_pool(anchor_tag: str, seed_id: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT a.anime_title
        FROM Anime a
        JOIN Anime_Category ac ON ac.anime_id = a.anime_id
        JOIN Category cat ON cat.category_id = ac.category_id
        WHERE cat.category_name = ? AND a.anime_id != ? AND a.score_count >= 500
        """, (anchor_tag, seed_id)).fetchall()
    return [r["anime_title"] for r in rows]


def main() -> None:
    eval_cases = {c["id"]: c for c in json.loads(EVAL.read_text(encoding="utf-8"))}
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    changed = 0
    for r in data["details"]:
        if r["type"] != "similar_recommendation":
            continue
        case = eval_cases.get(r["id"])
        ev = case["gold_evidence"]
        pool = expand_pool(ev["anchor_tag"], ev["seed_anime_id"])
        hits = [c for c in pool if c != ev["seed_title"] and _contains(r["answer"] or "", c)]
        new_correct = len(hits) >= 2
        if new_correct != r["correct"]:
            changed += 1
            print("  [{}] {}->{}  hits={} pool={}".format(
                r["id"], r["correct"], "T" if new_correct else "F", len(hits), len(pool)))
        r["correct"] = new_correct
        r["partial_score"] = min(1.0, len(hits) / 2)
        r["score_detail"] = {"correct": new_correct, "hits": hits[:10], "n_hits": len(hits),
                             "anchor_tag": ev["anchor_tag"], "expanded_pool_size": len(pool),
                             "note": "rescored with widened pool (matches H1)"}

    # recompute summary
    rows = data["details"]
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    kb_only = [r for r in rows if not r["web_used"]]
    web_rows = [r for r in rows if r["web_used"]]
    by_type: dict = {}
    for r in rows:
        b = by_type.setdefault(r["type"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(r["correct"])
    for b in by_type.values():
        b["acc"] = round(b["correct"] / b["n"], 3)
    data["summary"]["strict_accuracy"] = correct / n
    data["summary"]["kb_only_correct"] = sum(1 for r in kb_only if r["correct"])
    data["summary"]["web_used_correct"] = sum(1 for r in web_rows if r["correct"])
    data["summary"]["by_type"] = by_type

    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nrescored {} similar rows".format(changed))
    print("B5 overall: {}/{} = {:.1%}".format(correct, n, correct / n))
    for t, b in by_type.items():
        print("  {}: {}/{} = {:.0%}".format(t, b["correct"], b["n"], b["acc"]))


if __name__ == "__main__":
    main()
