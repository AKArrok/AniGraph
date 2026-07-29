"""Widened-pool rescore for the three component ablation runs (A1/A2/A3).

Same policy as tests/rescore_similar.py: for every similar_recommendation
row, replace the shipped gold_pool with all non-seed anime that carry the
anchor_tag with score_count >= 500. No LLM/embedding calls.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tests.deterministic_scorer import _contains

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "anime_data.db"
EVAL = ROOT / "tests" / "eval_hard_final.json"

FILES = [
    ROOT / "tests" / "component_ablation_A1_results.json",
    ROOT / "tests" / "component_ablation_A2_results.json",
    ROOT / "tests" / "component_ablation_A3_results.json",
]

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row


def expand_pool(anchor_tag, seed_id):
    rows = conn.execute(
        """
        SELECT DISTINCT a.anime_title
        FROM Anime a
        JOIN Anime_Category ac ON ac.anime_id = a.anime_id
        JOIN Category cat ON cat.category_id = ac.category_id
        WHERE cat.category_name = ?
          AND a.anime_id != ?
          AND a.score_count >= 500
        """, (anchor_tag, seed_id)).fetchall()
    return [r["anime_title"] for r in rows]


def rescore(answer, seed_title, pool, min_hits=2):
    hits = [c for c in pool if c != seed_title and _contains(answer, c)]
    return {
        "correct": len(hits) >= min_hits,
        "partial_score": min(1.0, len(hits) / max(1, min_hits)),
        "hits": hits[:10],
        "n_hits": len(hits),
    }


def main():
    eval_cases = {c["id"]: c for c in json.loads(EVAL.read_text(encoding="utf-8"))}
    for path in FILES:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = 0
        for r in payload.get("details", []):
            if r["type"] != "similar_recommendation":
                continue
            case = eval_cases.get(r["id"])
            if not case:
                continue
            ev = case["gold_evidence"]
            pool = expand_pool(ev["anchor_tag"], ev["seed_anime_id"])
            new = rescore(r.get("answer") or "", ev["seed_title"], pool, min_hits=2)
            if r.get("correct") != new["correct"]:
                changed += 1
            r["correct"] = new["correct"]
            r["partial_score"] = new["partial_score"]
            r["score_detail"] = {**new, "anchor_tag": ev["anchor_tag"],
                                  "expanded_pool_size": len(pool),
                                  "note": "rescored with widened pool"}
        rows = payload.get("details", [])
        n = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        by_type = {}
        for r in rows:
            b = by_type.setdefault(r["type"], {"n": 0, "correct": 0})
            b["n"] += 1
            b["correct"] += int(r["correct"])
        for b in by_type.values():
            b["acc"] = round(b["correct"] / b["n"], 3)
        payload["summary"]["strict_accuracy"] = correct / n
        payload["summary"]["by_type"] = by_type
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.name}: rescored {changed} rows, acc now {correct}/{n} = {correct/n:.1%}")


if __name__ == "__main__":
    main()
