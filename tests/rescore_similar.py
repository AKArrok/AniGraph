"""Rescore similar_recommendation cases in tests/hard_eval_results.json using
a widened candidate pool: any anime that shares the anchor_tag with the
seed (score_count >= 500), not just the top-12 that shipped in gold_pool.

Does not call any LLM or embedding API. Runs against saved answers only.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tests.deterministic_scorer import _contains

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "anime_data.db"
RESULTS = ROOT / "tests" / "hard_eval_results.json"
EVAL = ROOT / "tests" / "eval_hard_final.json"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row


def expand_pool(anchor_tag: str, seed_id: int) -> list[str]:
    """All non-seed anime carrying anchor_tag with modest audience."""
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


def rescore(answer: str, seed_title: str, expanded_pool: list[str], min_hits: int = 2) -> dict:
    hits: list[str] = []
    for c in expanded_pool:
        if c == seed_title:
            continue
        if _contains(answer, c):
            hits.append(c)
    return {
        "correct": len(hits) >= min_hits,
        "partial_score": min(1.0, len(hits) / max(1, min_hits)),
        "hits": hits[:10],
        "n_hits": len(hits),
    }


def main() -> None:
    eval_cases = {c["id"]: c for c in json.loads(EVAL.read_text(encoding="utf-8"))}
    data = json.loads(RESULTS.read_text(encoding="utf-8"))

    changed = 0
    for section in ("details_h0", "details_h1", "details_h2", "details_h3", "details_h4", "details_h5", "details_h6"):
        rows = data.get(section, [])
        for r in rows:
            if r["type"] != "similar_recommendation":
                continue
            case = eval_cases.get(r["id"])
            if not case:
                continue
            ev = case["gold_evidence"]
            pool = expand_pool(ev["anchor_tag"], ev["seed_anime_id"])
            new_score = rescore(r["answer"] or "", ev["seed_title"], pool, min_hits=2)
            old_correct = r["correct"]
            r["correct"] = new_score["correct"]
            r["partial_score"] = new_score["partial_score"]
            r["score_detail"] = {
                **new_score,
                "anchor_tag": ev["anchor_tag"],
                "expanded_pool_size": len(pool),
                "note": "rescored with widened pool",
            }
            if old_correct != new_score["correct"]:
                changed += 1
                mark = "T" if new_score["correct"] else "F"
                print(f"  [{section} {r['id']}] {old_correct}->{mark}  hits={new_score['n_hits']} pool={len(pool)}")

    # Recompute per-section summary
    for section in ("details_h0", "details_h1", "details_h2", "details_h3", "details_h4", "details_h5", "details_h6"):
        rows = data.get(section, [])
        if not rows:
            continue
        n = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        by_type: dict[str, dict] = {}
        for r in rows:
            b = by_type.setdefault(r["type"], {"n": 0, "correct": 0})
            b["n"] += 1
            b["correct"] += int(r["correct"])
        for b in by_type.values():
            b["acc"] = round(b["correct"] / b["n"], 3)
        summary_key = section.replace("details_", "")
        summary_key = {"h0": "H0", "h1": "H1", "h2": "H2", "h3": "H3", "h4": "H4", "h5": "H5", "h6": "H6"}.get(summary_key, summary_key)
        # find its summary entry
        sec_summary = data.get("summary", {}).get(section.replace("details_", ""))
        if isinstance(sec_summary, dict):
            sec_summary["strict_accuracy"] = correct / n
            sec_summary["by_type"] = by_type

    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nrescored {changed} similar_recommendation rows")
    for section in ("details_h0", "details_h1", "details_h2", "details_h3", "details_h4", "details_h5", "details_h6"):
        rows = data.get(section, [])
        if not rows:
            continue
        n = len(rows)
        c = sum(1 for r in rows if r["correct"])
        print(f"  {section}: {c}/{n} = {c/n:.1%}")


if __name__ == "__main__":
    main()
