"""Print eval_hard_final.json in a compact, readable form."""
import json
from pathlib import Path

cases = json.loads((Path(__file__).parent / "eval_hard_final.json").read_text(encoding="utf-8"))

TYPE_ORDER = [
    "metadata_cross", "longtail_fact", "similar_recommendation",
    "bangumi_tags", "bangumi_score_precise", "kb_boundary",
]
by_type: dict[str, list[dict]] = {t: [] for t in TYPE_ORDER}
for c in cases:
    by_type.setdefault(c["type"], []).append(c)

i = 0
for t in TYPE_ORDER:
    for c in by_type.get(t, []):
        i += 1
        print(f"\n{i:2}. [{c['id']}] {c['type']} ({c['difficulty']})")
        print(f"    Q: {c['query']}")
        print(f"    gold_kw: {c['gold_answer_keywords']}")
        ev = c.get("gold_evidence", {})
        if "titles" in ev:
            print(f"    gold_titles: {ev['titles']}")
        if "score" in ev and "release_date" in ev:
            print(f"    gold_score: {ev['score']}  release_date: {ev['release_date']}")
        if "anchor_tag" in ev:
            print(f"    anchor_tag: {ev['anchor_tag']}")
            print(f"    similar_pool: {ev['candidate_similar_titles'][:6]}")
        if "all_tags" in ev:
            print(f"    all_tags (first 10): {ev['all_tags'][:10]}")
        if "target_anime" in ev:
            print(f"    target: {ev['target_anime']}  (DB has no OST field)")
