"""Preview only the newly added Bangumi cases in eval_hard_final.json."""
import json
from pathlib import Path

cases = json.loads((Path(__file__).parent / "eval_hard_final.json").read_text(encoding="utf-8"))
new_types = {"bangumi_tags", "bangumi_score_precise", "similar_recommendation"}

for c in cases:
    if c["type"] not in new_types:
        continue
    print(f"[{c['id']}] {c['type']} / {c['difficulty']}")
    print(f"  Q: {c['query']}")
    print(f"  gold_kw: {c['gold_answer_keywords']}")
    if c["type"] == "bangumi_tags":
        ev = c["gold_evidence"]
        print(f"  all_tags (first 10): {ev['all_tags'][:10]}")
    elif c["type"] == "similar_recommendation":
        ev = c["gold_evidence"]
        print(f"  anchor_tag: {ev['anchor_tag']}")
        print(f"  candidate_similar: {ev['candidate_similar_titles'][:6]}")
    print()
