"""Inspect failed H1 cases from hard_eval_results.json."""
import json
from pathlib import Path

data = json.loads((Path(__file__).parent / "hard_eval_results.json").read_text(encoding="utf-8"))
for r in data.get("details_h1", []):
    if r["correct"]:
        continue
    print(f"\n=== {r['id']} ({r['type']}) ===")
    print(f"Q: {r['query']}")
    print(f"A: {(r['answer'] or '')[:600]}")
    print(f"score_detail: {r['score_detail']}")
