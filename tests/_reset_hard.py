"""Drop appended Bangumi cases from eval_hard_final.json so we can regenerate cleanly."""
import json
from pathlib import Path

p = Path(__file__).parent / "eval_hard_final.json"
cases = json.loads(p.read_text(encoding="utf-8"))
before = len(cases)
cases = [c for c in cases if c["type"] not in {
    "bangumi_tags", "bangumi_score_precise", "similar_recommendation",
}]
p.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"reset: {before} -> {len(cases)}")
