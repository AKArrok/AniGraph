"""Show which cases have empty answers and inspect their token usage."""
import json
from pathlib import Path

data = json.loads((Path(__file__).parent / "ablation_results.json").read_text(encoding="utf-8"))
for section in ("details_b1", "details_b2", "details_b3", "details_b4"):
    print(f"\n=== {section} ===")
    for r in data.get(section, []):
        ans = r.get("answer") or ""
        if len(ans.strip()) < 5:
            tot = r.get("tokens")
            comp = r.get("completion_tokens")
            reason = r.get("judge", {}).get("reason", "")[:80]
            print(f"  {r['id']:6} ans_len={len(ans)}  total={tot} comp={comp}  judge={reason}")
