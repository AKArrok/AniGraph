"""Reset B3/B4 sections (B2 already completed) so run_ablation.py reruns them.

If an argument is passed with a specific section name, only that section is
cleared.
"""
import json
import sys
from pathlib import Path

p = Path(__file__).parent / "ablation_results.json"
data = json.loads(p.read_text(encoding="utf-8"))

targets = sys.argv[1:] or ["b3", "b4"]
for t in targets:
    data.pop(f"details_{t}", None)
    key_map = {
        "b2": "B2_dense_topk3",
        "b3": "B3_dense_topk10",
        "b4": "B4_no_cite_prompt",
    }
    data.get("summary", {}).pop(key_map.get(t, ""), None)

p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"cleared sections: {targets}")
