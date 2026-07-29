"""Print a compact ablation summary from ablation_results.json."""
import json
from pathlib import Path

p = Path(__file__).parent / "ablation_results.json"
d = json.loads(p.read_text(encoding="utf-8"))
s = d.get("summary", {})

order = [
    "B0_direct_llm",
    "B1_dense_rag",
    "B2_dense_topk3",
    "B3_dense_topk10",
    "B4_no_cite_prompt",
]
print(f"{'exp':<22} {'n':>3} {'acc':>7} {'faith':>7} {'lat':>7} {'p95':>7} {'tok':>6}")
for k in order:
    row = s.get(k)
    if not row:
        print(f"{k:<22}  --")
        continue
    print(
        f"{k:<22} {row['n']:>3} "
        f"{row['accuracy']:>7.1%} "
        f"{row['faithfulness']:>7.3f} "
        f"{row['avg_latency']:>7.2f} "
        f"{row['p95_latency']:>7.2f} "
        f"{row['avg_tokens']:>6}"
    )
