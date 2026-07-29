import json
p = "tests/ablation_results.json"
d = json.load(open(p, encoding="utf-8"))
for k in ("details_b0", "details_b1", "details_b1_partial"):
    if k in d: del d[k]
d["summary"] = {}
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("Reset B0+B1")
