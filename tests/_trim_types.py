"""Trim metadata_cross to 3 and longtail_fact to 3 in place."""
import json
from pathlib import Path

p = Path(__file__).parent / "eval_hard_final.json"
cases = json.loads(p.read_text(encoding="utf-8"))

LIMITS = {"metadata_cross": 3, "longtail_fact": 3}
counts: dict[str, int] = {}
kept: list[dict] = []
for c in cases:
    t = c["type"]
    limit = LIMITS.get(t)
    if limit is not None:
        counts[t] = counts.get(t, 0) + 1
        if counts[t] > limit:
            continue
    kept.append(c)

p.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(cases)} -> {len(kept)} cases")
for t, n in [(c["type"], 1) for c in kept]:
    pass
from collections import Counter
print(Counter(c["type"] for c in kept))
