"""Append-mode runner for hard ablation.

Purpose: `tests/eval_hard_final.json` has grown from 17 to 50 cases (33 new
cold-KB questions). The existing `tests/hard_eval_results.json` already has
H0-H6 results for the original 17 IDs. This script runs each variant on
ONLY the missing IDs, appends to `details_hX`, and recomputes summaries.

No LLM calls are wasted on cases that were already scored.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.run_hard_ablation import (
    EVAL_PATH, OUT_PATH, run_variant, summarize,
)


VARIANTS = [
    ("H0", "details_h0", None),
    ("H1", "details_h1", 20),
    ("H2", "details_h2", 3),
    ("H3", "details_h3", 10),
    ("H4", "details_h4", 20),
    ("H5", "details_h5", 40),
    ("H6", "details_h6", 60),
]


def main() -> None:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    payload.setdefault("summary", {})
    payload["cases_meta"] = [{"id": c["id"], "type": c["type"]} for c in cases]

    def dump():
        OUT_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    for tag, key, top_k in VARIANTS:
        existing = payload.get(key) or []
        done_ids = {r["id"] for r in existing}
        missing = [c for c in cases if c["id"] not in done_ids]
        if not missing:
            print(f"--- {tag}: nothing to append ({len(existing)} already scored) ---")
            continue
        print(f"\n--- {tag} (top_k={top_k}): appending {len(missing)} new cases ---")
        for c in missing:
            new_results = run_variant([c], tag=tag, top_k=top_k)
            existing.extend(new_results)
            payload[key] = existing
            dump()
        payload["summary"][key] = summarize(tag, existing)
        dump()

    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()
