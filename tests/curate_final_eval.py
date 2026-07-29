"""Curate eval_hard_final.json down to a lean, well-balanced 20-case set.

Composition target:
  5 metadata_cross   (studio+year, director filmography)
  4 longtail_fact    (score+year of low-vote anime)
  4 similar_recommendation (natural-language 'anime like X' queries)
  3 bangumi_tags     (community-specific tags, Bangumi-only info)
  3 bangumi_score_precise (one-decimal Bangumi rating)
  1 kb_boundary      (field not in DB)

Total: 20 cases. All existing items in eval_hard_final.json are the
candidate pool; this script picks the strongest N per type and writes back.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tests" / "eval_hard_final.json"

TARGET = {
    "metadata_cross": 3,
    "longtail_fact": 3,
    "similar_recommendation": 4,
    "bangumi_tags": 3,
    "bangumi_score_precise": 3,
    "kb_boundary": 1,
}

# Ordering hints: prioritise these IDs (previously curated / interesting) first.
PRIORITY_IDS = {
    # keep the studio-year metadata cases already crafted
    "M001", "M002", "M003", "M004", "M005", "M006",
    # keep director-filmography cases
    "M007", "M008", "M009", "M010",
    # newer Bangumi cases (all worth showing)
    "T000440", "T001453", "T000247", "T004583", "T175599", "T003627",
    "S072941", "S129807", "S131891", "S001608", "S000310", "S531159",
    "R000793", "R000298", "R216371", "R000295", "R000822", "R012426",
    # boundary
    "B001", "B002",
}

cases = json.loads(SRC.read_text(encoding="utf-8"))

picked: dict[str, list[dict]] = {t: [] for t in TARGET}
for c in cases:
    t = c["type"]
    if t not in picked:
        continue
    picked[t].append(c)

# Prefer priority-listed IDs first, then any remaining
final: list[dict] = []
for t, n in TARGET.items():
    pool = picked.get(t, [])
    pool.sort(key=lambda c: (0 if c["id"] in PRIORITY_IDS else 1, c["id"]))
    kept = pool[:n]
    final.extend(kept)
    print(f"{t}: kept {len(kept)}/{len(pool)} -> {[c['id'] for c in kept]}")

SRC.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\ntotal: {len(final)} cases written to {SRC.name}")
