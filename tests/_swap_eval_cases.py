"""Swap out the two risky cases in eval_hard_final.json:
  B001 (你的名字。) - too famous, LLM knows OST is RADWIMPS
  L001307 (星际海盗) - generic name, LLM can guess
  L067745 (一千零一夜) - generic name, LLM can guess

Replace with obscure anime queried live from the DB.
"""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "anime_data.db"
SRC = ROOT / "tests" / "eval_hard_final.json"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

cases = json.loads(SRC.read_text(encoding="utf-8"))

REMOVE_IDS = {"B001", "L001307", "L067745"}
cases = [c for c in cases if c["id"] not in REMOVE_IDS]
print(f"after removal: {len(cases)} cases")

refusal_kw = ["未提供", "未记录", "没有相关信息", "无法确认", "不清楚",
              "查不到", "无相关", "无法从", "不在", "not available",
              "no data", "无该信息"]

# Replacement B001: obscure anime, OST composer question.
obscure = conn.execute(
    """
    SELECT anime_id, anime_title FROM Anime
    WHERE score_count BETWEEN 400 AND 1500
      AND SUBSTR(release_date,1,4) BETWEEN '2010' AND '2018'
      AND LENGTH(anime_title) BETWEEN 3 AND 12
    ORDER BY RANDOM() LIMIT 1
    """
).fetchone()
cases.append({
    "id": "B001",
    "type": "kb_boundary",
    "difficulty": "hard",
    "query": f"《{obscure['anime_title']}》原声带的作曲家是谁？请只根据知识库回答。",
    "gold_entities": [obscure["anime_title"]],
    "gold_answer_keywords": refusal_kw,
    "gold_evidence": {
        "note": "DB does not store OST composer",
        "correct_behavior": "acknowledge missing info; do not invent a composer name",
        "target_anime": obscure["anime_title"],
        "anime_id": obscure["anime_id"],
    },
    "expects_rag": False,
    "expects_refusal": True,
    "scoring": {
        "must_contain_refusal": True,
        "must_not_contain_specific_name": True,
        "rule": "answer must admit missing info; MUST NOT invent a composer name",
    },
})
print(f"B001 replaced -> OST question about 《{obscure['anime_title']}》")

# Replace two longtail cases with new obscure picks.
existing_ids = {c["id"] for c in cases}
picks = conn.execute(
    """
    SELECT anime_id, anime_title, score, score_count, release_date
    FROM Anime
    WHERE score IS NOT NULL AND score_count BETWEEN 30 AND 200
      AND release_date IS NOT NULL
      AND SUBSTR(release_date,1,4) BETWEEN '1995' AND '2020'
      AND score >= 5.8
      AND LENGTH(anime_title) >= 6
      AND anime_title NOT LIKE '%星际海盗%'
      AND anime_title NOT LIKE '%一千零一%'
    ORDER BY RANDOM() LIMIT 12
    """
).fetchall()
added = 0
for r in picks:
    aid = f"L{r['anime_id']:06d}"
    if aid in existing_ids:
        continue
    year = r["release_date"][:4]
    cases.append({
        "id": aid,
        "type": "longtail_fact",
        "difficulty": "hard",
        "query": f"《{r['anime_title']}》的评分是多少？上映年份是哪一年？",
        "gold_entities": [r["anime_title"]],
        "gold_answer_keywords": [f"{r['score']:.1f}", year],
        "gold_evidence": {
            "anime_id": r["anime_id"],
            "score": r["score"],
            "score_count": r["score_count"],
            "release_date": r["release_date"],
        },
        "expects_rag": True,
        "scoring": {
            "score_exact": f"{r['score']:.1f}",
            "year_exact": year,
            "rule": "answer must contain BOTH the exact score AND the year",
        },
    })
    added += 1
    if added >= 2:
        break

SRC.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"final count: {len(cases)}")
