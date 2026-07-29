"""Swap B001 to a truly obscure anime."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "anime_data.db"
SRC = ROOT / "tests" / "eval_hard_final.json"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

cases = json.loads(SRC.read_text(encoding="utf-8"))
cases = [c for c in cases if c["id"] != "B001"]

refusal_kw = ["未提供", "未记录", "没有相关信息", "无法确认", "不清楚",
              "查不到", "无相关", "无法从", "不在", "not available",
              "no data", "无该信息"]

# Small-audience anime: exclude Doraemon / Conan / Pokemon / Anpanman.
row = conn.execute(
    """
    SELECT anime_id, anime_title FROM Anime
    WHERE score_count BETWEEN 200 AND 1000
      AND SUBSTR(release_date,1,4) BETWEEN '2010' AND '2018'
      AND LENGTH(anime_title) BETWEEN 4 AND 10
      AND anime_title NOT LIKE '%哆啦A梦%'
      AND anime_title NOT LIKE '%柯南%'
      AND anime_title NOT LIKE '%宝可梦%'
      AND anime_title NOT LIKE '%皮卡丘%'
      AND anime_title NOT LIKE '%面包超人%'
      AND anime_title NOT LIKE '%蜡笔小新%'
      AND anime_title NOT LIKE '%迪士尼%'
      AND anime_title NOT LIKE '%航海王%'
      AND anime_title NOT LIKE '%海贼王%'
      AND anime_title NOT LIKE '%龙珠%'
      AND anime_title NOT LIKE '%银魂%'
      AND anime_title NOT LIKE '%火影%'
      AND anime_title NOT LIKE '%你的名字%'
    ORDER BY RANDOM() LIMIT 1
    """
).fetchone()

cases.append({
    "id": "B001",
    "type": "kb_boundary",
    "difficulty": "hard",
    "query": f"《{row['anime_title']}》原声带的作曲家是谁？请只根据知识库回答。",
    "gold_entities": [row["anime_title"]],
    "gold_answer_keywords": refusal_kw,
    "gold_evidence": {
        "note": "DB does not store OST composer",
        "correct_behavior": "acknowledge missing info; do not invent a composer name",
        "target_anime": row["anime_title"],
        "anime_id": row["anime_id"],
    },
    "expects_rag": False,
    "expects_refusal": True,
    "scoring": {
        "must_contain_refusal": True,
        "must_not_contain_specific_name": True,
        "rule": "answer must admit missing info; MUST NOT invent a composer name",
    },
})

SRC.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"B001 swapped -> 《{row['anime_title']}》")
