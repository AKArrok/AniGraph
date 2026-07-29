"""Swap R000295 (Ghibli) and B001 (Toriko OST) with harder alternatives."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "anime_data.db"
SRC = ROOT / "tests" / "eval_hard_final.json"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

cases = json.loads(SRC.read_text(encoding="utf-8"))
# Drop any previously-swapped-in cases (R000295 original ghibli one, plus the
# 命运石之门 iteration if it exists) and the boundary case.
DROP_IDS = {"R000295", "R010380", "B001"}
cases = [c for c in cases if c["id"] not in DROP_IDS]

# --- New R: seed anime with a distinctive non-Ghibli tag ------------------
# Pick a seed like 命运石之门 / 心理测量者 / 齐木楠雄 - something with clear
# stylistic anchor that's NOT "宮崎駿" or "吉卜力".
SEED_CANDIDATES = [
    ("排球少年!!", 63432),         # sports
    ("虫师 续章", 105969),         # atmospheric folklore
    ("四畳半神話大系", 4917),      # yuasa/experimental
    ("心理测量者", 27364),          # cyber dystopia
    ("齐木楠雄的灾难", 132734),
]
EXCLUDED_TAGS = {"剧情", "神作", "经典", "神配乐"}  # already used by other R cases
seed_row = None
for title, aid in SEED_CANDIDATES:
    r = conn.execute("SELECT anime_id, anime_title FROM Anime WHERE anime_id=?", (aid,)).fetchone()
    if r:
        seed_row = r
        break

if seed_row is None:
    seed_row = conn.execute(
        "SELECT anime_id, anime_title FROM Anime WHERE score>=8.4 AND score_count>=8000 ORDER BY RANDOM() LIMIT 1"
    ).fetchone()

seed_id = seed_row["anime_id"]
seed_title = seed_row["anime_title"]

anchor = conn.execute(
    """
    SELECT cat.category_name
    FROM Anime_Category ac JOIN Category cat ON cat.category_id = ac.category_id
    WHERE ac.anime_id = ?
      AND cat.category_name NOT IN ('TV','TVA','日本','日本动画','动画','漫改',
                                     '漫画改','原创','续作','剧场版')
      AND cat.category_name NOT LIKE '%宮崎%'
      AND cat.category_name NOT LIKE '%吉卜力%'
      AND LENGTH(cat.category_name) BETWEEN 2 AND 6
    ORDER BY RANDOM() LIMIT 1
    """, (seed_id,)).fetchone()

# Force a distinctive tag: retry a few times if we land on a used one
for _ in range(20):
    if anchor and anchor["category_name"] not in EXCLUDED_TAGS:
        break
    anchor = conn.execute(
        """
        SELECT cat.category_name
        FROM Anime_Category ac JOIN Category cat ON cat.category_id = ac.category_id
        WHERE ac.anime_id = ?
          AND cat.category_name NOT IN ('TV','TVA','日本','日本动画','动画','漫改',
                                         '漫画改','原创','续作','剧场版','剧情',
                                         '神作','经典','神配乐')
          AND LENGTH(cat.category_name) BETWEEN 2 AND 6
        ORDER BY RANDOM() LIMIT 1
        """, (seed_id,)).fetchone()

tag_name = anchor["category_name"]
sibs = conn.execute(
    """
    SELECT DISTINCT a.anime_title
    FROM Anime a
    JOIN Anime_Category ac ON ac.anime_id = a.anime_id
    JOIN Category cat ON cat.category_id = ac.category_id
    WHERE cat.category_name = ?
      AND a.anime_id != ?
      AND a.score >= 7.5
      AND a.score_count >= 2000
    ORDER BY a.score DESC LIMIT 12
    """, (tag_name, seed_id)).fetchall()
sib_titles = [s["anime_title"] for s in sibs]

cases.append({
    "id": f"R{seed_id:06d}",
    "type": "similar_recommendation",
    "difficulty": "medium",
    "query": f"有没有跟《{seed_title}》相似的动画推荐？可以基于标签、导演或制作公司相似的角度来推荐几部。",
    "gold_entities": [seed_title],
    "gold_answer_keywords": sib_titles[:8],
    "gold_evidence": {
        "seed_anime_id": seed_id,
        "seed_title": seed_title,
        "anchor_tag": tag_name,
        "candidate_similar_titles": sib_titles,
    },
    "expects_rag": True,
    "scoring": {
        "min_recommendations": 2,
        "match_from_candidates_gte": 2,
        "rule": "answer must recommend >=2 anime titles that appear in candidate_similar_titles",
    },
})
print(f"new R seed=《{seed_title}》 anchor={tag_name} candidates={len(sib_titles)}")

# --- New B: obscure anime for OST-composer boundary -----------------------
refusal_kw = ["未提供", "未记录", "没有相关信息", "无法确认", "不清楚",
              "查不到", "无相关", "无法从", "不在", "not available",
              "no data", "无该信息"]

row = conn.execute(
    """
    SELECT anime_id, anime_title FROM Anime
    WHERE score_count BETWEEN 100 AND 500
      AND SUBSTR(release_date,1,4) BETWEEN '2010' AND '2020'
      AND LENGTH(anime_title) BETWEEN 4 AND 12
      AND anime_title NOT LIKE '%哆啦A梦%'
      AND anime_title NOT LIKE '%柯南%'
      AND anime_title NOT LIKE '%美食%'
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
print(f"new B001 target=《{row['anime_title']}》 (obscure)")

SRC.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"total: {len(cases)} cases")
