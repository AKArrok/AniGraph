"""Extend eval_hard_final.json with Bangumi-specific queries:

  T (Tags):  "What Bangumi tags did users apply to <anime>?"
  S (Score): "What is <anime>'s Bangumi score (one decimal)?"

Both use community-only data that LLM training cannot know.
"""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
import re

random.seed(20260728)

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "anime_data.db"
SRC = ROOT / "tests" / "eval_hard_final.json"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

cases = json.loads(SRC.read_text(encoding="utf-8"))
existing = {c["id"] for c in cases}


def _title_char_set(title: str) -> set[str]:
    """Characters (>=2 length substrings) drawn from the title, used to filter
    out tags that are just parts of the title (like '哈尔' from
    '哈尔的移动城堡'). Excludes trivial punctuation."""
    cleaned = re.sub(r"[\s《》「」『』（）()：:；;,，.。！!？?、\-—–_]+", "", title)
    return set(cleaned)


def _tag_is_title_derived(tag: str, title_chars: set[str]) -> bool:
    """True if the tag shares >=2 chars with the title AND those chars form
    a majority of the tag."""
    if not tag:
        return True
    shared = sum(1 for ch in tag if ch in title_chars)
    return shared >= 2 and shared >= len(tag) - 1

# --- Tag lookup (T) --------------------------------------------------------
# Bangumi anime typically carry 25-30 tags, so instead of requiring a small
# tag set we take rich-tagged anime and hand-pick 5 non-generic tags as
# gold. LLM without RAG cannot know community tags like "神作", "京阿尼",
# "钉宫理惠", etc.
GENERIC_TAGS = {"日本", "日本动画", "TV", "TVA", "OVA", "动画", "剧场版",
                "漫改", "漫画改", "小说改", "轻小说改", "轻改", "游戏改",
                "原创", "续作", "TVA", "季番", "半年番", "长篇", "补番",
                "补标"}
GENERIC_YEAR = None  # markers like '2013', '2013年', '2013年10月'

rows = conn.execute(
    """
    SELECT a.anime_id, a.anime_title
    FROM Anime a
    WHERE a.score >= 7.8 AND a.score_count >= 3000
    ORDER BY RANDOM() LIMIT 60
    """
).fetchall()

tag_cases = []
YEAR_RE = re.compile(r"^\d{4}(?:年(?:\d{1,2}月?)?)?$")
ROMAJI_RE = re.compile(r"^[A-Za-z0-9._\-]+$")
for r in rows:
    tags = conn.execute(
        """
        SELECT cat.category_name
        FROM Anime_Category ac JOIN Category cat ON cat.category_id = ac.category_id
        WHERE ac.anime_id = ?
        """, (r["anime_id"],)
    ).fetchall()
    tag_names = [t["category_name"] for t in tags]
    if len(tag_names) < 5:
        continue
    title_chars = _title_char_set(r["anime_title"])
    distinctive = [
        t for t in tag_names
        if t not in GENERIC_TAGS
        and not YEAR_RE.match(t)
        and not ROMAJI_RE.match(t)  # drop romaji nicknames like "SpaceDandy"
        and not _tag_is_title_derived(t, title_chars)
        and 1 < len(t) < 8  # drop 1-char stubs and huge phrases
    ]
    if len(distinctive) < 5:
        continue
    # Pick 5 tags: prefer common/canonical (short, Chinese) over long fandom.
    picked = sorted(distinctive, key=lambda x: (len(x), x))[:5]
    tag_cases.append({
        "id": f"T{r['anime_id']:06d}",
        "type": "bangumi_tags",
        "difficulty": "medium",
        "query": f"在 Bangumi 上，《{r['anime_title']}》被观众/编辑打上了哪些标签？请列出主要标签。",
        "gold_entities": [r["anime_title"]],
        "gold_answer_keywords": picked,
        "gold_evidence": {
            "anime_id": r["anime_id"],
            "all_tags": tag_names,
            "picked_gold_tags": picked,
        },
        "expects_rag": True,
        "scoring": {
            "match_ratio_gte": 0.6,
            "rule": "answer must mention >=60% (>=3/5) of picked gold tags",
            "gold_tags": picked,
        },
    })
    if len(tag_cases) >= 6:
        break

# --- Precise-score (S) -----------------------------------------------------
# Famous anime with exact one-decimal Bangumi score. Ask the model for the
# score; only the exact decimal string counts.
score_pool = conn.execute(
    """
    SELECT anime_id, anime_title, score, score_count
    FROM Anime
    WHERE score IS NOT NULL AND score_count >= 8000 AND score >= 7.5
    ORDER BY RANDOM() LIMIT 40
    """
).fetchall()

score_cases = []
for r in score_pool:
    score_str = f"{r['score']:.1f}"
    score_cases.append({
        "id": f"S{r['anime_id']:06d}",
        "type": "bangumi_score_precise",
        "difficulty": "medium",
        "query": f"《{r['anime_title']}》在 Bangumi 上的评分是多少（精确到小数点后一位）？",
        "gold_entities": [r["anime_title"]],
        "gold_answer_keywords": [score_str],
        "gold_evidence": {
            "anime_id": r["anime_id"],
            "score": r["score"],
            "score_str": score_str,
            "score_count": r["score_count"],
        },
        "expects_rag": True,
        "scoring": {
            "score_exact": score_str,
            "rule": "answer must contain the exact one-decimal score",
        },
    })
    if len(score_cases) >= 6:
        break

added = 0
for c in tag_cases + score_cases:
    if c["id"] in existing:
        continue
    cases.append(c)
    existing.add(c["id"])
    added += 1

# --- Similar recommendation (R) --------------------------------------------
# Casual-sounding queries: "recommend anime like <X>". Gold answer is any
# other anime that shares tag or director/writer/production with the seed.
# Scoring: answer must (a) name >=2 distinct anime titles from the KB and
# (b) each named title must actually exist in the DB.

seed_pool = conn.execute(
    """
    SELECT anime_id, anime_title FROM Anime
    WHERE score >= 8.3 AND score_count >= 8000
    ORDER BY RANDOM() LIMIT 20
    """
).fetchall()

rec_cases: list[dict] = []
for seed in seed_pool:
    seed_id = seed["anime_id"]
    seed_title = seed["anime_title"]
    # Anchor by shared tag (most descriptive)
    anchor_tag = conn.execute(
        """
        SELECT cat.category_name
        FROM Anime_Category ac JOIN Category cat ON cat.category_id = ac.category_id
        WHERE ac.anime_id = ? AND cat.category_name NOT IN
          ('TV','TVA','日本','日本动画','动画','漫改','漫画改','原创','续作')
          AND LENGTH(cat.category_name) BETWEEN 2 AND 6
        ORDER BY RANDOM() LIMIT 1
        """, (seed_id,)).fetchone()
    if not anchor_tag:
        continue
    tag_name = anchor_tag["category_name"]
    # Fetch other anime carrying the same tag with good score
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
    if len(sib_titles) < 4:
        continue
    rec_cases.append({
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
    if len(rec_cases) >= 6:
        break

for c in rec_cases:
    if c["id"] in existing:
        continue
    cases.append(c)
    existing.add(c["id"])
    added += 1

SRC.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"added {added} new cases; total now {len(cases)}")
for t in ("bangumi_tags", "bangumi_score_precise", "similar_recommendation"):
    n = sum(1 for c in cases if c["type"] == t)
    print(f"  {t}: {n}")
