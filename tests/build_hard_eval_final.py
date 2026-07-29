"""Curate a 20-case KB-heavy eval set from data/anime_data.db.

Composition:
  10 metadata_cross (studio+year, director filmography) - gold: exact titles
   8 longtail_fact  (score_count 20-200)                - gold: score, year
   2 kb_boundary    (field not in DB)                   - gold: refusal phrases

Deliberately excludes comment_anchor: gold-keyword extraction from raw
comments is too noisy for automatic scoring (would need semantic Judge).

Output: tests/eval_hard_final.json
"""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

random.seed(20260728)

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "anime_data.db"
OUT = ROOT / "tests" / "eval_hard_final.json"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row


def longtail(n: int) -> list[dict]:
    """Low-vote anime facts. Exclude titles that are so recent LLM cannot know,
    and exclude score_count < 20 (too obscure, DB may be incomplete)."""
    rows = conn.execute(
        """
        SELECT anime_id, anime_title, score, score_count, release_date
        FROM Anime
        WHERE score IS NOT NULL
          AND score_count BETWEEN 30 AND 200
          AND release_date IS NOT NULL
          AND SUBSTR(release_date, 1, 4) BETWEEN '1990' AND '2023'
          AND score >= 5.5
        ORDER BY RANDOM() LIMIT 40
        """
    ).fetchall()
    out: list[dict] = []
    seen_years: dict[str, int] = {}
    for r in rows:
        year = r["release_date"][:4]
        # Diversify years so we don't pick 5 from 2020.
        if seen_years.get(year, 0) >= 2:
            continue
        seen_years[year] = seen_years.get(year, 0) + 1
        prods = conn.execute(
            """
            SELECT p.production_name FROM Production p
            JOIN Anime_Production ap ON ap.production_id = p.production_id
            WHERE ap.anime_id = ?
            """,
            (r["anime_id"],),
        ).fetchall()
        out.append({
            "id": f"L{r['anime_id']:06d}",
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
                "production": [p["production_name"] for p in prods],
            },
            "expects_rag": True,
            "scoring": {
                "score_exact": f"{r['score']:.1f}",
                "year_exact": year,
                "rule": "answer must contain BOTH the exact score (one decimal) AND the year",
            },
        })
        if len(out) >= n:
            break
    return out


def metadata_cross(n_studio_year: int = 6, n_director: int = 4) -> list[dict]:
    out: list[dict] = []
    studio_rows = conn.execute(
        """
        SELECT p.production_name AS studio, SUBSTR(a.release_date,1,4) AS year,
               COUNT(*) AS n, GROUP_CONCAT(a.anime_title, '||') AS titles
        FROM Anime a
        JOIN Anime_Production ap ON ap.anime_id = a.anime_id
        JOIN Production p ON p.production_id = ap.production_id
        WHERE a.release_date IS NOT NULL
          AND p.production_name IN ('京都アニメーション', 'MADHOUSE', 'Production I.G',
                                    'A-1 Pictures', 'ボンズ', 'サンライズ',
                                    'ufotable', 'シャフト', 'WIT STUDIO', 'MAPPA')
        GROUP BY p.production_name, year
        HAVING n BETWEEN 2 AND 4
          AND CAST(year AS INTEGER) BETWEEN 2000 AND 2022
        ORDER BY RANDOM() LIMIT 20
        """
    ).fetchall()
    for r in studio_rows:
        titles = [t.strip() for t in (r["titles"] or "").split("||") if t.strip()]
        if not titles:
            continue
        out.append({
            "id": f"M{len(out) + 1:03d}",
            "type": "metadata_cross",
            "difficulty": "medium",
            "query": f"{r['studio']} 在 {r['year']} 年制作了哪些动画？",
            "gold_entities": titles,
            "gold_answer_keywords": titles,
            "gold_evidence": {"studio": r["studio"], "year": r["year"], "titles": titles},
            "expects_rag": True,
            "scoring": {
                "must_include_titles": titles,
                "rule": "answer must mention ALL gold titles (partial substring match ok)",
            },
        })
        if len(out) >= n_studio_year:
            break

    dir_rows = conn.execute(
        """
        SELECT d.director_id, d.director_name, COUNT(*) AS n,
               GROUP_CONCAT(a.anime_title, '||') AS titles
        FROM Director d
        JOIN Anime_Director ad ON ad.director_id = d.director_id
        JOIN Anime a ON a.anime_id = ad.anime_id
        GROUP BY d.director_id
        HAVING n BETWEEN 3 AND 5
        ORDER BY RANDOM() LIMIT 15
        """
    ).fetchall()
    picked = 0
    for r in dir_rows:
        titles = [t.strip() for t in (r["titles"] or "").split("||") if t.strip()]
        if not titles:
            continue
        out.append({
            "id": f"M{len(out) + 1:03d}",
            "type": "metadata_cross",
            "difficulty": "hard",
            "query": f"导演 {r['director_name']} 执导或参与执导过哪些动画？",
            "gold_entities": titles,
            "gold_answer_keywords": titles,
            "gold_evidence": {"director": r["director_name"], "titles": titles},
            "expects_rag": True,
            "scoring": {
                "must_include_titles": titles,
                "rule": "answer must mention ALL gold titles (partial substring match ok)",
            },
        })
        picked += 1
        if picked >= n_director:
            break
    return out


def boundary(n: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT anime_id, anime_title FROM Anime
        WHERE score IS NOT NULL AND score_count > 5000
        ORDER BY RANDOM() LIMIT 6
        """
    ).fetchall()
    templates = [
        ("《{title}》原声带的作曲家是谁？请只根据知识库回答。",
         "boundary_ost", "OST composer"),
        ("《{title}》的音响监督是谁？请只根据知识库回答。",
         "boundary_sound", "sound director"),
        ("《{title}》官方公布的英文标语（tagline）是什么？请只根据知识库回答。",
         "boundary_tagline", "official tagline"),
    ]
    refusal_keywords = ["未提供", "未记录", "没有相关信息", "无法确认", "不清楚",
                        "查不到", "无相关", "无法从", "不在", "not available",
                        "no data", "无该信息"]
    out: list[dict] = []
    for r, (tpl, subtype, field) in zip(rows, templates):
        out.append({
            "id": f"B{len(out) + 1:03d}",
            "type": "kb_boundary",
            "difficulty": "hard",
            "query": tpl.format(title=r["anime_title"]),
            "gold_entities": [r["anime_title"]],
            "gold_answer_keywords": refusal_keywords,
            "gold_evidence": {
                "note": f"DB does not store field '{field}'.",
                "correct_behavior": "acknowledge missing info; do not fabricate a name",
            },
            "expects_rag": False,
            "expects_refusal": True,
            "scoring": {
                "must_contain_refusal": True,
                "must_not_contain_specific_name": True,
                "rule": "answer must admit missing info; MUST NOT invent a composer/director name",
            },
        })
        if len(out) >= n:
            break
    return out


def main() -> None:
    cases = metadata_cross() + longtail(8) + boundary(2)
    random.shuffle(cases)
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT} with {len(cases)} cases")
    dist: dict[str, int] = {}
    for c in cases:
        dist[c["type"]] = dist.get(c["type"], 0) + 1
    for k, v in dist.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
