"""Generate a KB-heavy eval set from data/anime_data.db.

Four categories, all designed to punish direct LLM answers and reward RAG:

  L (Long-tail):     ask a fact about a low-vote anime (score_count small).
                     LLM training rarely covers it; DB has the answer verbatim.
  M (Metadata cross):structured cross-table questions (studio+year, director+studio).
                     Requires exact ID joins; LLM confabulates.
  C (Comment-anchor):paraphrase a distinctive user comment; ask what people say
                     about the anime. gold_keywords = tokens from the comment.
  B (Boundary):      ask for a field the DB does not store (OST composer, etc.)
                     Correct behavior is to answer "unknown/not in KB".

Output: tests/eval_hard.json (30 cases -> we curate down to 20 later).
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
from pathlib import Path

random.seed(20260728)

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "anime_data.db"
OUT = ROOT / "tests" / "eval_hard.json"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

CN_STOP = set(list("的了是和与及以之在有没个于对为从被把也都还就更最很非常一二三四五六七八九十等这那"))


def _tokens(text: str, k: int = 4) -> list[str]:
    """Very lightweight keyword extraction: pull 2-4 char content ngrams."""
    text = re.sub(r"[\s\W_]+", " ", text)
    words = [w for w in text.split() if w]
    # Take short substrings that look content-bearing.
    grams: list[str] = []
    for w in words:
        for n in (4, 3, 2):
            for i in range(len(w) - n + 1):
                g = w[i : i + n]
                if any(ch in CN_STOP for ch in g):
                    continue
                if g not in grams:
                    grams.append(g)
    return grams[:k]


def category_L() -> list[dict]:
    """Long-tail hard facts: low score_count anime."""
    rows = conn.execute(
        """
        SELECT anime_id, anime_title, score, score_count, release_date
        FROM Anime
        WHERE score IS NOT NULL AND score_count BETWEEN 20 AND 200
        ORDER BY RANDOM() LIMIT 40
        """
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        if not r["release_date"]:
            continue
        year = r["release_date"][:4]
        prods = conn.execute(
            """
            SELECT p.production_name FROM Production p
            JOIN Anime_Production ap ON ap.production_id = p.production_id
            WHERE ap.anime_id = ?
            """,
            (r["anime_id"],),
        ).fetchall()
        prod_names = [p["production_name"] for p in prods]
        out.append({
            "id": f"L{r['anime_id']:05d}",
            "type": "longtail_fact",
            "difficulty": "hard",
            "query": f"《{r['anime_title']}》的评分是多少？上映年份是哪一年？",
            "gold_entities": [r["anime_title"]],
            "gold_answer_keywords": [
                str(r["score"]) if r["score"] else "",
                year,
            ],
            "gold_evidence": {
                "anime_id": r["anime_id"],
                "score": r["score"],
                "score_count": r["score_count"],
                "release_date": r["release_date"],
                "production": prod_names,
            },
            "expects_rag": True,
        })
        if len(out) >= 10:
            break
    return out


def category_M() -> list[dict]:
    """Metadata cross-table questions."""
    out: list[dict] = []
    # 1. Studio + year: count/list of anime from studio X in year Y
    studio_year = conn.execute(
        """
        SELECT p.production_name AS studio, SUBSTR(a.release_date,1,4) AS year,
               COUNT(*) AS n, GROUP_CONCAT(a.anime_title, ' | ') AS titles
        FROM Anime a
        JOIN Anime_Production ap ON ap.anime_id = a.anime_id
        JOIN Production p ON p.production_id = ap.production_id
        WHERE a.release_date IS NOT NULL
          AND p.production_name IN ('京都アニメーション', 'MADHOUSE', 'Production I.G',
                                    'A-1 Pictures', 'ボンズ', 'サンライズ')
        GROUP BY p.production_name, year
        HAVING n BETWEEN 2 AND 5
        ORDER BY RANDOM() LIMIT 6
        """
    ).fetchall()
    for r in studio_year:
        titles = [t.strip() for t in (r["titles"] or "").split("|") if t.strip()]
        out.append({
            "id": f"M{len(out) + 1:03d}",
            "type": "metadata_cross",
            "difficulty": "medium",
            "query": f"{r['studio']} 在 {r['year']} 年制作了哪些动画？",
            "gold_entities": titles,
            "gold_answer_keywords": titles[:3],
            "gold_evidence": {"studio": r["studio"], "year": r["year"], "titles": titles},
            "expects_rag": True,
        })

    # 2. Director's full filmography (short list)
    dir_rows = conn.execute(
        """
        SELECT d.director_id, d.director_name, COUNT(*) AS n,
               GROUP_CONCAT(a.anime_title, ' | ') AS titles
        FROM Director d
        JOIN Anime_Director ad ON ad.director_id = d.director_id
        JOIN Anime a ON a.anime_id = ad.anime_id
        GROUP BY d.director_id
        HAVING n BETWEEN 3 AND 6
        ORDER BY RANDOM() LIMIT 4
        """
    ).fetchall()
    for r in dir_rows:
        titles = [t.strip() for t in (r["titles"] or "").split("|") if t.strip()]
        out.append({
            "id": f"M{len(out) + 1:03d}",
            "type": "metadata_cross",
            "difficulty": "hard",
            "query": f"导演 {r['director_name']} 参与执导过哪些动画？",
            "gold_entities": titles,
            "gold_answer_keywords": titles[:3],
            "gold_evidence": {"director": r["director_name"], "titles": titles},
            "expects_rag": True,
        })
    return out


def category_C() -> list[dict]:
    """Comment-anchored: paraphrase distinctive user comment."""
    rows = conn.execute(
        """
        SELECT c.anime_id, c.comment, a.anime_title, a.score, a.score_count
        FROM Comments c
        JOIN Anime a ON a.anime_id = c.anime_id
        WHERE LENGTH(c.comment) BETWEEN 40 AND 200
          AND a.score_count > 500
        ORDER BY RANDOM() LIMIT 30
        """
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        toks = _tokens(r["comment"], k=4)
        if len(toks) < 3:
            continue
        out.append({
            "id": f"C{len(out) + 1:03d}",
            "type": "comment_anchor",
            "difficulty": "hard",
            "query": f"根据用户评论，《{r['anime_title']}》被观众提到最多的一个具体评价点是什么？（引用一句原话作为佐证）",
            "gold_entities": [r["anime_title"]],
            "gold_answer_keywords": toks,
            "gold_evidence": {"comment": r["comment"], "anime_id": r["anime_id"]},
            "expects_rag": True,
        })
        if len(out) >= 6:
            break
    return out


def category_B() -> list[dict]:
    """Boundary: ask for a field DB does not store; expects 'don't know'."""
    # OST composer, official synopsis, English tagline — none exist in DB.
    rows = conn.execute(
        """
        SELECT anime_id, anime_title FROM Anime
        WHERE score IS NOT NULL AND score_count > 3000
        ORDER BY RANDOM() LIMIT 4
        """
    ).fetchall()
    out: list[dict] = []
    templates = [
        ("《{title}》的原声带作曲家是谁？", "boundary_ost"),
        ("《{title}》官方英文标语是什么？", "boundary_tagline"),
        ("《{title}》第一话的播出准确时间（几点几分）是什么？", "boundary_airtime"),
        ("《{title}》的音响监督是谁？", "boundary_sound"),
    ]
    for r, (tpl, subtype) in zip(rows, templates):
        out.append({
            "id": f"B{len(out) + 1:03d}",
            "type": "kb_boundary",
            "difficulty": "hard",
            "query": tpl.format(title=r["anime_title"]),
            "gold_entities": [r["anime_title"]],
            "gold_answer_keywords": ["未记录", "不知道", "无相关信息", "not in database", subtype],
            "gold_evidence": {"note": "DB does not store this field; correct answer is 'unknown'"},
            "expects_rag": False,
            "expects_refusal": True,
        })
    return out


def main() -> None:
    cases = category_L() + category_M() + category_C() + category_B()
    random.shuffle(cases)
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT} with {len(cases)} cases")
    counts: dict[str, int] = {}
    for c in cases:
        counts[c["type"]] = counts.get(c["type"], 0) + 1
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
