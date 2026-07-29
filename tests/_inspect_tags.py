"""Explore the tag distribution to design tag-lookup queries."""
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "anime_data.db"
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

# Most common tags
print("== top 30 tags ==")
rows = c.execute(
    "SELECT cat.category_name, COUNT(*) AS n "
    "FROM Anime_Category ac JOIN Category cat ON cat.category_id = ac.category_id "
    "GROUP BY cat.category_id ORDER BY n DESC LIMIT 30"
).fetchall()
for r in rows:
    print(f"  {r['category_name']:20} {r['n']}")

print("\n== 12 anime with 5-8 tags (score>=7.5, score_count>=2000) ==")
rows = c.execute(
    """
    SELECT a.anime_id, a.anime_title, a.score, a.score_count,
           (SELECT GROUP_CONCAT(cat.category_name, '|')
              FROM Anime_Category ac2
              JOIN Category cat ON cat.category_id = ac2.category_id
              WHERE ac2.anime_id = a.anime_id) AS tags,
           (SELECT COUNT(*) FROM Anime_Category ac3 WHERE ac3.anime_id = a.anime_id) AS n_tags
    FROM Anime a
    WHERE a.score >= 7.5 AND a.score_count >= 2000
    ORDER BY RANDOM() LIMIT 40
    """
).fetchall()
picked = 0
for r in rows:
    n = r['n_tags'] or 0
    if n < 5 or n > 10:
        continue
    print(f"  [{r['anime_id']}] {r['anime_title']}  score={r['score']} n_tags={n} tags={r['tags']}")
    picked += 1
    if picked >= 12:
        break

# Bangumi-flavor tags (community-specific, not generic genre)
print("\n== anime with distinctive Bangumi tags ==")
for tag in ("神作", "意识流", "催泪", "京都动画", "京阿尼", "百合", "GAR", "萌", "厨力"):
    rows = c.execute(
        """
        SELECT a.anime_id, a.anime_title
        FROM Anime a JOIN Anime_Category ac ON ac.anime_id = a.anime_id
        JOIN Category cat ON cat.category_id = ac.category_id
        WHERE cat.category_name = ? AND a.score_count >= 5000
        ORDER BY a.score_count DESC LIMIT 3
        """, (tag,)).fetchall()
    if rows:
        titles = ", ".join(r["anime_title"] for r in rows)
        print(f"  '{tag}' -> {titles}")

# High-precision score anime (well known)
print("\n== 10 famous anime with high score (for score-recall test) ==")
rows = c.execute(
    """
    SELECT anime_id, anime_title, score, score_count
    FROM Anime
    WHERE score >= 8.5 AND score_count >= 10000
    ORDER BY score_count DESC LIMIT 10
    """
).fetchall()
for r in rows:
    print(f"  [{r['anime_id']}] {r['anime_title']}  score={r['score']} n={r['score_count']}")
