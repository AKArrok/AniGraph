"""Pick 2 obscure anime titles for kb_boundary swap-in."""
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "anime_data.db"
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

rows = c.execute(
    """
    SELECT anime_id, anime_title, score, score_count, release_date
    FROM Anime
    WHERE score IS NOT NULL AND score_count BETWEEN 300 AND 1500
      AND SUBSTR(release_date,1,4) BETWEEN '2005' AND '2020'
    ORDER BY RANDOM() LIMIT 20
    """
).fetchall()
for r in rows:
    print(r["anime_id"], r["anime_title"], r["score"], r["score_count"], r["release_date"])
