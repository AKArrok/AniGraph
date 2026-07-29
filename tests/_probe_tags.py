import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "anime_data.db"
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row
GENERIC = {"日本","日本动画","TV","TVA","OVA","动画","剧场版","漫改","漫画改",
           "小说改","轻小说改","轻改","游戏改","原创","续作"}

rows = c.execute("""
    SELECT a.anime_id, a.anime_title
    FROM Anime a
    WHERE a.score >= 7.8 AND a.score_count >= 3000
    ORDER BY RANDOM() LIMIT 30
""").fetchall()

for r in rows:
    tags = [t[0] for t in c.execute("""
        SELECT cat.category_name
        FROM Anime_Category ac JOIN Category cat ON cat.category_id=ac.category_id
        WHERE ac.anime_id=?""", (r['anime_id'],)).fetchall()]
    distinctive = [t for t in tags if t not in GENERIC]
    print(f"[{r['anime_id']}] {r['anime_title']}  n_tags={len(tags)} distinctive={len(distinctive)}  tags={tags}")
