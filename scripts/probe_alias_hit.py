"""看几个 seed alias 能不能通过 Alias 表反查到正确的 anime。"""
import sqlite3
conn = sqlite3.connect("data/anime_data.db")

PROBES = [
    "石头门", "EVA", "シュタゲ", "叛逆的鲁路修",
    "素晴", "为美好世界献上祝福", "巨人", "钢炼",
    "命运石之门", "Fullmetal Alchemist", "克拉那多",
    "推子", "我推", "花名", "未闻花名",
    "咒回", "咒术", "小圆", "圆神",
]

for q in PROBES:
    # 别名精确
    rows = conn.execute("""
        SELECT DISTINCT a.anime_id, a.anime_title
        FROM Anime a JOIN Alias al ON a.anime_id = al.anime_id
        WHERE al.alias = ? COLLATE NOCASE
    """, (q,)).fetchall()
    # 别名 LIKE 匹配（模糊）
    like_rows = conn.execute("""
        SELECT DISTINCT a.anime_id, a.anime_title
        FROM Anime a JOIN Alias al ON a.anime_id = al.anime_id
        WHERE al.alias LIKE ? COLLATE NOCASE
        LIMIT 3
    """, (f"%{q}%",)).fetchall()
    print(f"[{q}] exact={rows[:3]}  like={like_rows[:3]}")
