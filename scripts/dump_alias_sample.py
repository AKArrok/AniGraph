import sqlite3
conn = sqlite3.connect("data/anime_data.db")

# 汇总
total = conn.execute("SELECT COUNT(DISTINCT anime_id) FROM Alias").fetchone()[0]
aliases = conn.execute("SELECT COUNT(*) FROM Alias WHERE key != '__none__'").fetchone()[0]
no_alias = conn.execute("SELECT COUNT(*) FROM Alias WHERE key='__none__'").fetchone()[0]
print(f"已处理 anime={total}, 有效别名条数={aliases}, 无别名标记={no_alias}")

# 取前 20 部有别名的
print("\n前 20 部有别名的（anime_title | key | alias）:")
for row in conn.execute("""
    SELECT a.anime_id, a.anime_title, al.key, al.alias
    FROM Anime a JOIN Alias al ON a.anime_id = al.anime_id
    WHERE al.key != '__none__'
    ORDER BY a.anime_id
    LIMIT 30
""").fetchall():
    print(f"  [{row[0]}] {row[1][:20]:<20}  {row[2]:<6}  {row[3]}")

# 每部平均别名数
row = conn.execute("""
    SELECT AVG(cnt) FROM (
        SELECT COUNT(*) as cnt FROM Alias WHERE key != '__none__' GROUP BY anime_id
    )
""").fetchone()
print(f"\n有别名的番平均别名数: {row[0]:.1f}")
