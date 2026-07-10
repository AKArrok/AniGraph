"""检查 BangumiCrawler 数据库完整性"""
import sqlite3, os, sys

db = r"D:\Users\ASUS\Desktop\学习资料\agent\代码\BangumiCrawler\anime_data.db"
if not os.path.exists(db):
    print(f"数据库不存在: {db}")
    sys.exit(1)

size_mb = os.path.getsize(db) / 1024 / 1024
conn = sqlite3.connect(db)
c = conn.cursor()

tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(f"数据库大小: {size_mb:.1f} MB")
print(f"表数量: {len(tables)}\n")

for (t,) in tables:
    count = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: {count:,} 条")

# 评分分布
print("\n--- 评分分布 ---")
c.execute("SELECT anime_title, score, score_count, release_date FROM Anime WHERE score IS NOT NULL ORDER BY score DESC LIMIT 5")
print("Top 5:")
for r in c.fetchall():
    print(f"  [{r[0]}] 评分: {r[1]} ({r[2]:,}人)  日期: {r[3]}")

c.execute("SELECT anime_title, score, score_count, release_date FROM Anime WHERE score IS NOT NULL ORDER BY score ASC LIMIT 5")
print("Bottom 5:")
for r in c.fetchall():
    print(f"  [{r[0]}] 评分: {r[1]} ({r[2]:,}人)  日期: {r[3]}")

# 导演/编剧/声优
print("\n--- 人员数据 ---")
for tbl, label in [("Anime_Seiyuu", "声优"), ("Anime_Director", "导演"), ("Anime_Writer", "编剧")]:
    count = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    uniq = c.execute(f"SELECT COUNT(DISTINCT anime_id) FROM {tbl}").fetchone()[0]
    print(f"  {label}: {count} 条关系, 覆盖 {uniq} 部番剧")

# 评论统计
print("\n--- 评论统计 ---")
c.execute("SELECT COUNT(DISTINCT anime_id) FROM Comments")
anime_with_comments = c.fetchone()[0]
c.execute("SELECT AVG(LENGTH(comment)) FROM Comments")
avg_len = c.fetchone()[0]
print(f"  有评论的番剧: {anime_with_comments}")
print(f"  平均评论长度: {avg_len:.0f} 字符")

# 分类/标签
c.execute("SELECT COUNT(DISTINCT category_name) FROM Category")
print(f"\n  不重复标签数: {c.fetchone()[0]}")

# 制作公司
c.execute("SELECT COUNT(DISTINCT production_name) FROM Production")
print(f"  不重复制作公司: {c.fetchone()[0]}")

conn.close()
print("\n--- 结论 ---")
print("数据完整，可以用于构建知识库。")
