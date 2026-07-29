import sqlite3
c = sqlite3.connect("data/anime_data.db")
print("tables:")
for x in c.execute("select name from sqlite_master where type='table'").fetchall():
    print(" ", x[0])
for tbl in [row[0] for row in c.execute("select name from sqlite_master where type='table'").fetchall()]:
    print()
    print(f"[{tbl}] cols:")
    for x in c.execute(f"pragma table_info({tbl})").fetchall():
        print(" ", x)
    print(f"[{tbl}] rows:", c.execute(f"select count(*) from {tbl}").fetchone())
