import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "anime_data.db"
c = sqlite3.connect(DB)

tables = [r[0] for r in c.execute("select name from sqlite_master where type='table'").fetchall()]
print("tables:", tables)
for t in tables:
    cols = c.execute(f"pragma table_info({t})").fetchall()
    n = c.execute(f"select count(*) from {t}").fetchone()[0]
    print(f"\n== {t} (n={n}) ==")
    for col in cols:
        print(" ", col)
    if n:
        sample = c.execute(f"select * from {t} limit 2").fetchall()
        for s in sample:
            print("  sample:", s)
