"""对照现有 seed HARDCODED_ALIASES 和 SQLite Alias 表，判断哪些 seed 已被覆盖。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.alias import HARDCODED_ALIASES, _load_alias_db

db_idx = _load_alias_db()
print(f"SQLite alias 索引大小: {len(db_idx)}\n")

covered = []
missing = []
conflict = []
for k, v in HARDCODED_ALIASES.items():
    kl = k.lower()
    if kl in db_idx:
        if db_idx[kl].strip() == v.strip():
            covered.append((k, v))
        else:
            conflict.append((k, v, db_idx[kl]))
    else:
        missing.append((k, v))

print(f"[COVERED]  seed 已被 SQLite 相同映射覆盖: {len(covered)}")
for k, v in covered:
    print(f"  {k:<20} -> {v}")
print(f"\n[MISSING]  SQLite 里没有的（需要在 seed 保留）: {len(missing)}")
for k, v in missing:
    print(f"  {k:<20} -> {v}")
print(f"\n[CONFLICT] SQLite 与 seed 映射到不同 anime（需要人工判定）: {len(conflict)}")
for k, seed_v, db_v in conflict:
    print(f"  {k:<20} seed={seed_v}  |  db={db_v}")
