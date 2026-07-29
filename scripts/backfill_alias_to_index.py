"""backfill_alias_to_index.py — 把 SQLite Alias 表回灌进 metadata_index.json 的 alias 字段。

不重跑 embedding、不碰向量库；仅按 anime_id 对齐，原子写回 JSON。
下次全量重建时 data/build_kb.py 已能直接带别名，此脚本用于对存量索引补齐。

用法:
    python scripts/backfill_alias_to_index.py            # 执行回填
    python scripts/backfill_alias_to_index.py --dry-run  # 只报告不写回
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config


def _load_alias_by_id(db_path: str) -> dict[str, list[str]]:
    """anime_id(str) -> sorted 去重别名列表（跳过 '__none__' 哨兵）"""
    conn = sqlite3.connect(db_path)
    out: dict[str, set[str]] = {}
    for aid, alias in conn.execute(
        "SELECT anime_id, alias FROM Alias WHERE key != '__none__' AND alias IS NOT NULL"
    ):
        if alias and alias.strip():
            out.setdefault(str(aid), set()).add(alias.strip())
    conn.close()
    return {k: sorted(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    idx_path = config.METADATA_INDEX_PATH
    db_path = config.ALIAS_DB_PATH
    if not os.path.exists(idx_path):
        print(f"[err] 索引不存在: {idx_path}")
        return 1
    if not os.path.exists(db_path):
        print(f"[err] SQLite 库不存在: {db_path}")
        return 1

    with open(idx_path, encoding="utf-8") as f:
        data = json.load(f)

    alias_by_id = _load_alias_by_id(db_path)
    print(f"索引条目: {len(data)}  |  SQLite 有别名的番剧: {len(alias_by_id)}")

    updated = filled = 0
    for item in data:
        sid = str(item.get("id", ""))
        new_aliases = alias_by_id.get(sid, [])
        if not new_aliases:
            continue
        old = item.get("alias", []) or []
        # 合并去重（保留索引里可能已有的），排除与标题完全相同的冗余
        title_forms = {
            (item.get("name_cn") or "").strip(),
            (item.get("name") or "").strip(),
        }
        merged = sorted({*old, *new_aliases} - {""} - title_forms)
        if merged != (old or []):
            item["alias"] = merged
            updated += 1
            filled += len(merged)

    total_alias = sum(len(i.get("alias", []) or []) for i in data)
    print(f"更新条目: {updated}  |  索引内别名总数(回填后): {total_alias}")

    if args.dry_run:
        print("[dry-run] 未写回")
        return 0

    d = os.path.dirname(idx_path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, idx_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"[ok] 已原子写回 {idx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
