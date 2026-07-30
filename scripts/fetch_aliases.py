"""抓 Bangumi v0 API 的 subject infobox alias 字段，写入 anime_data.db 的 Alias 表。

用法：
    python scripts/fetch_aliases.py            # 增量抓取 5000 条
    python scripts/fetch_aliases.py --limit 50 # 只抓前 50 条（试跑）
    python scripts/fetch_aliases.py --restart  # 忽略已抓过的，全量重跑

API 文档: https://bangumi.github.io/api/#/Subjects
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "anime_data.db")
API_BASE = "https://api.bgm.tv/v0/subjects"
PROXY = os.environ.get("HTTPS_PROXY") or "http://127.0.0.1:7897"
UA = "AniRAG-alias/1.0 (https://github.com/anonymous/anirag)"

ALIAS_KEYS = {"别名", "英文名", "日文名", "第二中文名", "其他", "其他名"}

_opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
) if PROXY else urllib.request.build_opener()


def _ensure_alias_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Alias (
            anime_id INTEGER NOT NULL,
            alias    TEXT    NOT NULL,
            key      TEXT    NOT NULL,
            PRIMARY KEY (anime_id, alias)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_alias_alias ON Alias(alias)")
    conn.commit()


def _ids_to_fetch(conn: sqlite3.Connection, restart: bool, limit: int | None) -> list[int]:
    cur = conn.execute("SELECT anime_id FROM Anime ORDER BY anime_id")
    all_ids = [r[0] for r in cur.fetchall()]
    if not restart:
        done = {r[0] for r in conn.execute(
            "SELECT DISTINCT anime_id FROM Alias").fetchall()}
        missing = [a for a in all_ids if a not in done]
    else:
        missing = all_ids
    if limit is not None:
        missing = missing[:limit]
    return missing


def _extract_aliases_from_infobox(infobox: list[dict]) -> list[tuple[str, str]]:
    """Return list of (key, alias)."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in infobox or []:
        key = str(row.get("key", "")).strip()
        if key not in ALIAS_KEYS:
            continue
        val = row.get("value")
        candidates: list[str] = []
        if isinstance(val, list):
            for v in val:
                if isinstance(v, dict):
                    s = str(v.get("v", "")).strip()
                else:
                    s = str(v).strip()
                if s:
                    candidates.append(s)
        elif isinstance(val, str):
            candidates.append(val.strip())
        else:
            continue

        for s in candidates:
            if not s or len(s) > 200:  # 极端长值丢弃
                continue
            sig = (key, s)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(sig)
    return out


def _fetch_one(anime_id: int, timeout: int = 20) -> tuple[int, list[tuple[str, str]] | None, str | None]:
    """返回 (anime_id, [(key,alias)...] 或 None 表示失败, error)"""
    url = f"{API_BASE}/{anime_id}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with _opener.open(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return anime_id, [], None  # 404 视为无别名（保留空数组以标记已尝试）
        return anime_id, None, f"http{e.code}"
    except Exception as e:
        return anime_id, None, str(e)[:80]

    aliases = _extract_aliases_from_infobox(data.get("infobox") or [])
    return anime_id, aliases, None


def _write(conn: sqlite3.Connection, anime_id: int,
            aliases: list[tuple[str, str]]) -> None:
    if not aliases:
        # 写一条哨兵：key='__no_alias__' alias='__no_alias__' 让下次跳过；
        # 但为避免污染查询，用一个不可能匹配的字符串
        conn.execute(
            "INSERT OR IGNORE INTO Alias(anime_id, alias, key) VALUES (?,?,?)",
            (anime_id, f"__none__{anime_id}", "__none__"),
        )
        return
    conn.executemany(
        "INSERT OR IGNORE INTO Alias(anime_id, alias, key) VALUES (?,?,?)",
        [(anime_id, alias, key) for key, alias in aliases],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--restart", action="store_true",
                     help="忽略已抓记录，全量重跑（不删旧数据，去重靠 PRIMARY KEY）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sleep", type=float, default=0.05,
                     help="每个成功请求后主线程 sleep（做限流）")
    args = ap.parse_args()

    if not os.path.exists(DB_PATH):
        logger.error("SQLite 不存在: %s", DB_PATH)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    _ensure_alias_table(conn)
    ids = _ids_to_fetch(conn, restart=args.restart, limit=args.limit)
    logger.info("待抓取 anime 数: %d (workers=%d, proxy=%s)", len(ids), args.workers, PROXY)

    if not ids:
        logger.info("没有需要抓取的 anime，退出")
        return

    n_ok = n_fail = n_alias = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch_one, aid): aid for aid in ids}
        for i, fut in enumerate(as_completed(futures), 1):
            aid = futures[fut]
            _, aliases, err = fut.result()
            if err is not None:
                n_fail += 1
                if n_fail <= 20:
                    logger.warning("[%s] fail: %s", aid, err)
                continue
            n_ok += 1
            n_alias += len(aliases)
            _write(conn, aid, aliases)
            if i % 100 == 0:
                conn.commit()
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                logger.info("进度 %d/%d  ok=%d fail=%d aliases=%d  %.1f req/s",
                             i, len(ids), n_ok, n_fail, n_alias, rate)
            time.sleep(args.sleep)

    conn.commit()
    conn.close()
    logger.info("完成 ok=%d fail=%d aliases=%d 耗时 %.1fs",
                 n_ok, n_fail, n_alias, time.time() - t0)


if __name__ == "__main__":
    main()
