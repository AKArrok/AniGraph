"""知识库构建脚本 — 从 BangumiCrawler SQLite 构建 Pinecone + Whoosh + MetadataIndex

用法:
  python data/build_kb.py              # 全新构建
  python data/build_kb.py --resume      # 断点续跑
  python data/build_kb.py --metadata-only  # 仅生成 metadata_index.json
  python data/build_kb.py --whoosh-only    # 仅构建 Whoosh 索引
"""
import os
import sys
import json
import time
import hashlib
import sqlite3
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from llms import embeddings
from data.chunking import CHUNK_SCHEMA_VERSION, make_anime_chunks

# BangumiCrawler 数据库路径（本地 data 目录）
BANGUMI_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anime_data.db")

CHECKPOINT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_kb.json")
METADATA_PATH = config.METADATA_INDEX_PATH
WHOOSH_DIR = config.WHOOSH_INDEX_DIR

BATCH_SIZE = 10          # 每批处理的番剧数
SAVE_INTERVAL = 20       # 每多少条保存 checkpoint
MAX_COMMENTS_PER_ANIME = 10  # 每部番剧最多取多少评论


def _stale_child_ids(anime_id: int, chunks: list) -> list[str]:
    """Return predictable child IDs that are no longer emitted for an anime."""
    current_ids = {chunk.id for chunk in chunks}
    candidates = {f"anime_{anime_id}_staff_0", f"anime_{anime_id}_cast_0"}
    candidates.update(
        f"anime_{anime_id}_reviews_{index}"
        for index in range(MAX_COMMENTS_PER_ANIME)
    )
    return sorted(candidates - current_ids)


# ══════════════════════════════════════════════════════════════════════
# 1. 数据读取
# ══════════════════════════════════════════════════════════════════════

def _get_db_conn():
    db_path = os.path.abspath(BANGUMI_DB)
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return None
    return sqlite3.connect(db_path)


def _fetch_anime_ids(conn) -> list[int]:
    """获取所有番剧 ID"""
    rows = conn.execute("SELECT anime_id FROM Anime ORDER BY anime_id").fetchall()
    return [r[0] for r in rows]


def _fetch_anime_detail(conn, anime_id: int) -> dict | None:
    """获取单部番剧完整信息"""
    # 基本信息
    row = conn.execute(
        "SELECT anime_id, anime_title, score, score_count, release_date FROM Anime WHERE anime_id = ?",
        (anime_id,)
    ).fetchone()
    if not row:
        return None

    info = {
        "id": row[0],
        "title": row[1],
        "score": row[2],
        "score_count": row[3],
        "date": row[4] or "",
    }

    # 分类/标签
    tags = conn.execute("""
        SELECT c.category_name FROM Category c
        JOIN Anime_Category ac ON c.category_id = ac.category_id
        WHERE ac.anime_id = ?
    """, (anime_id,)).fetchall()
    info["tags"] = [t[0] for t in tags]

    # 制作公司
    studios = conn.execute("""
        SELECT p.production_name FROM Production p
        JOIN Anime_Production ap ON p.production_id = ap.production_id
        WHERE ap.anime_id = ?
    """, (anime_id,)).fetchall()
    info["studios"] = [s[0] for s in studios]

    # 导演
    directors = conn.execute("""
        SELECT d.director_name FROM Director d
        JOIN Anime_Director ad ON d.director_id = ad.director_id
        WHERE ad.anime_id = ?
    """, (anime_id,)).fetchall()
    info["directors"] = [d[0] for d in directors]

    # 编剧
    writers = conn.execute("""
        SELECT w.writer_name FROM Writer w
        JOIN Anime_Writer aw ON w.writer_id = aw.writer_id
        WHERE aw.anime_id = ?
    """, (anime_id,)).fetchall()
    info["writers"] = [w[0] for w in writers]

    # 声优
    seiyuu = conn.execute("""
        SELECT s.seiyuu_name FROM Seiyuu s
        JOIN Anime_Seiyuu as2 ON s.seiyuu_id = as2.seiyuu_id
        WHERE as2.anime_id = ?
    """, (anime_id,)).fetchall()
    info["seiyuu"] = [s[0] for s in seiyuu]

    # 评论
    comments = conn.execute(
        "SELECT comment FROM Comments WHERE anime_id = ? ORDER BY comment_id LIMIT ?",
        (anime_id, MAX_COMMENTS_PER_ANIME)
    ).fetchall()
    info["comments"] = [c[0] for c in comments if c[0]]

    return info


def _make_metadata_entry(info: dict) -> dict:
    """生成 metadata_index.json 条目"""
    return {
        "id": str(info["id"]),
        "name": info.get("title", ""),
        "name_cn": info.get("title", ""),  # Bangumi 爬取的是中文标题
        "score": info.get("score"),
        "rank": 0,  # 由外部计算
        "date": info.get("date", ""),
        "tags": info.get("tags", []),
        "studio": info.get("studios", [None])[0] if info.get("studios") else "",
        "director": info.get("directors", [None])[0] if info.get("directors") else "",
        "writer": info.get("writers", [None])[0] if info.get("writers") else "",
        "seiyuu": info.get("seiyuu", []),
        "alias": [],
    }


# ══════════════════════════════════════════════════════════════════════
# 2. 构建 Pinecone 知识库
# ══════════════════════════════════════════════════════════════════════

def _init_pinecone():
    """初始化 Pinecone 索引"""
    from pinecone import Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    if config.PINECONE_INDEX not in [idx["name"] for idx in pc.list_indexes()]:
        print(f"Pinecone index '{config.PINECONE_INDEX}' 不存在，请先创建")
        return None

    return pc.Index(config.PINECONE_INDEX)


def build_pinecone(resume: bool = True):
    """构建 Pinecone 知识库"""
    conn = _get_db_conn()
    if not conn:
        return

    pc_index = _init_pinecone()
    if not pc_index:
        conn.close()
        return

    all_ids = _fetch_anime_ids(conn)
    total = len(all_ids)
    print(f"总番剧数: {total}")

    # 获取 Pinecone 索引维度
    idx_stats = pc_index.describe_index_stats()
    pinecone_dim = idx_stats.dimension
    print(f"Pinecone 索引维度: {pinecone_dim}")
    if pinecone_dim != embeddings.dimension:
        raise ValueError(
            f"Pinecone 索引维度 {pinecone_dim} 与 Embedding 维度 {embeddings.dimension} 不匹配。"
            " 请创建新索引或调整 ARK_EMBEDDING_DIMENSIONS。"
        )
    dim_errors = 0  # 维度不匹配计数

    # 加载 checkpoint
    processed = set()
    embed_count = 0
    metadata_entries = []

    if resume and os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            ck = json.load(f)
        if (
            ck.get("chunk_schema_version") == CHUNK_SCHEMA_VERSION
            and ck.get("embedding_model_identity") == embeddings.model_identity
        ):
            processed = set(str(x) for x in ck.get("processed_ids", []))
            embed_count = ck.get("embed_count", 0)
            metadata_entries = ck.get("metadata_entries", [])
            print(f"断点续跑: 已处理 {len(processed)} 条, 从第 {embed_count} 条继续")
        elif ck.get("chunk_schema_version") == CHUNK_SCHEMA_VERSION:
            print("Embedding 模型/维度已变更，全量重建")
        else:
            print("检测到旧版 checkpoint，将按新分块 schema 全量重建")

    anime_batch: list[tuple[dict, list]] = []
    errors = 0
    last_saved_count = embed_count

    def flush_batch() -> None:
        nonlocal anime_batch, embed_count, errors, last_saved_count
        if not anime_batch:
            return
        chunks = [chunk for _, anime_chunks in anime_batch for chunk in anime_chunks]
        try:
            vectors = embeddings.embed_documents(
                [chunk.text for chunk in chunks], target_dim=pinecone_dim
            )
            titles = {info["id"]: info.get("title", "") for info, _ in anime_batch}
            pc_index.upsert(vectors=[{
                "id": chunk.id,
                "values": vector,
                "metadata": {
                    "anime_id": chunk.anime_id,
                    "title": titles[chunk.anime_id],
                    "chunk_type": chunk.chunk_type,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            } for chunk, vector in zip(chunks, vectors)])
            # Upsert first: a failed write must not erase the last usable vectors.
            stale_ids = [
                stale_id
                for info, anime_chunks in anime_batch
                for stale_id in _stale_child_ids(info["id"], anime_chunks)
            ]
            if stale_ids:
                pc_index.delete(ids=stale_ids)
        except Exception as exc:
            errors += len(anime_batch)
            print(f"  批量嵌入/upsert 失败 ({len(anime_batch)} 部番剧): {exc}")
            anime_batch = []
            return

        for info, _ in anime_batch:
            processed.add(str(info["id"]))
            metadata_entries.append(_make_metadata_entry(info))
            embed_count += 1
        anime_batch = []
        if embed_count - last_saved_count >= SAVE_INTERVAL:
            _save_checkpoint(processed, embed_count, metadata_entries)
            last_saved_count = embed_count
            print(f"  进度: {embed_count}/{total} ({embed_count/total*100:.1f}%), "
                  f"模型: {embeddings.model}, 错误: {errors}, 维度跳过: {dim_errors}")

    for i, aid in enumerate(all_ids):
        if str(aid) in processed:
            continue

        try:
            info = _fetch_anime_detail(conn, aid)
            if not info:
                continue

            anime_batch.append((info, make_anime_chunks(info)))

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  错误 (活跃模型: {embeddings.model}, id={aid}): {e}")
            continue

        # 批量上传
        if len(anime_batch) >= BATCH_SIZE:
            flush_batch()

    # 最后一批
    flush_batch()

    _save_checkpoint(processed, embed_count, metadata_entries)
    conn.close()
    print(f"Pinecone 构建完成: {embed_count} 条, 错误: {errors}, 维度跳过: {dim_errors}")


# ══════════════════════════════════════════════════════════════════════
# 3. 构建 Whoosh 索引
# ══════════════════════════════════════════════════════════════════════

def build_whoosh():
    """构建 Whoosh 稀疏检索索引"""
    import shutil
    from whoosh.index import create_in, exists_in
    from whoosh.fields import Schema, TEXT, ID
    from whoosh.analysis import NgramAnalyzer

    # 中文没有空格边界，使用字符 n-gram 避免整句被当成单个 token。
    chinese_analyzer = NgramAnalyzer(minsize=2, maxsize=4)

    conn = _get_db_conn()
    if not conn:
        return

    # 清理旧索引
    if os.path.exists(WHOOSH_DIR):
        shutil.rmtree(WHOOSH_DIR)
    os.makedirs(WHOOSH_DIR, exist_ok=True)

    analyzer = chinese_analyzer
    schema = Schema(
        id=ID(stored=True, unique=True),
        anime_id=ID(stored=True),
        chunk_type=ID(stored=True),
        content=TEXT(stored=True, analyzer=analyzer),
    )

    if not exists_in(WHOOSH_DIR):
        idx = create_in(WHOOSH_DIR, schema)
    else:
        from whoosh.index import open_dir
        idx = open_dir(WHOOSH_DIR)

    writer = idx.writer()
    all_ids = _fetch_anime_ids(conn)
    total = len(all_ids)
    count = 0

    for aid in all_ids:
        try:
            info = _fetch_anime_detail(conn, aid)
            if not info:
                continue
            chunks = make_anime_chunks(info)
            for chunk in chunks:
                writer.add_document(
                    id=chunk.id,
                    anime_id=str(aid),
                    chunk_type=chunk.chunk_type,
                    content=chunk.text,
                )
            count += len(chunks)
        except Exception as e:
            print(f"  Whoosh 错误 ({aid}): {e}")
            continue

        if count % 500 == 0:
            print(f"  Whoosh 已写入: {count} chunks")

    writer.commit()
    conn.close()
    print(f"Whoosh 索引构建完成: {count} chunks")


# ══════════════════════════════════════════════════════════════════════
# 4. 构建 Metadata Index
# ══════════════════════════════════════════════════════════════════════

def build_metadata():
    """从 SQLite 生成 metadata_index.json"""
    conn = _get_db_conn()
    if not conn:
        return

    all_ids = _fetch_anime_ids(conn)
    total = len(all_ids)
    entries = []

    for i, aid in enumerate(all_ids):
        try:
            info = _fetch_anime_detail(conn, aid)
            if not info:
                continue
            entries.append(_make_metadata_entry(info))
        except Exception as e:
            print(f"  Metadata 错误 ({aid}): {e}")
            continue

        if (i + 1) % 500 == 0:
            print(f"  Metadata 进度: {i+1}/{total}")

    # 按评分排序并分配 rank
    entries.sort(key=lambda x: x.get("score") or 0, reverse=True)
    for rank, entry in enumerate(entries, 1):
        entry["rank"] = rank

    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    conn.close()
    print(f"metadata_index.json 构建完成: {len(entries)} 条, 路径: {METADATA_PATH}")


# ══════════════════════════════════════════════════════════════════════
# 5. Checkpoint
# ══════════════════════════════════════════════════════════════════════

def _save_checkpoint(processed_ids: set, embed_count: int, metadata_entries: list):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "chunk_schema_version": CHUNK_SCHEMA_VERSION,
            "embedding_model_identity": embeddings.model_identity,
            "processed_ids": list(processed_ids),
            "embed_count": embed_count,
            "metadata_entries": metadata_entries,
        }, f, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 ACG 知识库")
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    parser.add_argument("--metadata-only", action="store_true", help="仅构建 metadata_index.json")
    parser.add_argument("--whoosh-only", action="store_true", help="仅构建 Whoosh 索引")
    args = parser.parse_args()

    config.validate()
    t0 = time.time()

    if args.metadata_only:
        build_metadata()
    elif args.whoosh_only:
        build_whoosh()
    else:
        print("=" * 60)
        print("  ACG 知识库构建")
        print(f"  数据源: {BANGUMI_DB}")
        print(f"  目标: Pinecone + Whoosh + MetadataIndex")
        print(f"  Embedding 后端: {config.EMBEDDING_BACKEND}")
        print(f"  Embedding 模型: {embeddings.model_identity}")
        print(f"  断点续跑: {'是' if args.resume else '否'}")
        print("=" * 60)

        # 1. Pinecone
        print("\n[1/3] 构建 Pinecone 索引...")
        build_pinecone(resume=args.resume)

        # 2. Metadata Index（优先从 checkpoint 保存）
        print("\n[2/3] 构建 Metadata Index...")
        if args.resume and os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                ck = json.load(f)
            entries = ck.get("metadata_entries", [])
            if entries:
                entries.sort(key=lambda x: x.get("score") or 0, reverse=True)
                for rank, entry in enumerate(entries, 1):
                    entry["rank"] = rank
                os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
                with open(METADATA_PATH, "w", encoding="utf-8") as f:
                    json.dump(entries, f, ensure_ascii=False, indent=2)
                print(f"  metadata_index.json 从 checkpoint 生成: {len(entries)} 条")
            else:
                build_metadata()
        else:
            build_metadata()

        # 3. Whoosh
        print("\n[3/3] 构建 Whoosh 索引...")
        build_whoosh()

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
