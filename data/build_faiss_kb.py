"""从本地 SQLite 构建 FAISS 向量库（Pinecone 不可用时的降级方案）

用法: python data/build_faiss_kb.py
输出: data/faiss_index/ (FAISS 索引 + 文档缓存)
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from data.build_kb import _fetch_anime_detail, _fetch_anime_ids, _get_db_conn
from data.chunking import CHUNK_SCHEMA_VERSION, make_anime_chunks
from llms import embeddings

FAISS_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
os.makedirs(FAISS_DIR, exist_ok=True)


def build_faiss_kb(max_anime: int = 100):
    """构建 FAISS 向量库（限制数量以加速评估）"""
    cache_path = os.path.join(FAISS_DIR, "chunks.pkl")
    expected_identity = embeddings.model_identity

    # 尝试从缓存加载
    if os.path.exists(cache_path) and os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
        print("Loading cached FAISS index...")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if (
            isinstance(cached, dict)
            and cached.get("schema_version") == CHUNK_SCHEMA_VERSION
            and cached.get("embedding_identity") == expected_identity
        ):
            vs = FAISS.load_local(
                FAISS_DIR, embeddings, allow_dangerous_deserialization=True
            )
            chunks = cached["chunks"]
            print(f"Loaded {len(chunks)} chunks from cache")
            return vs, chunks
        print("Chunk schema or embedding model changed; rebuilding FAISS index")

    conn = _get_db_conn()
    if conn is None:
        return None, []
    anime_ids = _fetch_anime_ids(conn)[:max_anime]
    chunks: list[Document] = []
    for anime_id in anime_ids:
        info = _fetch_anime_detail(conn, anime_id)
        if not info:
            continue
        chunks.extend(Document(
            page_content=chunk.text,
            metadata={
                "id": chunk.id,
                "anime_id": chunk.anime_id,
                "title": info.get("title", ""),
                "chunk_type": chunk.chunk_type,
                "chunk_index": chunk.chunk_index,
            },
        ) for chunk in make_anime_chunks(info))
    conn.close()
    print(f"Built {len(chunks)} semantic chunks from {len(anime_ids)} anime")

    # 构建 FAISS
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(FAISS_DIR)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "schema_version": CHUNK_SCHEMA_VERSION,
            "embedding_identity": expected_identity,
            "chunks": chunks,
        }, f)
    print(f"FAISS index saved: {len(chunks)} vectors")

    return vs, chunks


if __name__ == "__main__":
    vs, chunks = build_faiss_kb(max_anime=100)
    if vs:
        # 快速检索测试
        results = vs.similarity_search("科幻番剧推荐", k=5)
        print("\n--- Quick retrieval test ---")
        for i, doc in enumerate(results):
            print(f"\n[{i+1}] {doc.page_content[:150]}...")
