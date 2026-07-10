"""RAG 全链路优化 — 门面（组合 Query Processing + Knowledge Retrieval）

架构:
  tools/query_processing.py    — 查询分类 + 优化（Alias/Rewrite/HyDE/Decompose）
  tools/knowledge_retrieval.py — 混合检索 + 融合 + 精排 + 压缩 + 验证
  tools/rag_optimizer.py       — 门面（本文件），组合上述模块

对外接口: retrieve_with_optimization() / get_last_debug()
"""
from tools.query_processing import classify, multi_query_rewrite, hyde_generate, decompose
from tools.knowledge_retrieval import search_whoosh, fusion, rerank, compress_docs, verify_answer
import config

# ══════════════════════════════════════════════════════════════════════
# Re-export 所有子模块符号（向后兼容）
# ══════════════════════════════════════════════════════════════════════

__all__ = [
    # Query Processing
    "classify", "multi_query_rewrite", "hyde_generate", "decompose",
    # Knowledge Retrieval
    "search_whoosh", "fusion", "fusion_rrf", "fusion_weighted", "fusion_max",
    "rerank", "compress_docs", "verify_answer",
    # Main
    "retrieve_with_optimization", "get_last_debug",
]

from tools.knowledge_retrieval import fusion_rrf, fusion_weighted, fusion_max

# ══════════════════════════════════════════════════════════════════════
# 昵称解析 — 联网搜索番剧简称 → 正式名称
# ══════════════════════════════════════════════════════════════════════

_RESOLVE_PROMPT = """从搜索结果中，提取查询词对应的番剧正式名称（如"素晴"→"为美好的世界献上祝福"）。如果搜索结果没有明确指向某部番剧，输出"无"。

查询: {query}

搜索结果:
{results}

正式名称或"无":"""


def _call_llm_raw(prompt: str, temperature: float = 0.7) -> str:
    from llms import answer_LLM
    from langchain_core.messages import HumanMessage
    resp = answer_LLM.bind(temperature=temperature).invoke([HumanMessage(content=prompt)])
    return resp.content.strip()


def _resolve_nickname(query: str) -> str | None:
    """通过联网搜索解析番剧昵称 → 正式名称（先查本地 alias 字典 + MetadataIndex）"""
    # 0. 本地 MetadataIndex 优先
    try:
        from agents.metadata_index import index
        result = index.get_by_alias(query)
        if result:
            return result.get("name_cn") or query
    except Exception:
        pass

    # 1. 本地 Alias 字典
    try:
        from agents.alias import resolve_alias_dict
        resolved = resolve_alias_dict(query)
        if resolved:
            return resolved
    except Exception:
        pass

    # 2. 只对短查询（疑似昵称）触发联网
    if len(query) >= 20:
        return None
    skip_markers = ["推荐", "评分", "排名", "好看的", "有哪些", "类似的"]
    if any(m in query for m in skip_markers):
        return None

    try:
        from tools.web_search import search_web
        search_text = search_web.invoke(f"{query} 是什么动漫 原名 番剧")
        if not search_text or len(search_text) < 20:
            return None

        prompt = _RESOLVE_PROMPT.format(query=query, results=search_text[:1500])
        name = _call_llm_raw(prompt, temperature=0).strip()
        if name and name != "无" and name != query:
            return name
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════
# 调试信息
# ══════════════════════════════════════════════════════════════════════

_last_debug: dict = {}

def get_last_debug() -> dict:
    return _last_debug


# ══════════════════════════════════════════════════════════════════════
# 主入口 — 全链路优化
# ══════════════════════════════════════════════════════════════════════

def retrieve_with_optimization(
    query: str,
    dense_retriever,
    k_final: int = 5,
    skip_optimization: bool = False,
) -> tuple[list[str], str]:
    """全链路 RAG 检索，返回 (文档列表, strategy)

    skip_optimization=True: 上游已做查询优化，跳过 rewrite/hyde/decompose
    """
    global _last_debug
    _last_debug = {"query": query}

    # Step 0: 昵称解析（本地优先，联网 fallback）
    resolved_name = _resolve_nickname(query)
    if resolved_name:
        _last_debug["nickname_resolved"] = f"{query} → {resolved_name}"
        search_query = f"{query} {resolved_name}"
    else:
        search_query = query

    # Step 1: 分类（跳过时直接 direct）
    if skip_optimization:
        strategy = "direct"
        queries = [search_query]
        _last_debug["optimization"] = "已由上游优化，跳过"
        _last_debug["classification"] = "direct"
    else:
        strategy = classify(search_query) if config.ENABLE_QUERY_OPTIMIZATION else "rewrite"
        _last_debug["classification"] = strategy

        # Step 2: 查询优化
        if strategy == "rewrite":
            queries = multi_query_rewrite(search_query)
            _last_debug["optimization"] = "Multi-Query Rewrite"
        elif strategy == "hyde":
            queries = hyde_generate(search_query)
            _last_debug["optimization"] = "HyDE (假设性答案生成)"
        elif strategy == "decompose":
            queries = decompose(search_query)
            _last_debug["optimization"] = "Query Decompose (查询拆分)"
        else:
            queries = [search_query]
            _last_debug["optimization"] = "无（Direct 直通）"

        if not config.ENABLE_QUERY_OPTIMIZATION:
            queries = [search_query]
            _last_debug["optimization"] = "已禁用"

    _last_debug["rewritten_queries"] = queries
    _last_debug["query_count"] = len(queries)

    # Step 3: 多路检索（并行 Dense + Sparse）
    all_dense: list = []
    dense_counts: dict = {}
    all_sparse: list = []
    sparse_counts: dict = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _retrieve_dense(q: str):
        try:
            docs = dense_retriever.invoke(q)
            return ("dense", q, [(d.page_content, 1.0) for d in docs], len(docs))
        except Exception:
            return ("dense", q, [], 0)

    def _retrieve_sparse(q: str):
        docs = search_whoosh(q, k=config.HYBRID_SPARSE_K)
        return ("sparse", q, docs, len(docs))

    max_workers = min(len(queries) * 2, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for q in queries:
            futures.append(executor.submit(_retrieve_dense, q))
            futures.append(executor.submit(_retrieve_sparse, q))
        for future in as_completed(futures):
            kind, q, docs, cnt = future.result()
            if kind == "dense":
                all_dense.extend(docs)
                dense_counts[q] = cnt
            else:
                all_sparse.extend(docs)
                sparse_counts[q] = cnt

    _last_debug["dense_retrieved"] = sum(dense_counts.values())
    _last_debug["sparse_retrieved"] = sum(sparse_counts.values())
    _last_debug["dense_per_query"] = dense_counts
    _last_debug["sparse_per_query"] = sparse_counts

    # Step 4: 融合
    if all_sparse:
        fused_docs = fusion(all_dense, all_sparse)
        _last_debug["fusion_strategy"] = config.FUSION_STRATEGY
    else:
        seen = set()
        fused_docs = []
        for doc, _ in all_dense:
            if doc not in seen:
                seen.add(doc)
                fused_docs.append(doc)
        _last_debug["fusion_strategy"] = "Dense-only (无稀疏索引)"

    _last_debug["post_fusion_count"] = len(fused_docs)

    # Step 5: 精排
    if config.ENABLE_RERANKING and len(fused_docs) > k_final:
        fused_docs = rerank(search_query, fused_docs, top_k=k_final * 2)
        _last_debug["reranking"] = f"{config.RERANKER_MODEL} (CrossEncoder)"
    else:
        _last_debug["reranking"] = "跳过" if not config.ENABLE_RERANKING else f"跳过 (仅 {len(fused_docs)} 条，无需精排)"
    _last_debug["post_rerank_count"] = len(fused_docs)

    # Step 6: 压缩
    pre_compress = len(fused_docs)
    if config.ENABLE_COMPRESSION:
        fused_docs = compress_docs(fused_docs, query, top_k=k_final)
        _last_debug["compression"] = f"{pre_compress} → {len(fused_docs)} (去重 + 截断)"
    else:
        _last_debug["compression"] = "已禁用"

    _last_debug["final_count"] = len(fused_docs[:k_final])

    return fused_docs[:k_final], strategy
