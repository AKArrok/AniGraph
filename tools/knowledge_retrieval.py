"""Knowledge Retrieval Layer — 混合检索 + 融合 + 精排 + 压缩 + 验证

模块:
  WhooshRetriever   — 稀疏检索（BM25F）
  Fusion            — RRF / Weighted / Max 融合
  CrossEncoderReranker — 精排
  ContextCompressor — 去重 + 关键句提取
  AnswerVerifier    — 检索依据检查
"""
import os
from typing import Literal

import config

# ══════════════════════════════════════════════════════════════════════
# 1. WhooshRetriever — 稀疏检索
# ══════════════════════════════════════════════════════════════════════

_whoosh_index = None


def _get_whoosh_index():
    global _whoosh_index
    if _whoosh_index is not None:
        return _whoosh_index
    if not os.path.exists(config.WHOOSH_INDEX_DIR):
        return None
    from whoosh.index import open_dir
    _whoosh_index = open_dir(config.WHOOSH_INDEX_DIR)
    return _whoosh_index


def search_whoosh(query: str, k: int = 10) -> list[tuple[str, float]]:
    """Whoosh 稀疏检索，返回 [(content, score), ...]

    优化策略:
    - 中文分词用RegexAnalyzer ([\u4e00-\u9fff]+|[a-zA-Z0-9]+)
    - 关键词提取: 去掉标点/停用词, 只保留2字以上中文词
    - OrGroup: 任一关键词命中即可 → 避免长句全词&导致的0结果
    - 空关键词兜底: 用原query做更宽容的搜索
    """
    idx = _get_whoosh_index()
    if idx is None:
        return []

    from whoosh.qparser import MultifieldParser, OrGroup, QueryParser
    from whoosh import scoring
    import re

    # 1. 提取中文关键词（2字以上）和英文token
    keywords = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]{2,}', query)

    # 2. 构建搜索词: 有足够关键词用OrGroup，否则用原query兜底
    if len(keywords) >= 2:
        search_query = " OR ".join(keywords)
    elif keywords:
        search_query = keywords[0]  # 单关键词直接搜
    else:
        search_query = query  # 空关键词兜底

    with idx.searcher(weighting=scoring.BM25F()) as searcher:
        parser = MultifieldParser(["content"], idx.schema, group=OrGroup)
        try:
            q = parser.parse(search_query)
            results = searcher.search(q, limit=k)
            return [(r["content"], r.score) for r in results]
        except Exception:
            # 解析失败兜底: 用更简单的 query parser
            try:
                qp = QueryParser("content", idx.schema)
                q = qp.parse(search_query)
                results = searcher.search(q, limit=k)
                return [(r["content"], r.score) for r in results]
            except Exception:
                return []


# ══════════════════════════════════════════════════════════════════════
# 2. Fusion — 多策略融合
# ══════════════════════════════════════════════════════════════════════

def _normalize_scores(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not items:
        return items
    scores = [s for _, s in items]
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [(d, 1.0) for d, _ in items]
    return [(d, (s - mn) / (mx - mn)) for d, s in items]


def fusion_rrf(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion"""
    scores: dict[str, float] = {}
    for rank, (doc, _) in enumerate(dense, 1):
        scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank)
    for rank, (doc, _) in enumerate(sparse, 1):
        scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fusion_weighted(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    w_dense: float = 0.7,
    w_sparse: float = 0.3,
) -> list[tuple[str, float]]:
    """加权融合"""
    d_norm = dict(_normalize_scores(dense))
    s_norm = dict(_normalize_scores(sparse))
    scores: dict[str, float] = {}
    for doc in set(d_norm) | set(s_norm):
        scores[doc] = w_dense * d_norm.get(doc, 0) + w_sparse * s_norm.get(doc, 0)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fusion_max(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Max 融合: 任一命中即高分"""
    d_norm = dict(_normalize_scores(dense))
    s_norm = dict(_normalize_scores(sparse))
    scores: dict[str, float] = {}
    for doc in set(d_norm) | set(s_norm):
        scores[doc] = max(d_norm.get(doc, 0), s_norm.get(doc, 0))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fusion(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
) -> list[str]:
    """统一融合入口，返回文档内容列表"""
    if config.FUSION_STRATEGY == "weighted":
        merged = fusion_weighted(dense, sparse)
    elif config.FUSION_STRATEGY == "max":
        merged = fusion_max(dense, sparse)
    else:
        merged = fusion_rrf(dense, sparse)

    seen = set()
    result = []
    for doc, _ in merged:
        if doc not in seen:
            seen.add(doc)
            result.append(doc)
    return result


# ══════════════════════════════════════════════════════════════════════
# 3. CrossEncoderReranker — 精排
# ══════════════════════════════════════════════════════════════════════

_reranker = None


def _init_reranker():
    """启动时预加载 CrossEncoder，避免首查询卡顿"""
    global _reranker
    try:
        from sentence_transformers import CrossEncoder
        model_path = config.LOCAL_RERANKER_MODEL or config.RERANKER_MODEL
        tag = "local" if config.LOCAL_RERANKER_MODEL and os.path.exists(config.LOCAL_RERANKER_MODEL) else "HF"
        print(f"  CrossEncoder ({tag}): {model_path} ...")
        _reranker = CrossEncoder(model_path)
        print(f"  CrossEncoder 就绪")
    except Exception as e:
        print(f"  CrossEncoder 加载失败: {e}, 精排禁用")
        _reranker = False


def _get_reranker():
    global _reranker
    if _reranker is None:
        _init_reranker()
    return _reranker


def rerank(query: str, docs: list[str], top_k: int = 5) -> list[str]:
    """交叉编码器精排"""
    if not docs:
        return []
    ranker = _get_reranker()
    if not ranker:
        return docs[:top_k]

    pairs = [(query, doc[:500]) for doc in docs]
    scores = ranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# ══════════════════════════════════════════════════════════════════════
# 4. ContextCompressor — 上下文压缩
# ══════════════════════════════════════════════════════════════════════

def compress_docs(docs: list[str], query: str, top_k: int = 5) -> list[str]:
    """去重 + 取最相关段落"""
    def _sim(a: str, b: str) -> float:
        a_grams = set(a[i:i+3] for i in range(len(a)-3))
        b_grams = set(b[i:i+3] for i in range(len(b)-3))
        if not a_grams or not b_grams:
            return 0
        return len(a_grams & b_grams) / len(a_grams | b_grams)

    unique = []
    for doc in docs:
        if all(_sim(doc, u) < 0.6 for u in unique):
            unique.append(doc)

    compressed = [doc[:500] for doc in unique[:top_k]]
    return compressed


# ══════════════════════════════════════════════════════════════════════
# 5. AnswerVerifier — 回答验证
# ══════════════════════════════════════════════════════════════════════

def verify_answer(answer: str, docs: list[str]) -> tuple[str, float]:
    """检查回答是否有检索依据支撑，返回 (回答, 置信度)"""
    if not docs:
        return answer, 0.5

    doc_text = " ".join(docs)
    ans_words = set(answer)
    doc_words = set(doc_text)

    if not ans_words:
        return answer, 0.5

    overlap = len(ans_words & doc_words) / len(ans_words)

    if overlap < 0.15:
        return answer + "\n\n(注: 以上内容部分超出知识库范围，仅供参考)", 0.3
    return answer, min(overlap + 0.3, 1.0)


# ── 启动时预加载模型（避免首查询 Loading weights 卡顿）──
_init_reranker()
