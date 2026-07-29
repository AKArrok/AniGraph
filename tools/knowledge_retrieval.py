"""Knowledge Retrieval Layer — 混合检索 + 融合 + 精排 + 压缩 + 验证

模块:
  WhooshRetriever   — 稀疏检索（BM25F）
  Fusion            — RRF / Weighted / Max 融合
  CrossEncoderReranker — 精排
  ContextCompressor — 去重 + 关键句提取
  AnswerVerifier    — 检索依据检查
"""
import os
import threading
import time
import config

_SPARSE_STOP_WORDS = {
    "动漫", "动画", "番剧", "推荐", "评价", "观众", "作品", "哪些", "什么",
    "好看", "类似", "有关", "一个", "一些", "比较", "很高", "很低",
}


def _sparse_terms(query: str) -> list[str]:
    """Extract useful 2-4 character terms for the n-gram sparse index."""
    import re

    terms: list[str] = []
    for segment in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9][a-zA-Z0-9._+-]*", query):
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            for stop_word in sorted(_SPARSE_STOP_WORDS, key=len, reverse=True):
                segment = segment.replace(stop_word, " ")
            for word in segment.split():
                if 2 <= len(word) <= 4:
                    terms.append(word)
                elif len(word) > 4:
                    terms.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            terms.append(segment.lower())
    return list(dict.fromkeys(terms))

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
    - 中文索引使用 2-4 字符 n-gram，查询由同一 analyzer 切分
    - 关键词提取: 去掉标点/停用词, 只保留2字以上中文词
    - OrGroup: 任一关键词命中即可 → 避免长句全词&导致的0结果
    - 空关键词兜底: 用原query做更宽容的搜索
    """
    idx = _get_whoosh_index()
    if idx is None:
        return []

    from whoosh.query import Or, Term
    from whoosh import scoring

    terms = _sparse_terms(query)
    if not terms:
        return []

    with idx.searcher(weighting=scoring.BM25F()) as searcher:
        try:
            clauses = [Term("content", term) for term in terms]
            q = clauses[0] if len(clauses) == 1 else Or(clauses)
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
_reranker_lock = threading.Lock()
_reranker_last_failure = 0.0
_RERANKER_RETRY_SECONDS = 300.0


def _init_reranker():
    """加载 CrossEncoder；失败后退避一段时间再重试。"""
    global _reranker, _reranker_last_failure
    if _reranker is not None:
        return _reranker
    if (
        _reranker_last_failure
        and time.monotonic() - _reranker_last_failure < _RERANKER_RETRY_SECONDS
    ):
        return None
    try:
        from sentence_transformers import CrossEncoder
        model_path = config.LOCAL_RERANKER_MODEL or config.RERANKER_MODEL
        tag = "local" if config.LOCAL_RERANKER_MODEL and os.path.exists(config.LOCAL_RERANKER_MODEL) else "HF"
        print(f"  CrossEncoder ({tag}): {model_path} ...")
        _reranker = CrossEncoder(model_path)
        _reranker_last_failure = 0.0
        print(f"  CrossEncoder 就绪")
    except Exception as e:
        print(f"  CrossEncoder 加载失败: {e}, 精排禁用")
        _reranker_last_failure = time.monotonic()
    return _reranker


def _get_reranker():
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _init_reranker()
    return _reranker


def warm_reranker():
    """显式预热精排模型；模块导入本身不会再加载模型。"""
    return _get_reranker()


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
    import re

    # 按番剧标题去重，优先保留首次出现的标题
    _TITLE_PAT = re.compile(r"^(?:番剧|【番剧】)[:：]?\s*([^\n（(]+)", re.MULTILINE)

    def _extract_title(text: str) -> str:
        m = _TITLE_PAT.search(text)
        if not m:
            return ""
        return re.sub(r"[\s《》【】\[\]（）()·:：;；!！?？._-]", "", m.group(1)).casefold()

    # 对无标题标记的文档，用 trigram 相似度去重。标题明确的文档只按
    # 标准化标题精确去重；同系列判断属于推荐专家的职责。
    def _sim(a: str, b: str) -> float:
        a_grams = set(a[i:i+3] for i in range(len(a)-3))
        b_grams = set(b[i:i+3] for i in range(len(b)-3))
        if not a_grams or not b_grams:
            return 0
        return len(a_grams & b_grams) / len(a_grams | b_grams)

    seen_titles: set[str] = set()
    unstructured_docs: list[str] = []
    unique: list[str] = []
    for doc in docs:
        title = _extract_title(doc)
        if title:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            unique.append(doc)
            continue

        if all(_sim(doc, other) < 0.6 for other in unstructured_docs):
            unstructured_docs.append(doc)
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
