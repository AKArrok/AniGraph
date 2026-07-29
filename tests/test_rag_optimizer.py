from langchain_core.documents import Document

from tools import rag_optimizer
from tools.knowledge_retrieval import compress_docs


class _FakeRetriever:
    def invoke(self, query):
        return [Document(page_content=f"dense result for {query}")]


def test_retrieve_with_optimization_records_phase_timings(monkeypatch):
    monkeypatch.setattr(rag_optimizer, "_resolve_nickname", lambda query: None)
    monkeypatch.setattr(
        rag_optimizer,
        "search_whoosh",
        lambda query, k: [(f"sparse result for {query}", 1.0)],
    )

    docs, strategy = rag_optimizer.retrieve_with_optimization(
        "科幻动画",
        _FakeRetriever(),
        k_final=2,
        skip_optimization=True,
    )

    debug = rag_optimizer.get_last_debug()
    assert strategy == "direct"
    assert docs
    assert debug["dense_retrieved"] == 1
    assert debug["sparse_retrieved"] == 1
    assert debug["total_seconds"] >= 0
    assert "科幻动画" in debug["dense_seconds_per_query"]
    assert "科幻动画" in debug["sparse_seconds_per_query"]


def test_compress_docs_keeps_distinct_anime_titles():
    docs = [
        "番剧: 命运石之门\n简介: 时间机器悬疑。",
        "番剧：命运石之门\n评论: 时间线设计严密。",
        "【番剧】: 命运石之门 0\n简介: 续作。",
        "番剧: 命运石之门 0\n评论: 另一条世界线。",
        "番剧: 全部成为F THE PERFECT INSIDER\n简介: 密室推理。",
        "番剧: 四叠半神话大系\n简介: 平行世界。",
    ]

    compressed = compress_docs(docs, "类似命运石之门的动画", top_k=4)

    assert len(compressed) == 4
    assert sum("番剧: 命运石之门\n" in doc for doc in compressed) == 1
    assert sum("命运石之门 0" in doc for doc in compressed) == 1
    assert any("全部成为F" in doc for doc in compressed)
    assert any("四叠半神话大系" in doc for doc in compressed)
