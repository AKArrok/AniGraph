import pytest

import config


def test_retrieval_settings_accept_default_candidate_funnel():
    config.validate_retrieval_settings()


def test_retrieval_settings_reject_mmr_pool_smaller_than_dense_pool(monkeypatch):
    monkeypatch.setattr(config, "RETRIEVER_FETCH_K", 10)
    monkeypatch.setattr(config, "HYBRID_DENSE_K", 20)

    with pytest.raises(ValueError, match="RETRIEVER_FETCH_K"):
        config.validate_retrieval_settings()


def test_retrieval_settings_reject_rerank_pool_smaller_than_final(monkeypatch):
    monkeypatch.setattr(config, "RETRIEVER_K", 5)
    monkeypatch.setattr(config, "RERANK_TOP_K", 4)

    with pytest.raises(ValueError, match="RERANK_TOP_K"):
        config.validate_retrieval_settings()


def test_validate_rejects_non_1024_coding_plan_dimension(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "ark")
    monkeypatch.setattr(config, "ARK_EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(config, "ARK_EMBEDDING_MODEL", "doubao-embedding-vision")
    monkeypatch.setattr(config, "ARK_EMBEDDING_DIMENSIONS", 2048)
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "PINECONE_API_KEY", "test-key")
    monkeypatch.setattr(config, "TAVILY_API_KEY", "test-key")

    with pytest.raises(ValueError, match="1024 dimensions"):
        config.validate()
