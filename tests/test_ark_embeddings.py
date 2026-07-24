from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from llms import ArkCodingEmbeddings


class FakeRateLimitError(Exception):
    status_code = 429


def _embedding_client(*side_effects):
    embedding = ArkCodingEmbeddings(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="doubao-embedding-vision",
        dimension=1024,
    )
    embedding.request_interval = 0
    embedding.client.embeddings.create = Mock(side_effect=side_effects)
    return embedding


def test_rate_limit_is_retried_with_backoff():
    response = SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 1024)])
    embedding = _embedding_client(FakeRateLimitError("TooManyRequests"), response)

    with patch("llms.random.uniform", return_value=0), patch("llms.time.sleep") as sleep:
        result = embedding.embed_query("test")

    assert len(result) == 1024
    assert embedding.client.embeddings.create.call_count == 2
    sleep.assert_called_once_with(1)


def test_rate_limit_backoff_never_undercuts_request_interval():
    response = SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 1024)])
    embedding = _embedding_client(FakeRateLimitError("TooManyRequests"), response)
    embedding.request_interval = 3

    with patch.object(embedding, "_wait_for_request_slot"), patch(
        "llms.random.uniform", return_value=0
    ), patch("llms.time.sleep") as sleep:
        embedding.embed_query("test")

    sleep.assert_called_once_with(3)


def test_non_rate_limit_error_is_not_retried():
    embedding = _embedding_client(ValueError("invalid request"))

    with patch("llms.time.sleep") as sleep, pytest.raises(ValueError, match="invalid request"):
        embedding.embed_query("test")

    assert embedding.client.embeddings.create.call_count == 1
    sleep.assert_not_called()
