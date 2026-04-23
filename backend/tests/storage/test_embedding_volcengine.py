"""EmbeddingGenerator when EMBEDDING_BACKEND=volcengine (Ark /embeddings)."""
from unittest.mock import MagicMock, patch

import pytest

from config import settings
from storage.vector_store import EmbeddingGenerator


def test_embed_volcengine_calls_openai_embeddings(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_BACKEND", "volcengine")
    monkeypatch.setattr(settings, "VOLCENGINE_API_KEY", "k-test")
    monkeypatch.setattr(settings, "VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "ep-embedding-test")

    row = MagicMock()
    row.index = 0
    row.embedding = [0.5, 0.25, 0.125]
    resp = MagicMock()
    resp.data = [row]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = resp

    with patch("storage.vector_store.openai.OpenAI", return_value=mock_client):
        gen = EmbeddingGenerator()
        out = gen.embed_texts(["hello"])

    assert out == [[0.5, 0.25, 0.125]]
    mock_client.embeddings.create.assert_called_once()
    kw = mock_client.embeddings.create.call_args.kwargs
    assert kw["model"] == "ep-embedding-test"
    assert kw["input"] == ["hello"]


def test_embed_volcengine_requires_model(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_BACKEND", "volcengine")
    monkeypatch.setattr(settings, "VOLCENGINE_API_KEY", "k")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "")

    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        EmbeddingGenerator().embed_text("x")
