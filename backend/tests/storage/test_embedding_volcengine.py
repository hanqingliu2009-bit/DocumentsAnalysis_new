"""EmbeddingGenerator when EMBEDDING_BACKEND=volcengine (Ark /embeddings)."""
from unittest.mock import MagicMock, patch

import pytest

from config import settings
from core.document import DocumentChunk
from storage.vector_store import EmbeddingGenerator, _ark_embedding_data_rows


def test_embed_volcengine_calls_openai_embeddings(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_BACKEND", "volcengine")
    monkeypatch.setattr(settings, "EMBEDDING_USE_MULTIMODAL_API", False)
    monkeypatch.setattr(settings, "VOLCENGINE_EMBEDDING_API_KEY", None)
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

    with patch("storage.vector_store.openai.OpenAI", return_value=mock_client) as OC:
        gen = EmbeddingGenerator()
        out = gen.embed_texts(["hello"])

    assert OC.call_args.kwargs["api_key"] == "k-test"
    assert out == [[0.5, 0.25, 0.125]]
    mock_client.embeddings.create.assert_called_once()
    kw = mock_client.embeddings.create.call_args.kwargs
    assert kw["model"] == "ep-embedding-test"
    assert kw["input"] == ["hello"]


def test_embed_volcengine_requires_model(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_BACKEND", "volcengine")
    monkeypatch.setattr(settings, "EMBEDDING_USE_MULTIMODAL_API", False)
    monkeypatch.setattr(settings, "VOLCENGINE_API_KEY", "k")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "")

    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        EmbeddingGenerator().embed_text("x")


def test_embed_volcengine_prefers_embedding_api_key(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_BACKEND", "volcengine")
    monkeypatch.setattr(settings, "EMBEDDING_USE_MULTIMODAL_API", False)
    monkeypatch.setattr(settings, "VOLCENGINE_API_KEY", "main-key")
    monkeypatch.setattr(settings, "VOLCENGINE_EMBEDDING_API_KEY", "embed-only-key")
    monkeypatch.setattr(settings, "VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "ep-emb")

    row = MagicMock(index=0, embedding=[1.0])
    resp = MagicMock(data=[row])
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = resp

    with patch("storage.vector_store.openai.OpenAI", return_value=mock_client) as OC:
        EmbeddingGenerator().embed_texts(["a"])

    OC.assert_called_once()
    assert OC.call_args.kwargs["api_key"] == "embed-only-key"


def test_ark_embedding_data_rows_single_dict_not_list():
    """Ark may return data as one object; iterating a dict would otherwise yield str keys."""
    body = {"data": {"index": 1, "embedding": [9.0, 8.0]}}
    rows = _ark_embedding_data_rows(body)
    assert rows == [{"index": 1, "embedding": [9.0, 8.0]}]


def test_embed_volcengine_multimodal_calls_httpx(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_BACKEND", "volcengine")
    monkeypatch.setattr(settings, "EMBEDDING_USE_MULTIMODAL_API", True)
    monkeypatch.setattr(settings, "VOLCENGINE_API_KEY", "mm-key")
    monkeypatch.setattr(settings, "VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "ep-mm")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
    mock_resp.raise_for_status = MagicMock()
    inner = MagicMock()
    inner.post.return_value = mock_resp
    client_cm = MagicMock()
    client_cm.__enter__ = MagicMock(return_value=inner)
    client_cm.__exit__ = MagicMock(return_value=False)

    with patch("storage.vector_store.httpx.Client", return_value=client_cm):
        out = EmbeddingGenerator().embed_texts(["hello"])

    assert out == [[0.1, 0.2]]
    inner.post.assert_called_once()
    url = inner.post.call_args[0][0]
    assert url.endswith("/embeddings/multimodal")
    body = inner.post.call_args.kwargs["json"]
    assert body["model"] == "ep-mm"
    assert body["input"] == [{"type": "text", "text": "hello"}]
    hdrs = inner.post.call_args.kwargs["headers"]
    assert hdrs["Authorization"] == "Bearer mm-key"


def test_embed_chunks_multimodal_includes_image_urls(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_BACKEND", "volcengine")
    monkeypatch.setattr(settings, "EMBEDDING_USE_MULTIMODAL_API", True)
    monkeypatch.setattr(settings, "VOLCENGINE_API_KEY", "k")
    monkeypatch.setattr(settings, "VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "ep-mm")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"index": 0, "embedding": [3.0]}]}
    mock_resp.raise_for_status = MagicMock()
    inner = MagicMock()
    inner.post.return_value = mock_resp
    client_cm = MagicMock()
    client_cm.__enter__ = MagicMock(return_value=inner)
    client_cm.__exit__ = MagicMock(return_value=False)

    chunk = DocumentChunk(
        document_id="d1",
        text="caption",
        chunk_index=0,
        start_char=0,
        end_char=7,
        metadata={"embedding_image_urls": ["https://example.com/x.png"]},
    )

    with patch("storage.vector_store.httpx.Client", return_value=client_cm):
        EmbeddingGenerator().embed_chunks([chunk])

    assert chunk.embedding == [3.0]
    body = inner.post.call_args.kwargs["json"]["input"]
    assert body[0] == {"type": "text", "text": "caption"}
    assert body[1] == {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}


def test_embed_volcengine_multimodal_parses_data_as_single_object(monkeypatch):
    """Warmup-style path when API returns data as object instead of array."""
    monkeypatch.setattr(settings, "EMBEDDING_BACKEND", "volcengine")
    monkeypatch.setattr(settings, "EMBEDDING_USE_MULTIMODAL_API", True)
    monkeypatch.setattr(settings, "VOLCENGINE_API_KEY", "k")
    monkeypatch.setattr(settings, "VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "ep-mm")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {"index": 0, "embedding": [0.25, 0.5]},
    }
    mock_resp.raise_for_status = MagicMock()
    inner = MagicMock()
    inner.post.return_value = mock_resp
    client_cm = MagicMock()
    client_cm.__enter__ = MagicMock(return_value=inner)
    client_cm.__exit__ = MagicMock(return_value=False)

    with patch("storage.vector_store.httpx.Client", return_value=client_cm):
        out = EmbeddingGenerator().embed_texts(["x"])

    assert out == [[0.25, 0.5]]
