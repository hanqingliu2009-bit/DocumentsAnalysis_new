"""RAG behavior when vector search returns no chunks."""
from unittest.mock import MagicMock, patch

import pytest

from config import settings
from core.rag import RAGPipeline


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="General LLM reply"))]
    client.chat.completions.create.return_value = resp
    return client


def test_empty_retrieval_calls_llm_when_key_and_model_set(mock_llm_client):
    """Empty retrieval + configured Key/model -> always general LLM, no sources."""
    vs = MagicMock()
    vs.search.return_value = []
    vs.get_chunk_count.return_value = 0
    with patch("storage.vector_store.EmbeddingGenerator") as EG:
        EG.return_value.embed_text.return_value = [0.01] * 384
        with patch.object(settings, "VOLCENGINE_API_KEY", "test-key"), patch.object(
            settings, "LLM_MODEL", "ep-test"
        ):
            pipeline = RAGPipeline(vs, openai_client=mock_llm_client)
            result = pipeline.query("What is 2+2?")

    assert result["answer"] == "General LLM reply"
    assert result["sources"] == []
    assert result["confidence"] == 0.0
    assert result["context_used"] == 0
    assert result.get("answer_mode") == "llm_direct"
    assert "error" not in result
    mock_llm_client.chat.completions.create.assert_called_once()
    call_kw = mock_llm_client.chat.completions.create.call_args
    messages = call_kw.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "语义检索" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is 2+2?"


def test_empty_retrieval_same_for_query_style_call(mock_llm_client):
    """No allow_general flag: /api/query and /api/chat both get LLM on empty retrieval."""
    vs = MagicMock()
    vs.search.return_value = []
    vs.get_chunk_count.return_value = 0
    with patch("storage.vector_store.EmbeddingGenerator") as EG:
        EG.return_value.embed_text.return_value = [0.01] * 384
        with patch.object(settings, "VOLCENGINE_API_KEY", "test-key"), patch.object(
            settings, "LLM_MODEL", "ep-test"
        ):
            pipeline = RAGPipeline(vs, openai_client=mock_llm_client)
            result = pipeline.query("unrelated question xyz")

    mock_llm_client.chat.completions.create.assert_called_once()
    assert result["answer"] == "General LLM reply"
    assert result.get("answer_mode") == "llm_direct"


def test_empty_retrieval_no_api_key_returns_config_message(mock_llm_client):
    """No API key -> no LLM; operational message only."""
    vs = MagicMock()
    vs.search.return_value = []
    vs.get_chunk_count.return_value = 0
    with patch("storage.vector_store.EmbeddingGenerator") as EG:
        EG.return_value.embed_text.return_value = [0.01] * 384
        with patch.object(settings, "VOLCENGINE_API_KEY", None), patch.object(
            settings, "LLM_MODEL", "ep-test"
        ):
            pipeline = RAGPipeline(vs, openai_client=mock_llm_client)
            result = pipeline.query("hello")

    mock_llm_client.chat.completions.create.assert_not_called()
    assert "VOLCENGINE_API_KEY" in result["answer"]
    assert "LLM_MODEL" in result["answer"]
    assert result.get("answer_mode") == "system"


def test_empty_retrieval_blank_model_skips_llm(mock_llm_client):
    """Empty LLM_MODEL -> no LLM call."""
    vs = MagicMock()
    vs.search.return_value = []
    vs.get_chunk_count.return_value = 0
    with patch("storage.vector_store.EmbeddingGenerator") as EG:
        EG.return_value.embed_text.return_value = [0.01] * 384
        with patch.object(settings, "VOLCENGINE_API_KEY", "k"), patch.object(
            settings, "LLM_MODEL", "   "
        ):
            pipeline = RAGPipeline(vs, openai_client=mock_llm_client)
            result = pipeline.query("hello")

    mock_llm_client.chat.completions.create.assert_not_called()
    assert "LLM_MODEL" in result["answer"]
    assert result.get("answer_mode") == "system"


def test_no_retrieval_system_prompt_when_index_non_empty(mock_llm_client):
    """System prompt tells LLM not to claim user never uploaded when chunks exist."""
    vs = MagicMock()
    vs.search.return_value = []
    vs.get_chunk_count.return_value = 12
    ds = MagicMock()
    ds.count_by_status.return_value = {"completed": 2, "failed": 0, "pending": 0, "processing": 0}
    with patch("storage.vector_store.EmbeddingGenerator") as EG:
        EG.return_value.embed_text.return_value = [0.01] * 384
        with patch.object(settings, "VOLCENGINE_API_KEY", "k"), patch.object(
            settings, "LLM_MODEL", "ep"
        ):
            pipeline = RAGPipeline(vs, openai_client=mock_llm_client, document_store=ds)
            pipeline.query("看一下我上传的文档")
    system = mock_llm_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "12" in system
    assert "不要" in system
    assert "SIMILARITY_THRESHOLD" in system
