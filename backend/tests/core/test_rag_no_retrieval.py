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


def test_empty_retrieval_with_chat_flag_calls_llm(mock_llm_client):
    """Chat path: empty retrieval + key/model + setting -> general LLM, no sources."""
    vs = MagicMock()
    vs.search.return_value = []
    vs.get_chunk_count.return_value = 0
    with patch("storage.vector_store.EmbeddingGenerator") as EG:
        EG.return_value.embed_text.return_value = [0.01] * 384
        with patch.object(settings, "VOLCENGINE_API_KEY", "test-key"), patch.object(
            settings, "LLM_MODEL", "ep-test"
        ), patch.object(settings, "CHAT_ALLOW_GENERAL_WITHOUT_DOCS", True):
            pipeline = RAGPipeline(vs, openai_client=mock_llm_client)
            result = pipeline.query("What is 2+2?", allow_general_llm_when_empty=True)

    assert result["answer"] == "General LLM reply"
    assert result["sources"] == []
    assert result["confidence"] == 0.0
    assert result["context_used"] == 0
    assert "error" not in result
    mock_llm_client.chat.completions.create.assert_called_once()
    call_kw = mock_llm_client.chat.completions.create.call_args
    messages = call_kw.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "语义检索" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is 2+2?"


def test_empty_retrieval_query_endpoint_skips_llm(mock_llm_client):
    """/api/query path: default allow_general False -> no LLM call."""
    vs = MagicMock()
    vs.search.return_value = []
    vs.get_chunk_count.return_value = 0
    with patch("storage.vector_store.EmbeddingGenerator") as EG:
        EG.return_value.embed_text.return_value = [0.01] * 384
        with patch.object(settings, "VOLCENGINE_API_KEY", "test-key"), patch.object(
            settings, "LLM_MODEL", "ep-test"
        ), patch.object(settings, "CHAT_ALLOW_GENERAL_WITHOUT_DOCS", True):
            pipeline = RAGPipeline(vs, openai_client=mock_llm_client)
            result = pipeline.query("unrelated question xyz")

    mock_llm_client.chat.completions.create.assert_not_called()
    assert "没有检索到" not in result.get("answer", "")
    assert "已索引的文档" in result["answer"] or "上传" in result["answer"]


def test_empty_retrieval_chat_disabled_by_config(mock_llm_client):
    """CHAT_ALLOW_GENERAL_WITHOUT_DOCS=False -> fixed message, no LLM."""
    vs = MagicMock()
    vs.search.return_value = []
    vs.get_chunk_count.return_value = 0
    with patch("storage.vector_store.EmbeddingGenerator") as EG:
        EG.return_value.embed_text.return_value = [0.01] * 384
        with patch.object(settings, "VOLCENGINE_API_KEY", "test-key"), patch.object(
            settings, "LLM_MODEL", "ep-test"
        ), patch.object(settings, "CHAT_ALLOW_GENERAL_WITHOUT_DOCS", False):
            pipeline = RAGPipeline(vs, openai_client=mock_llm_client)
            result = pipeline.query("hello", allow_general_llm_when_empty=True)

    mock_llm_client.chat.completions.create.assert_not_called()
    assert result["sources"] == []


def test_empty_retrieval_no_api_key_no_llm(mock_llm_client):
    """No API key -> never call LLM on empty retrieval even for chat."""
    vs = MagicMock()
    vs.search.return_value = []
    vs.get_chunk_count.return_value = 0
    with patch("storage.vector_store.EmbeddingGenerator") as EG:
        EG.return_value.embed_text.return_value = [0.01] * 384
        with patch.object(settings, "VOLCENGINE_API_KEY", None), patch.object(
            settings, "LLM_MODEL", "ep-test"
        ), patch.object(settings, "CHAT_ALLOW_GENERAL_WITHOUT_DOCS", True):
            pipeline = RAGPipeline(vs, openai_client=mock_llm_client)
            result = pipeline.query("hello", allow_general_llm_when_empty=True)

    mock_llm_client.chat.completions.create.assert_not_called()
    assert result["answer"]
    assert "向量库" in result["answer"] or "文档" in result["answer"]


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
        ), patch.object(settings, "CHAT_ALLOW_GENERAL_WITHOUT_DOCS", True):
            pipeline = RAGPipeline(vs, openai_client=mock_llm_client, document_store=ds)
            pipeline.query("看一下我上传的文档", allow_general_llm_when_empty=True)
    system = mock_llm_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "12" in system
    assert "不要" in system
    assert "SIMILARITY_THRESHOLD" in system
