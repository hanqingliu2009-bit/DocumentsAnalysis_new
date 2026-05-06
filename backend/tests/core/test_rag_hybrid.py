"""Hybrid RAG: graph + splite vector in parallel."""
from unittest.mock import MagicMock, patch

import pytest

from config import settings
from core.rag import RAGPipeline


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Hybrid answer"))]
    client.chat.completions.create.return_value = resp
    return client


def test_hybrid_parallel_merge_and_llm(monkeypatch, mock_llm_client):
    monkeypatch.setattr(settings, "RAG_BACKEND", "hybrid")
    monkeypatch.setattr(settings, "HYBRID_SPLITE_COLLECTION", "splite_bge_zh_v15")
    monkeypatch.setattr(settings, "HYBRID_VECTOR_TOP_K", 3)
    monkeypatch.setattr(settings, "VOLCENGINE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LLM_MODEL", "ep-test")

    base_vs = MagicMock()
    with patch("core.external_graph.fetch_graph_context") as fg:
        fg.return_value = (
            "三元组示例",
            [
                {
                    "chunk_id": "graph-triplet-1",
                    "score": 1.0,
                    "text": "A -rel-> B",
                    "document_id": None,
                    "document_title": "图数据库·三元组",
                }
            ],
            {},
        )
        with patch("storage.vector_store.EmbeddingGenerator") as EG:
            EG.return_value.embed_texts.return_value = [[0.02] * 1024]
            with patch("storage.vector_store.VectorStore") as VS:
                VS.return_value.search.return_value = [
                    ("F_pdf_x:1", 0.88, "手册片段", "F_pdf_x"),
                ]
                pipe = RAGPipeline(base_vs, openai_client=mock_llm_client)
                out = pipe.query("如何检查电源？")

    assert out["answer"] == "Hybrid answer"
    assert out["answer_mode"] == "hybrid_graph_vector"
    assert len(out["sources"]) == 2
    assert out["sources"][0]["source_channel"] == "graph"
    assert out["sources"][1]["source_channel"] == "vector"
    EG.return_value.embed_texts.assert_called_once()
    assert EG.return_value.embed_texts.call_args.kwargs.get("embedding_backend") == "local"
    VS.assert_called_once()
    assert VS.call_args.kwargs["collection_name"] == "splite_bge_zh_v15"
    mock_llm_client.chat.completions.create.assert_called_once()
    user_msg = mock_llm_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "### 图数据库检索结果" in user_msg
    assert "### 本地手册向量检索" in user_msg


def test_hybrid_no_hits_llm_direct(monkeypatch, mock_llm_client):
    monkeypatch.setattr(settings, "RAG_BACKEND", "hybrid")
    monkeypatch.setattr(settings, "VOLCENGINE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LLM_MODEL", "ep-test")

    base_vs = MagicMock()
    with patch("core.external_graph.fetch_graph_context") as fg:
        fg.return_value = ("", [], {})
        with patch("storage.vector_store.EmbeddingGenerator") as EG:
            EG.return_value.embed_texts.return_value = [[0.02] * 1024]
            with patch("storage.vector_store.VectorStore") as VS:
                VS.return_value.search.return_value = []
                pipe = RAGPipeline(base_vs, openai_client=mock_llm_client)
                out = pipe.query("无关问题")

    assert out["answer_mode"] == "llm_direct"
    assert out["sources"] == []
