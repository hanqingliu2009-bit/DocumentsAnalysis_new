"""Tests for supplier graph HTTP client."""
from unittest.mock import MagicMock, patch

from config import Settings


def test_fetch_graph_context_builds_context_and_sources(tmp_path):
    from core import external_graph

    body = {
        "success": True,
        "message": {
            "triplet": ["A -rel-> B", "C -rel-> D"],
            "summary": ["summary line"],
            "chunk": ["chunk text"],
        },
    }
    mock_response = MagicMock()
    mock_response.json.return_value = body
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_response

    s = Settings(
        DATA_DIR=tmp_path,
        VECTOR_DB_PATH=tmp_path / "v",
        DOCUMENT_STORE_PATH=tmp_path / "d",
        FILE_STORAGE_PATH=tmp_path / "f",
        EXTERNAL_GRAPH_API_URL="http://example.test/graph",
        EXTERNAL_GRAPH_DOMAIN="domain-x",
        EXTERNAL_GRAPH_SEARCH_TYPE="triplet",
    )

    with patch("config.settings", s):
        with patch("core.external_graph.httpx.Client", return_value=mock_client):
            context, sources, dbg = external_graph.fetch_graph_context("  hello  ")

    mock_client.post.assert_called_once()
    call_kw = mock_client.post.call_args
    assert call_kw[0][0] == "http://example.test/graph"
    assert call_kw[1]["json"]["query"] == "hello"
    assert call_kw[1]["json"]["domain"] == "domain-x"
    assert call_kw[1]["json"]["search_type"] == "triplet"

    assert "A -rel-> B" in context
    assert "summary line" in context
    assert "chunk text" in context
    assert len(sources) == 4
    assert sources[0]["chunk_id"] == "graph-triplet-1"
    assert dbg["meta"]["triplet_count"] == 2


def test_triplet_dict_formatted_as_line(tmp_path):
    """Structured triple objects from the graph API become head -rel-> tail lines."""
    from core import external_graph

    body = {
        "success": True,
        "message": {
            "triplet": [
                {"head": "A", "relation": "rel", "tail": "B"},
                {"subject": "X", "predicate": "p", "object": "Y"},
            ],
            "summary": [],
            "chunk": [],
        },
    }
    mock_response = MagicMock()
    mock_response.json.return_value = body
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_response

    s = Settings(
        DATA_DIR=tmp_path,
        VECTOR_DB_PATH=tmp_path / "v",
        DOCUMENT_STORE_PATH=tmp_path / "d",
        FILE_STORAGE_PATH=tmp_path / "f",
        EXTERNAL_GRAPH_API_URL="http://example.test/graph",
    )

    with patch("config.settings", s):
        with patch("core.external_graph.httpx.Client", return_value=mock_client):
            context, sources, _ = external_graph.fetch_graph_context("q")

    assert "A -rel-> B" in context
    assert "X -p-> Y" in context
    assert len(sources) == 2


def test_fetch_graph_context_empty_message(tmp_path):
    from core import external_graph

    mock_response = MagicMock()
    mock_response.json.return_value = {"success": True, "message": {"triplet": [], "summary": [], "chunk": []}}
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_response

    s = Settings(
        DATA_DIR=tmp_path,
        VECTOR_DB_PATH=tmp_path / "v",
        DOCUMENT_STORE_PATH=tmp_path / "d",
        FILE_STORAGE_PATH=tmp_path / "f",
        EXTERNAL_GRAPH_API_URL="http://example.test/graph",
    )

    with patch("config.settings", s):
        with patch("core.external_graph.httpx.Client", return_value=mock_client):
            context, sources, _ = external_graph.fetch_graph_context("q")

    assert context == ""
    assert sources == []
