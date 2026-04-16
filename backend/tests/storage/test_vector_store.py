"""Tests for vector store module."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.document import DocumentChunk
from storage.vector_store import EmbeddingGenerator, VectorStore


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator."""

    def test_initialization(self):
        """Test that generator initializes with model name."""
        with patch("sentence_transformers.SentenceTransformer") as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            generator = EmbeddingGenerator(model_name="test-model")
            assert generator.model_name == "test-model"

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_texts(self, mock_model_class):
        """Test embedding multiple texts."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384, [0.2] * 384]
        mock_model_class.return_value = mock_model

        generator = EmbeddingGenerator()
        texts = ["Text one", "Text two"]
        embeddings = generator.embed_texts(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert embeddings[0][0] == 0.1

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_text(self, mock_model_class):
        """Test embedding single text."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]
        mock_model_class.return_value = mock_model

        generator = EmbeddingGenerator()
        embedding = generator.embed_text("Single text")

        assert len(embedding) == 384
        assert embedding[0] == 0.1

    def test_embed_empty_list(self):
        """Test embedding empty list returns empty."""
        generator = EmbeddingGenerator()
        embeddings = generator.embed_texts([])
        assert embeddings == []

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_chunks(self, mock_model_class):
        """Test embedding chunks in place."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384, [0.2] * 384]
        mock_model_class.return_value = mock_model

        generator = EmbeddingGenerator()
        chunks = [
            DocumentChunk(
                document_id="doc-1",
                text="Text one",
                chunk_index=0,
                start_char=0,
                end_char=100,
            ),
            DocumentChunk(
                document_id="doc-1",
                text="Text two",
                chunk_index=1,
                start_char=100,
                end_char=200,
            ),
        ]

        generator.embed_chunks(chunks)

        assert chunks[0].embedding is not None
        assert len(chunks[0].embedding) == 384
        assert chunks[1].embedding is not None


class TestVectorStore:
    """Test cases for VectorStore."""

    @pytest.fixture
    def temp_chroma_dir(self):
        """Create temporary directory for ChromaDB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def vector_store(self, temp_chroma_dir):
        """Create a VectorStore instance."""
        with patch("storage.vector_store.settings") as mock_settings:
            mock_settings.VECTOR_DB_PATH = temp_chroma_dir
            mock_settings.EMBEDDING_DIMENSION = 384
            store = VectorStore(collection_name="test_collection")
            try:
                yield store
            finally:
                # Release SQLite handles before tempfile cleanup (Windows).
                if getattr(store, "client", None) is not None:
                    try:
                        store.client.close()
                    except Exception:
                        pass
                store.collection = None
                store.client = None

    def test_initialization(self, vector_store):
        """Test that vector store initializes correctly."""
        assert vector_store.collection_name == "test_collection"
        assert vector_store.client is not None
        assert vector_store.collection is not None

    def test_add_chunks(self, vector_store):
        """Test adding chunks to vector store."""
        chunks = [
            DocumentChunk(
                id="chunk-1",
                document_id="doc-1",
                text="Test chunk 1",
                chunk_index=0,
                start_char=0,
                end_char=100,
                embedding=[0.1] * 384,
            ),
            DocumentChunk(
                id="chunk-2",
                document_id="doc-1",
                text="Test chunk 2",
                chunk_index=1,
                start_char=100,
                end_char=200,
                embedding=[0.2] * 384,
            ),
        ]

        vector_store.add_chunks(chunks)

        assert vector_store.get_chunk_count() == 2

    def test_add_chunks_without_embeddings(self, vector_store):
        """Test that chunks without embeddings are skipped."""
        chunks = [
            DocumentChunk(
                id="chunk-1",
                document_id="doc-1",
                text="Test chunk 1",
                chunk_index=0,
                start_char=0,
                end_char=100,
                embedding=None,
            ),
        ]

        vector_store.add_chunks(chunks)

        assert vector_store.get_chunk_count() == 0

    def test_add_empty_chunks(self, vector_store):
        """Test adding empty chunks list."""
        vector_store.add_chunks([])
        assert vector_store.get_chunk_count() == 0

    def test_search(self, vector_store):
        """Test searching for similar chunks."""
        # Add test chunks
        chunks = [
            DocumentChunk(
                id="chunk-1",
                document_id="doc-1",
                text="Apple is a fruit",
                chunk_index=0,
                start_char=0,
                end_char=100,
                embedding=[1.0] + [0.0] * 383,
            ),
            DocumentChunk(
                id="chunk-2",
                document_id="doc-1",
                text="Banana is also a fruit",
                chunk_index=1,
                start_char=100,
                end_char=200,
                embedding=[0.9] + [0.0] * 383,
            ),
            DocumentChunk(
                id="chunk-3",
                document_id="doc-1",
                text="Car is a vehicle",
                chunk_index=2,
                start_char=200,
                end_char=300,
                embedding=[0.0] + [1.0] + [0.0] * 382,
            ),
        ]
        vector_store.add_chunks(chunks)

        # Search with embedding similar to first chunk
        query_embedding = [0.95] + [0.0] * 383
        results = vector_store.search(query_embedding, top_k=2)

        assert len(results) == 2
        # First result should be most similar
        assert results[0][0] == "chunk-1"  # chunk_id

    def test_search_no_results(self, vector_store):
        """Test searching with empty store."""
        query_embedding = [0.1] * 384
        results = vector_store.search(query_embedding, top_k=5)

        assert len(results) == 0

    def test_search_by_document_id(self, vector_store):
        """Test searching with document filter."""
        # Add chunks from different documents
        chunks = [
            DocumentChunk(
                id="chunk-1",
                document_id="doc-1",
                text="Document 1 chunk",
                chunk_index=0,
                start_char=0,
                end_char=100,
                embedding=[0.5] * 384,
            ),
            DocumentChunk(
                id="chunk-2",
                document_id="doc-2",
                text="Document 2 chunk",
                chunk_index=0,
                start_char=0,
                end_char=100,
                embedding=[0.5] * 384,
            ),
        ]
        vector_store.add_chunks(chunks)

        # Search only in doc-1
        query_embedding = [0.5] * 384
        results = vector_store.search(query_embedding, top_k=5, document_ids=["doc-1"])

        assert len(results) == 1
        assert results[0][0] == "chunk-1"

    def test_search_filters_below_similarity_threshold(self, vector_store):
        """Chunks with score = 1 - distance below SIMILARITY_THRESHOLD are dropped."""
        fake = {
            "ids": [["keep", "drop"]],
            "distances": [[0.15, 0.75]],
            "metadatas": [[{"document_id": "d"}, {"document_id": "d"}]],
            "documents": [["good text", "weak text"]],
        }
        with patch("storage.vector_store.settings", SimpleNamespace(SIMILARITY_THRESHOLD=0.7)):
            with patch.object(vector_store.collection, "query", return_value=fake):
                results = vector_store.search([0.1] * 384, top_k=5)
        assert len(results) == 1
        assert results[0][0] == "keep"
        assert results[0][1] == pytest.approx(0.85)
        assert results[0][2] == "good text"

    def test_delete_by_document_id(self, vector_store):
        """Test deleting chunks by document ID."""
        # Add chunks
        chunks = [
            DocumentChunk(
                id="chunk-1",
                document_id="doc-1",
                text="Chunk 1",
                chunk_index=0,
                start_char=0,
                end_char=100,
                embedding=[0.1] * 384,
            ),
            DocumentChunk(
                id="chunk-2",
                document_id="doc-1",
                text="Chunk 2",
                chunk_index=1,
                start_char=100,
                end_char=200,
                embedding=[0.2] * 384,
            ),
            DocumentChunk(
                id="chunk-3",
                document_id="doc-2",
                text="Chunk 3",
                chunk_index=0,
                start_char=0,
                end_char=100,
                embedding=[0.3] * 384,
            ),
        ]
        vector_store.add_chunks(chunks)
        assert vector_store.get_chunk_count() == 3

        # Delete doc-1 chunks
        deleted_count = vector_store.delete_by_document_id("doc-1")

        assert deleted_count == 2
        assert vector_store.get_chunk_count() == 1

    def test_delete_nonexistent_document(self, vector_store):
        """Test deleting non-existent document returns 0."""
        deleted_count = vector_store.delete_by_document_id("nonexistent")
        assert deleted_count == 0

    def test_get_stats(self, vector_store):
        """Test getting vector store statistics."""
        chunks = [
            DocumentChunk(
                id="chunk-1",
                document_id="doc-1",
                text="Test chunk",
                chunk_index=0,
                start_char=0,
                end_char=100,
                embedding=[0.1] * 384,
            ),
        ]
        vector_store.add_chunks(chunks)

        stats = vector_store.get_stats()

        assert stats["collection_name"] == "test_collection"
        assert stats["total_chunks"] == 1
