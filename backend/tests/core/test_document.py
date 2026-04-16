"""Tests for document models."""
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from core.document import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    ProcessingStatus,
)
from core.processor import get_document_type


class TestDocumentType:
    """Test cases for DocumentType enum."""

    def test_valid_types(self):
        """Test that valid document types are recognized."""
        assert DocumentType.PDF == "pdf"
        assert DocumentType.DOCX == "docx"
        assert DocumentType.TXT == "txt"
        assert DocumentType.MD == "md"
        assert DocumentType.UNKNOWN == "unknown"


class TestProcessingStatus:
    """Test cases for ProcessingStatus enum."""

    def test_valid_statuses(self):
        """Test that valid processing statuses are recognized."""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.PROCESSING == "processing"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"


class TestDocumentMetadata:
    """Test cases for DocumentMetadata model."""

    def test_default_creation(self):
        """Test creating metadata with default values."""
        metadata = DocumentMetadata()
        assert metadata.author is None
        assert metadata.title is None
        assert metadata.word_count is None
        assert metadata.custom == {}

    def test_custom_values(self):
        """Test creating metadata with custom values."""
        metadata = DocumentMetadata(
            author="John Doe",
            title="Test Document",
            page_count=10,
            word_count=500,
            custom={"key": "value"},
        )
        assert metadata.author == "John Doe"
        assert metadata.title == "Test Document"
        assert metadata.page_count == 10
        assert metadata.word_count == 500
        assert metadata.custom == {"key": "value"}


class TestDocumentChunk:
    """Test cases for DocumentChunk model."""

    def test_default_creation(self):
        """Test creating a chunk with default values."""
        doc_id = str(uuid4())
        chunk = DocumentChunk(
            document_id=doc_id,
            text="Test chunk content",
            chunk_index=0,
            start_char=0,
            end_char=100,
        )
        assert chunk.document_id == doc_id
        assert chunk.text == "Test chunk content"
        assert chunk.chunk_index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 100
        assert chunk.embedding is None
        assert chunk.metadata == {}
        assert isinstance(chunk.id, str)

    def test_with_embedding(self):
        """Test creating a chunk with embedding."""
        chunk = DocumentChunk(
            document_id=str(uuid4()),
            text="Test",
            chunk_index=0,
            start_char=0,
            end_char=10,
            embedding=[0.1] * 384,
        )
        assert len(chunk.embedding) == 384
        assert chunk.embedding[0] == 0.1

    def test_serialization(self):
        """Test that chunks can be serialized to dict."""
        chunk = DocumentChunk(
            document_id=str(uuid4()),
            text="Test",
            chunk_index=0,
            start_char=0,
            end_char=10,
        )
        data = chunk.model_dump()
        assert "id" in data
        assert "text" in data
        assert data["text"] == "Test"


class TestDocument:
    """Test cases for Document model."""

    def test_default_creation(self):
        """Test creating a document with default values."""
        doc = Document(
            title="Test Document",
            source_path="/path/to/test.pdf",
            doc_type=DocumentType.PDF,
        )
        assert doc.title == "Test Document"
        assert doc.source_path == "/path/to/test.pdf"
        assert doc.doc_type == DocumentType.PDF
        assert doc.status == ProcessingStatus.PENDING
        assert doc.chunk_count == 0
        assert isinstance(doc.id, str)
        assert isinstance(doc.metadata, DocumentMetadata)

    def test_custom_creation(self):
        """Test creating a document with custom values."""
        doc = Document(
            title="Custom Title",
            source_path="/path/to/doc.docx",
            doc_type=DocumentType.DOCX,
            status=ProcessingStatus.COMPLETED,
            chunk_count=10,
            metadata=DocumentMetadata(
                author="Jane Smith",
                word_count=1000,
            ),
        )
        assert doc.doc_type == DocumentType.DOCX
        assert doc.status == ProcessingStatus.COMPLETED
        assert doc.chunk_count == 10
        assert doc.metadata.author == "Jane Smith"
        assert doc.metadata.word_count == 1000

    def test_serialization(self):
        """Test that documents can be serialized to dict."""
        doc = Document(
            title="Test",
            source_path="/test.pdf",
            doc_type=DocumentType.PDF,
        )
        data = doc.model_dump()
        assert "id" in data
        assert "title" in data
        assert data["title"] == "Test"
        assert data["doc_type"] == "pdf"

    def test_deserialization(self):
        """Test that documents can be deserialized from dict."""
        data = {
            "id": "test-id",
            "title": "Test Document",
            "source_path": "/test.pdf",
            "doc_type": "pdf",
            "status": "completed",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "chunk_count": 5,
        }
        doc = Document.model_validate(data)
        assert doc.id == "test-id"
        assert doc.title == "Test Document"
        assert doc.status == ProcessingStatus.COMPLETED


class TestGetDocumentType:
    """Test cases for get_document_type function."""

    def test_pdf_types(self):
        """Test PDF file extensions."""
        assert get_document_type("test.pdf") == DocumentType.PDF
        assert get_document_type("TEST.PDF") == DocumentType.PDF
        assert get_document_type("document.Pdf") == DocumentType.PDF

    def test_docx_types(self):
        """Test DOCX file extensions."""
        assert get_document_type("test.docx") == DocumentType.DOCX
        assert get_document_type("TEST.DOCX") == DocumentType.DOCX

    def test_txt_types(self):
        """Test TXT file extensions."""
        assert get_document_type("test.txt") == DocumentType.TXT
        assert get_document_type("TEST.TXT") == DocumentType.TXT

    def test_md_types(self):
        """Test Markdown file extensions."""
        assert get_document_type("test.md") == DocumentType.MD
        assert get_document_type("test.markdown") == DocumentType.MD

    def test_unknown_types(self):
        """Test unknown file extensions."""
        assert get_document_type("test.xyz") == DocumentType.UNKNOWN
        assert get_document_type("test") == DocumentType.UNKNOWN
        assert get_document_type("") == DocumentType.UNKNOWN
