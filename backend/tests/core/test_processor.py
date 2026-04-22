"""Tests for document processor module."""
import zipfile
import pytest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from core.document import Document, DocumentMetadata, DocumentType, ProcessingStatus
from core.processor import DocumentProcessor, _read_docx_page_count, get_document_type


class TestGetDocumentType:
    """Test cases for get_document_type function."""

    def test_pdf_extension(self):
        """Test PDF file detection."""
        assert get_document_type("test.pdf") == DocumentType.PDF
        assert get_document_type("test.PDF") == DocumentType.PDF
        assert get_document_type("TEST.Pdf") == DocumentType.PDF

    def test_docx_extension(self):
        """Test DOCX file detection."""
        assert get_document_type("test.docx") == DocumentType.DOCX
        assert get_document_type("test.DOCX") == DocumentType.DOCX

    def test_txt_extension(self):
        """Test TXT file detection."""
        assert get_document_type("test.txt") == DocumentType.TXT
        assert get_document_type("test.TXT") == DocumentType.TXT

    def test_md_extension(self):
        """Test Markdown file detection."""
        assert get_document_type("test.md") == DocumentType.MD
        assert get_document_type("test.markdown") == DocumentType.MD

    def test_unknown_extension(self):
        """Test unknown file detection."""
        assert get_document_type("test.xyz") == DocumentType.UNKNOWN
        assert get_document_type("test") == DocumentType.UNKNOWN
        assert get_document_type("") == DocumentType.UNKNOWN


class TestDocumentProcessorInit:
    """Test cases for DocumentProcessor initialization."""

    def test_default_initialization(self):
        """Test processor initializes with default settings."""
        processor = DocumentProcessor()
        assert processor.chunk_size == 512
        assert processor.chunk_overlap == 50

    @patch("core.processor.settings")
    def test_custom_settings(self, mock_settings):
        """Test processor uses custom settings."""
        mock_settings.CHUNK_SIZE = 1024
        mock_settings.CHUNK_OVERLAP = 100

        processor = DocumentProcessor()
        assert processor.chunk_size == 1024
        assert processor.chunk_overlap == 100


class TestDocumentProcessorProcessDocument:
    """Test cases for process_document method."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        return DocumentProcessor()

    @pytest.fixture
    def sample_text_content(self):
        """Create sample text content for testing."""
        return "This is a test document with multiple sentences. " * 50

    def test_process_txt_document(self, processor, sample_text_content):
        """Test processing a TXT document."""
        content = sample_text_content.encode("utf-8")

        with patch.object(processor, "_extract_text", return_value=(sample_text_content, DocumentMetadata())):
            document, chunks = processor.process_document(
                file_content=content,
                filename="test.txt",
                doc_type=DocumentType.TXT,
            )

        assert document.title == "test"
        assert document.doc_type == DocumentType.TXT
        assert document.source_path == "test.txt"
        assert document.status == ProcessingStatus.COMPLETED
        assert document.chunk_count == len(chunks)
        assert len(chunks) > 0

    def test_process_empty_document(self, processor):
        """Test processing an empty document."""
        with patch.object(processor, "_extract_text", return_value=("", DocumentMetadata())):
            document, chunks = processor.process_document(
                file_content=b"",
                filename="empty.txt",
                doc_type=DocumentType.TXT,
            )

        assert document.status == ProcessingStatus.COMPLETED
        assert len(chunks) == 0
        assert document.chunk_count == 0

    def test_process_document_with_error(self, processor):
        """Test handling errors during processing."""
        with patch.object(processor, "_extract_text", side_effect=Exception("Extraction failed")):
            document, chunks = processor.process_document(
                file_content=b"content",
                filename="test.txt",
                doc_type=DocumentType.TXT,
            )

        assert document.status == ProcessingStatus.FAILED
        assert document.error_message == "Extraction failed"
        assert len(chunks) == 0

    def test_process_document_reuses_api_document_id(self, processor):
        """Upload path passes the same Document so chunk document_id matches persisted id."""
        fixed_id = str(uuid4())
        outer = Document(
            id=fixed_id,
            title="api-title",
            source_path="file.txt",
            doc_type=DocumentType.TXT,
            status=ProcessingStatus.PROCESSING,
        )
        body = "First sentence here. " * 30
        with patch.object(processor, "_extract_text", return_value=(body, DocumentMetadata())):
            doc_out, chunks = processor.process_document(
                file_content=body.encode("utf-8"),
                filename="file.txt",
                doc_type=DocumentType.TXT,
                document=outer,
            )
        assert doc_out is outer
        assert doc_out.id == fixed_id
        assert len(chunks) > 0
        assert all(c.document_id == fixed_id for c in chunks)


class TestDocumentProcessorChunking:
    """Test cases for document chunking."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        return DocumentProcessor()

    def test_chunk_short_text(self, processor):
        """Test chunking short text."""
        text = "This is a short text."
        doc_id = "test-doc-id"

        chunks = processor._create_chunks(text, doc_id)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].document_id == doc_id
        assert chunks[0].chunk_index == 0

    def test_chunk_long_text(self, processor):
        """Test chunking long text into multiple chunks."""
        # Create text that will require multiple chunks
        text = ("This is a sentence with many words. " * 50).strip()
        doc_id = "test-doc-id"

        chunks = processor._create_chunks(text, doc_id)

        assert len(chunks) > 1
        # Verify chunks have correct document ID
        for chunk in chunks:
            assert chunk.document_id == doc_id

    def test_chunk_overlap(self, processor):
        """Test that chunks have correct overlap."""
        text = ("This is a sentence. " * 100).strip()
        doc_id = "test-doc-id"

        chunks = processor._create_chunks(text, doc_id)

        # Check that consecutive chunks overlap
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            # End of chunk1 should overlap with start of chunk2
            assert chunk1.end_char > chunk2.start_char

    def test_chunk_indices(self, processor):
        """Test that chunk indices are sequential."""
        text = ("This is a sentence. " * 100).strip()
        doc_id = "test-doc-id"

        chunks = processor._create_chunks(text, doc_id)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestDocumentProcessorExtractText:
    """Test cases for text extraction methods."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        return DocumentProcessor()

    def test_extract_text_utf8(self, processor):
        """Test extracting UTF-8 encoded text."""
        text = "Hello, World! Testing text extraction."
        content = text.encode("utf-8")

        extracted_text, metadata = processor._extract_text(content)

        assert extracted_text == text
        assert metadata.word_count == len(text.split())
        assert metadata.file_size == len(content)

    def test_extract_text_latin1_fallback(self, processor):
        """Test that Latin-1 is used as fallback for invalid UTF-8."""
        text = "Hëllö, Wörld!"
        content = text.encode("latin-1")

        extracted_text, metadata = processor._extract_text(content)

        assert extracted_text == text

    def test_extract_empty_text(self, processor):
        """Test extracting empty text content."""
        content = b""

        extracted_text, metadata = processor._extract_text(content)

        assert extracted_text == ""
        assert metadata.word_count == 0


def _zip_with_app_xml(app_xml: str) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("docProps/app.xml", app_xml.encode("utf-8"))
    return buf.getvalue()


class TestReadDocxPageCount:
    """docProps/app.xml Pages for DOCX metadata.page_count."""

    def test_reads_pages_from_app_xml(self):
        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
            "<Pages>26</Pages><Words>100</Words></Properties>"
        )
        assert _read_docx_page_count(_zip_with_app_xml(xml)) == 26

    def test_missing_pages_returns_none(self):
        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
            "<Words>10</Words></Properties>"
        )
        assert _read_docx_page_count(_zip_with_app_xml(xml)) is None

    def test_missing_app_xml_returns_none(self):
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("word/document.xml", b"<w:document/>")
        assert _read_docx_page_count(buf.getvalue()) is None

    def test_invalid_zip_returns_none(self):
        assert _read_docx_page_count(b"not a zip file") is None


class TestGetDocumentType:
    """Test cases for get_document_type function."""

    def test_pdf_extensions(self):
        """Test PDF extension detection."""
        assert get_document_type("test.pdf") == DocumentType.PDF
        assert get_document_type("test.PDF") == DocumentType.PDF
        assert get_document_type("test.Pdf") == DocumentType.PDF
        assert get_document_type("/path/to/test.pdf") == DocumentType.PDF

    def test_docx_extensions(self):
        """Test DOCX extension detection."""
        assert get_document_type("test.docx") == DocumentType.DOCX
        assert get_document_type("test.DOCX") == DocumentType.DOCX

    def test_txt_extensions(self):
        """Test TXT extension detection."""
        assert get_document_type("test.txt") == DocumentType.TXT
        assert get_document_type("test.TXT") == DocumentType.TXT

    def test_md_extensions(self):
        """Test Markdown extension detection."""
        assert get_document_type("test.md") == DocumentType.MD
        assert get_document_type("test.markdown") == DocumentType.MD

    def test_unknown_extensions(self):
        """Test unknown extension detection."""
        assert get_document_type("test.xyz") == DocumentType.UNKNOWN
        assert get_document_type("test") == DocumentType.UNKNOWN
        assert get_document_type("") == DocumentType.UNKNOWN
        assert get_document_type("test.doc") == DocumentType.UNKNOWN
