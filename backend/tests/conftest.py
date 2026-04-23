"""Pytest configuration and fixtures."""
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from config import Settings
from core.document import Document, DocumentChunk, DocumentMetadata, DocumentType, ProcessingStatus
from main import app
from storage.document_store import DocumentStore
from storage.vector_store import VectorStore


# Test fixtures


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_settings(temp_dir):
    """Create test settings with temporary directories."""
    return Settings(
        DATA_DIR=temp_dir,
        VECTOR_DB_PATH=temp_dir / "vector_db",
        DOCUMENT_STORE_PATH=temp_dir / "documents",
        FILE_STORAGE_PATH=temp_dir / "files",
        VOLCENGINE_API_KEY="test-api-key",
        VOLCENGINE_BASE_URL="https://ark.cn-beijing.volces.com/api/v3",
        DEBUG=True,
        EMBEDDING_BACKEND="local",
        EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        EMBEDDING_DIMENSION=384,
        CHROMADB_COLLECTION="test_chroma_collection",
    )


@pytest.fixture
def document_store(temp_dir, test_settings):
    """Create a document store with temporary storage."""
    with patch("config.settings", test_settings):
        store = DocumentStore(storage_path=test_settings.DOCUMENT_STORE_PATH)
        yield store


@pytest.fixture
def vector_store(temp_dir, test_settings):
    """Create a vector store with temporary storage."""
    with patch("config.settings", test_settings):
        store = VectorStore(collection_name="test_chunks")
        yield store


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id=str(uuid4()),
        title="Test Document",
        source_path="/path/to/test.pdf",
        doc_type=DocumentType.PDF,
        status=ProcessingStatus.COMPLETED,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        chunk_count=5,
        metadata=DocumentMetadata(
            title="Test Document",
            author="Test Author",
            page_count=10,
            word_count=500,
        ),
    )


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    doc_id = str(uuid4())
    return [
        DocumentChunk(
            id=str(uuid4()),
            document_id=doc_id,
            text=f"This is test chunk {i} with some content.",
            chunk_index=i,
            start_char=i * 100,
            end_char=(i + 1) * 100,
            embedding=None,
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_pdf_bytes():
    """Create a minimal valid PDF file in memory."""
    # This is a minimal valid PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF Content) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n

trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
298
%%EOF
"""
    return pdf_content


@pytest.fixture
def sample_docx_bytes():
    """Create a minimal valid DOCX file in memory (ZIP archive with XML)."""
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # [Content_Types].xml
        zf.writestr('[Content_Types].xml', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>''')

        # _rels/.rels
        zf.writestr('_rels/.rels', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>''')

        # word/_rels/document.xml.rels
        zf.writestr('word/_rels/document.xml.rels', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
</Relationships>''')

        # word/document.xml
        zf.writestr('word/document.xml', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr>
        <w:pStyle w:val="Normal"/>
      </w:pPr>
      <w:r>
        <w:t>This is a test paragraph for DOCX processing.</w:t>
      </w:r>
    </w:p>
    <w:p>
      <w:pPr>
        <w:pStyle w:val="Normal"/>
      </w:pPr>
      <w:r>
        <w:t>Another paragraph with more test content.</w:t>
      </w:r>
    </w:p>
  </w:body>
</w:document>''')

    return buffer.getvalue()


@pytest.fixture
def mock_volcengine_client():
    """Create a mock OpenAI-compatible LLM client (e.g. Volcengine Ark)."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "This is a test answer from the LLM."
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_embedding_model():
    """Create a mock sentence transformer model."""
    with patch("sentence_transformers.SentenceTransformer") as mock_model_class:
        mock_model = MagicMock()
        # Return 384-dimensional embeddings
        mock_model.encode.return_value = [[0.1] * 384]
        mock_model_class.return_value = mock_model
        yield mock_model


@pytest.fixture
def test_client(test_settings):
    """Create a FastAPI test client.

    Patch DocumentStore / VectorStore constructors so lifespan does not replace
    mocks with real stores (routes use ``main.document_store`` / ``main.vector_store``).
    """
    mock_doc_store = MagicMock()
    mock_doc_store.get_all.return_value = []
    mock_doc_store.count.return_value = 0
    mock_doc_store.count_by_status.return_value = {}
    mock_doc_store.get.return_value = None
    mock_doc_store.save = MagicMock()
    mock_doc_store.delete = MagicMock()

    mock_vec_store = MagicMock()
    mock_vec_store.get_chunk_count.return_value = 0
    mock_vec_store.search.return_value = []
    mock_vec_store.add_chunks = MagicMock()
    mock_vec_store.delete_by_document_id.return_value = 0

    with patch("config.settings", test_settings):
        with patch("main.DocumentStore", return_value=mock_doc_store):
            with patch("main.VectorStore", return_value=mock_vec_store):
                # Starlette only runs ASGI lifespan when TestClient is used as a context
                # manager; otherwise main.document_store stays None on every request.
                with patch("storage.vector_store.EmbeddingGenerator") as mock_embed_cls:
                    mock_embed_cls.return_value.embed_text.return_value = [0.0] * 384
                    with TestClient(app) as client:
                        yield client


# Factory fixtures for generating test data


@pytest.fixture
def document_factory():
    """Factory for creating test documents."""
    def factory(**kwargs):
        defaults = {
            "id": str(uuid4()),
            "title": "Test Document",
            "source_path": "/path/to/test.pdf",
            "doc_type": DocumentType.PDF,
            "status": ProcessingStatus.COMPLETED,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "chunk_count": 5,
            "metadata": DocumentMetadata(),
        }
        defaults.update(kwargs)
        return Document(**defaults)

    return factory


@pytest.fixture
def chunk_factory():
    """Factory for creating test chunks."""
    def factory(document_id: str = None, **kwargs):
        defaults = {
            "id": str(uuid4()),
            "document_id": document_id or str(uuid4()),
            "text": "This is a test chunk.",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 100,
            "embedding": None,
        }
        defaults.update(kwargs)
        return DocumentChunk(**defaults)

    return factory
