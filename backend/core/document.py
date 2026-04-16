"""Document models and data classes."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_serializer


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Document metadata extracted during ingestion."""
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    file_size: Optional[int] = None
    # Additional custom metadata
    custom: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Document model representing a single document."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source_path: str
    doc_type: DocumentType = DocumentType.UNKNOWN
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    chunk_count: int = 0
    error_message: Optional[str] = None

    @field_serializer('created_at', 'updated_at')
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat()

    class Config:
        populate_by_name = True


class DocumentChunk(BaseModel):
    """A chunk of a document with its embedding."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    embedding: Optional[List[float]] = None  # Set after embedding generation
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer('created_at')
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat()

    class Config:
        populate_by_name = True


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[Document]
    total: int
    page: int
    per_page: int


class DocumentUploadRequest(BaseModel):
    """Request model for uploading documents."""
    title: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    id: str
    title: str
    status: ProcessingStatus
    message: str = ""
