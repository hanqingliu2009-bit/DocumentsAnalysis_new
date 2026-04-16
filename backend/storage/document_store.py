"""Document metadata storage using JSON files."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from config import settings
from core.document import Document, ProcessingStatus


class DocumentStore:
    """Store and retrieve document metadata."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or settings.DOCUMENT_STORE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._documents: Dict[str, Document] = {}
        self._load_all()

    def _get_file_path(self, doc_id: str) -> Path:
        """Get the file path for a document."""
        return self.storage_path / f"{doc_id}.json"

    def _load_all(self) -> None:
        """Load all documents from storage."""
        self._documents = {}
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    doc = Document.model_validate(data)
                    self._documents[doc.id] = doc
            except Exception as e:
                print(f"Error loading document from {file_path}: {e}")

    def save(self, document: Document) -> None:
        """Save a document to storage."""
        document.updated_at = datetime.utcnow()
        file_path = self._get_file_path(document.id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(document.model_dump(), f, indent=2, default=str)
        self._documents[document.id] = document

    def get(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._documents.get(doc_id)

    def get_all(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """Get all documents with pagination."""
        docs = list(self._documents.values())
        # Sort by created_at descending
        docs.sort(key=lambda x: x.created_at, reverse=True)
        return docs[skip:skip + limit]

    def delete(self, doc_id: str) -> bool:
        """Delete a document from storage."""
        if doc_id not in self._documents:
            return False

        file_path = self._get_file_path(doc_id)
        if file_path.exists():
            file_path.unlink()

        del self._documents[doc_id]
        return True

    def count(self) -> int:
        """Get total document count."""
        return len(self._documents)

    def count_by_status(self) -> Dict[str, int]:
        """Get document count by processing status."""
        counts = {}
        for status in ProcessingStatus:
            counts[status.value] = 0
        for doc in self._documents.values():
            counts[doc.status.value] = counts.get(doc.status.value, 0) + 1
        return counts
