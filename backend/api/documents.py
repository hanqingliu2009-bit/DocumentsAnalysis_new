"""Document API routes."""
import shutil
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from config import settings
from core.document import (
    Document,
    DocumentMetadata,
    DocumentType,
    DocumentUploadResponse,
    ProcessingStatus,
)
from core.processor import DocumentProcessor, get_document_type
from storage.vector_store import EmbeddingGenerator

import main

router = APIRouter()
processor = DocumentProcessor()


@router.get("/", response_model=dict)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[ProcessingStatus] = None,
    doc_type: Optional[DocumentType] = None,
):
    """List all documents with optional filtering."""
    docs = main.document_store.get_all(skip=skip, limit=limit)

    # Apply filters
    if status:
        docs = [d for d in docs if d.status == status]
    if doc_type:
        docs = [d for d in docs if d.doc_type == doc_type]

    return {
        "documents": [doc.model_dump() for doc in docs],
        "total": main.document_store.count(),
        "skip": skip,
        "limit": limit,
    }


@router.post("/", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    """Upload a new document for processing."""
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check file size
    file_content = await file.read()
    file_size = len(file_content)

    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE / 1024 / 1024}MB",
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Determine document type
    doc_type = get_document_type(file.filename)
    if doc_type == DocumentType.UNKNOWN:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.filename}",
        )

    # Create document
    document = Document(
        title=title or Path(file.filename).stem,
        source_path=file.filename,
        doc_type=doc_type,
        status=ProcessingStatus.PROCESSING,
    )

    # Save to document store
    main.document_store.save(document)

    # Save original file
    file_storage_path = settings.FILE_STORAGE_PATH / f"{document.id}{Path(file.filename).suffix}"
    with open(file_storage_path, "wb") as f:
        f.write(file_content)

    # Process document (synchronous); chunk ids align with saved document id.
    try:
        # Pass same ``document`` so chunk ``document_id`` matches persisted id (see processor).
        document, chunks = processor.process_document(
            file_content=file_content,
            filename=file.filename,
            doc_type=doc_type,
            document=document,
        )

        main.document_store.save(document)

        if document.status != ProcessingStatus.COMPLETED:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing document: {document.error_message or 'processing failed'}",
            )

        # Generate embeddings and add to vector store
        if chunks:
            from storage.vector_store import EmbeddingGenerator

            embedder = EmbeddingGenerator()
            embedder.embed_chunks(chunks)

            # Filter out chunks without embeddings
            chunks_with_embeddings = [c for c in chunks if c.embedding is not None]
            if chunks_with_embeddings:
                main.vector_store.add_chunks(chunks_with_embeddings)

    except HTTPException:
        raise
    except Exception as e:
        document.status = ProcessingStatus.FAILED
        document.error_message = str(e)
        main.document_store.save(document)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}",
        )

    return DocumentUploadResponse(
        id=document.id,
        title=document.title,
        status=document.status,
        message=f"Document processed successfully with {document.chunk_count} chunks",
    )


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    """Get a document by ID."""
    doc = main.document_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc.model_dump()


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str):
    """Delete a document and its associated data."""
    doc = main.document_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from vector store
    main.vector_store.delete_by_document_id(doc_id)

    # Delete original file
    file_storage_path = settings.FILE_STORAGE_PATH / f"{doc_id}{Path(doc.source_path).suffix}"
    if file_storage_path.exists():
        file_storage_path.unlink()

    # Delete from document store
    main.document_store.delete(doc_id)

    return None


@router.get("/{doc_id}/download")
async def download_document(doc_id: str):
    """Download the original document file."""
    from fastapi.responses import FileResponse

    doc = main.document_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_storage_path = settings.FILE_STORAGE_PATH / f"{doc_id}{Path(doc.source_path).suffix}"
    if not file_storage_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_storage_path,
        filename=doc.source_path,
        media_type="application/octet-stream",
    )
