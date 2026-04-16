"""System API routes."""
from fastapi import APIRouter

from config import settings

import main

router = APIRouter()


@router.get("/stats")
async def get_stats():
    """Get system statistics."""
    status_counts = main.document_store.count_by_status()

    return {
        "documents": {
            "total": main.document_store.count(),
            "by_status": status_counts,
        },
        "chunks": {
            "total": main.vector_store.get_chunk_count(),
        },
        "storage": {
            "vector_db_path": str(settings.VECTOR_DB_PATH),
            "document_store_path": str(settings.DOCUMENT_STORE_PATH),
        },
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
    }
