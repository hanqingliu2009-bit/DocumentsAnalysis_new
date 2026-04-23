"""FastAPI application entry point."""
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from storage.document_store import DocumentStore
from storage.vector_store import VectorStore


# Global store instances
document_store: DocumentStore = None
vector_store: VectorStore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - startup and shutdown."""
    global document_store, vector_store

    # Startup
    print("Starting up Document Q&A API...")
    document_store = DocumentStore()
    vector_store = VectorStore()
    print(f"Loaded {document_store.count()} documents")
    print(f"Vector store contains {vector_store.get_chunk_count()} chunks")

    # Warm up embedding so first chat/upload does not appear to "hang" with no server log.
    try:
        from storage.vector_store import EmbeddingGenerator

        backend = (settings.EMBEDDING_BACKEND or "volcengine").strip().lower()
        if backend == "local":
            if settings.TRANSFORMERS_OFFLINE:
                print(
                    "Loading local embedding model（离线模式：从本地缓存加载，不访问 Hugging Face Hub）..."
                )
            else:
                print(
                    "Loading local embedding model (first run may download from Hugging Face; can take several minutes)..."
                )
        else:
            print(
                f"Warming up Ark embeddings (backend={settings.EMBEDDING_BACKEND}, model={settings.EMBEDDING_MODEL or 'unset'})..."
            )
        EmbeddingGenerator().embed_text("warmup")
        print("Embedding backend ready.")
    except Exception as e:
        print(f"Warning: embedding warmup failed (first request will retry): {e}")

    yield

    # Shutdown
    print("Shutting down Document Q&A API...")


# Create FastAPI app
app = FastAPI(
    title="Document Q&A API",
    description="API for document ingestion, indexing, and question answering",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger = logging.getLogger("uvicorn.error")


# Exception handler (must not swallow HTTPException or hide 4xx/5xx details)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log unexpected errors; pass through HTTPException with correct status and body."""
    if isinstance(exc, HTTPException):
        if exc.status_code >= 500:
            logger.error("HTTP %s %s: %s", request.method, request.url.path, exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    logger.error(
        "Unhandled %s %s: %s\n%s",
        request.method,
        request.url.path,
        exc,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "documents": document_store.count() if document_store else 0,
        "chunks": vector_store.get_chunk_count() if vector_store else 0,
    }


# Import and include API routers
from api.documents import router as documents_router
from api.query import router as query_router
from api.system import router as system_router

app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
app.include_router(query_router, prefix="/api", tags=["query"])
app.include_router(system_router, prefix="/api", tags=["system"])

# Mount static files for frontend (if built)
frontend_build_dir = Path(__file__).parent / "static"
if frontend_build_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_build_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
