"""Query and chat API routes."""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import settings
from core.rag import RAGPipeline

import main

logger = logging.getLogger(__name__)

router = APIRouter()

_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Lazily build RAG pipeline so it sees vector_store after app lifespan runs."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(
            vector_store=main.vector_store,
            document_store=main.document_store,
        )
    return _rag_pipeline


class QueryRequest(BaseModel):
    """Request model for single question query."""
    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document IDs to search within"
    )


class SourceCitation(BaseModel):
    """Source citation model."""
    chunk_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    """Response model for query."""
    answer: str
    sources: List[SourceCitation]
    confidence: float
    context_used: int


class SearchRequest(BaseModel):
    """Request model for semantic search (no LLM)."""
    query: str = Field(..., min_length=1, description="The search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document IDs to search within"
    )


class SearchResult(BaseModel):
    """Search result item."""
    chunk_id: str
    score: float
    text: str


class SearchResponse(BaseModel):
    """Response model for search."""
    results: List[SearchResult]
    total: int


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    """Request model for chat with history."""
    message: str = Field(..., min_length=1, description="The user's message")
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional chat history"
    )
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document IDs to search within"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class ChatResponse(BaseModel):
    """Response model for chat."""
    message: str
    sources: List[SourceCitation]
    confidence: float


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Execute a RAG query to answer a question.

    This endpoint retrieves relevant document chunks and generates
    an answer using the configured OpenAI-compatible LLM.
    """
    try:
        result = get_rag_pipeline().query(
            question=request.question,
            document_ids=request.document_ids,
            top_k=request.top_k,
        )

        if result.get("error"):
            logger.error("RAG /query failed: %s", result["error"])
            raise HTTPException(status_code=500, detail=result["error"])

        return QueryResponse(
            answer=result["answer"],
            sources=[SourceCitation(**s) for s in result["sources"]],
            confidence=result["confidence"],
            context_used=result["context_used"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query endpoint error")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform semantic search without LLM generation.

    This endpoint retrieves relevant document chunks based on
    semantic similarity to the query.
    """
    try:
        from storage.vector_store import EmbeddingGenerator

        # Generate query embedding
        embedder = EmbeddingGenerator()
        query_embedding = embedder.embed_text(request.query)

        # Search vector store
        results = main.vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )

        # Format results
        search_results = [
            SearchResult(
                chunk_id=chunk_id,
                score=score,
                text=text[:500] + "..." if len(text) > 500 else text,
            )
            for chunk_id, score, text in results
        ]

        return SearchResponse(
            results=search_results,
            total=len(search_results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with conversation history.

    This endpoint maintains context across multiple messages and
    generates responses using the configured LLM with RAG.
    """
    try:
        # For now, treat chat similarly to query but with history context
        # Build context from history if provided
        history_context = ""
        if request.history:
            # Take last 3 exchanges for context
            recent_history = request.history[-6:]  # 3 exchanges = 6 messages
            for msg in recent_history:
                role = "User" if msg.role == "user" else "Assistant"
                history_context += f"{role}: {msg.content}\n"
            if history_context:
                history_context = "\nPrevious conversation:\n" + history_context

        # Combine history with current question
        full_question = f"{history_context}\nCurrent question: {request.message}"

        result = get_rag_pipeline().query(
            question=full_question,
            document_ids=request.document_ids,
            top_k=request.top_k,
            # Embed only the latest user turn; long history wrecks semantic search.
            retrieval_query=request.message.strip(),
        )

        if result.get("error"):
            logger.error("RAG /chat failed: %s", result["error"])
            raise HTTPException(status_code=500, detail=result["error"])

        return ChatResponse(
            message=result["answer"],
            sources=[SourceCitation(**s) for s in result["sources"]],
            confidence=result["confidence"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
