"""Vector store implementation using ChromaDB."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from core.document import DocumentChunk


class VectorStore:
    """ChromaDB vector store for document chunks."""

    def __init__(self, collection_name: str = "document_chunks"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        # PersistentClient is the supported API; legacy Settings(chroma_db_impl=...) is rejected.
        chroma_settings = ChromaSettings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(
            path=str(settings.VECTOR_DB_PATH),
            settings=chroma_settings,
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks with their embeddings to the store."""
        if not chunks:
            return

        # Prepare data for batch insertion
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            if chunk.embedding is None:
                continue  # Skip chunks without embeddings

            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append({
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            })

        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Search for similar chunks using embedding.

        Results must meet ``settings.SIMILARITY_THRESHOLD`` (cosine similarity
        derived as ``1 - distance``). We over-fetch from Chroma so enough items
        remain after filtering.

        Returns:
            List of (chunk_id, score, text) tuples, sorted by relevance.
        """
        # Build where clause if document_ids specified
        where_clause = None
        if document_ids:
            if len(document_ids) == 1:
                where_clause = {"document_id": document_ids[0]}
            else:
                # ChromaDB doesn't support "in" directly, so we do multiple queries
                # For simplicity, we'll filter after retrieval
                pass

        # Chroma rejects np.float32 scalars in nested lists
        q = np.asarray(query_embedding, dtype=np.float64).tolist()

        threshold = float(settings.SIMILARITY_THRESHOLD)
        # Over-fetch so we can return up to top_k above threshold (multi-doc filter drops rows).
        if document_ids and len(document_ids) != 1:
            n_results = min(200, max(top_k * 15, 40))
        else:
            n_results = min(100, max(top_k * 8, 20))

        # Perform search
        results = self.collection.query(
            query_embeddings=[q],
            n_results=n_results,
            where=where_clause,
            include=["metadatas", "documents", "distances"],
        )

        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # Calculate similarity score (convert distance to similarity)
                distance = results["distances"][0][i]
                score = 1 - distance  # Cosine similarity from distance
                if score < threshold:
                    continue

                # Filter by document_ids if specified
                if document_ids:
                    metadata = results["metadatas"][0][i]
                    if metadata.get("document_id") not in document_ids:
                        continue

                text = results["documents"][0][i]
                formatted_results.append((chunk_id, score, text))

                if len(formatted_results) >= top_k:
                    break

        return formatted_results

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Returns count deleted."""
        # Find all chunks with this document_id
        results = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])

        return 0

    def get_chunk_count(self) -> int:
        """Get total number of chunks in the store."""
        return self.collection.count()

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "persist_directory": str(settings.VECTOR_DB_PATH),
        }


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        model = self._load_model()
        # Avoid convert_to_list=True: newer sentence-transformers reject unknown kwargs.
        raw = model.encode(texts)
        # Chroma expects nested lists of Python float, not np.float32 / numpy rows.
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 1:
            return [arr.tolist()]
        return arr.tolist()

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        result = self.embed_texts([text])
        return result[0] if result else []

    def embed_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Generate and assign embeddings to document chunks in-place."""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return settings.EMBEDDING_DIMENSION
