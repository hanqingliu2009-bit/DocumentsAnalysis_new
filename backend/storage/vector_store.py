"""Vector store implementation using ChromaDB."""
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import openai
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from core.document import DocumentChunk


class VectorStore:
    """ChromaDB vector store for document chunks."""

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or settings.CHROMADB_COLLECTION
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
    ) -> List[Tuple[str, float, str, str]]:
        """
        Search for similar chunks using embedding.

        Results must meet ``settings.SIMILARITY_THRESHOLD`` (cosine similarity
        derived as ``1 - distance``). We over-fetch from Chroma so enough items
        remain after filtering.

        Returns:
            List of (chunk_id, score, text, document_id) tuples, sorted by relevance.
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

                meta_row = results["metadatas"][0][i] if results.get("metadatas") else {}
                if not isinstance(meta_row, dict):
                    meta_row = {}
                if document_ids and meta_row.get("document_id") not in document_ids:
                    continue

                text = results["documents"][0][i]
                doc_id = str(meta_row.get("document_id") or "")
                formatted_results.append((chunk_id, score, text, doc_id))

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


def _ark_embedding_data_rows(body: Any) -> List[Dict[str, Any]]:
    """
    Normalize Ark embedding responses where ``data`` may be a list of objects or a single object.
    Iterating a dict yields string keys, which caused ``'str' object has no attribute 'get'``.
    """
    if not isinstance(body, dict):
        return []
    raw = body.get("data")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        if "embedding" in raw:
            return [raw]
        inner = raw.get("data")
        if isinstance(inner, list):
            return [x for x in inner if isinstance(x, dict)]
    return []


def _multimodal_input_parts_from_chunk(chunk: DocumentChunk) -> List[Dict[str, Any]]:
    """
    Build Ark ``/embeddings/multimodal`` ``input`` list: text block plus optional images.

    Public image URLs must be fetchable by Ark (localhost 一般不可用). Set on chunk metadata, e.g.::

        embedding_image_urls: ["https://.../a.png"]
        # or legacy key:
        image_urls: ["https://.../a.png"]
    """
    parts: List[Dict[str, Any]] = [{"type": "text", "text": chunk.text or ""}]
    md = chunk.metadata or {}
    raw = md.get("embedding_image_urls")
    if raw is None:
        raw = md.get("image_urls")
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, list):
        for item in raw:
            url: Optional[str] = None
            if isinstance(item, str):
                url = item.strip() or None
            elif isinstance(item, dict):
                url = (item.get("url") or "").strip() or None
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    def _load_model(self):
        """Lazy load the local sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            mid = (self.model_name or "").strip() or "sentence-transformers/all-MiniLM-L6-v2"
            self._model = SentenceTransformer(mid)
        return self._model

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        model = self._load_model()
        raw = model.encode(texts)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 1:
            return [arr.tolist()]
        return arr.tolist()

    def _volcengine_key_and_model(self) -> Tuple[str, str]:
        key = settings.VOLCENGINE_EMBEDDING_API_KEY or settings.VOLCENGINE_API_KEY
        model = (self.model_name or "").strip()
        if not key or not str(key).strip():
            raise ValueError(
                "EMBEDDING_BACKEND=volcengine requires VOLCENGINE_EMBEDDING_API_KEY or VOLCENGINE_API_KEY in backend/.env."
            )
        if not model:
            raise ValueError(
                "EMBEDDING_BACKEND=volcengine requires EMBEDDING_MODEL (embedding endpoint ID, e.g. ep-...)."
            )
        return str(key).strip(), model

    def _embed_volcengine_multimodal_one(self, input_parts: List[Dict[str, Any]]) -> List[float]:
        """Single Ark ``POST .../embeddings/multimodal`` call (doubao-embedding-vision 等)."""
        key, model = self._volcengine_key_and_model()
        base = str(settings.VOLCENGINE_BASE_URL).rstrip("/")
        url = f"{base}/embeddings/multimodal"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "input": input_parts}
        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = r.text[:500]
            except Exception:
                pass
            raise ValueError(
                f"Ark multimodal embeddings HTTP {r.status_code}: {detail or str(e)}"
            ) from e
        body = r.json()
        rows = _ark_embedding_data_rows(body)
        if not rows:
            raise ValueError(
                f"Ark multimodal embeddings: empty or unrecognized data in response: {repr(body)[:400]}"
            )
        ordered = sorted(rows, key=lambda d: d.get("index", 0))
        emb = ordered[0].get("embedding")
        if emb is None:
            raise ValueError(f"Ark multimodal embeddings: missing embedding in row: {ordered[0]!r}")
        return [float(x) for x in emb]

    def _embed_volcengine_multimodal_texts(self, texts: List[str]) -> List[List[float]]:
        """One multimodal request per string (text-only block)."""
        return [
            self._embed_volcengine_multimodal_one([{"type": "text", "text": t}])
            for t in texts
        ]

    def _embed_volcengine(self, texts: List[str]) -> List[List[float]]:
        """Call Ark OpenAI-compatible ``/embeddings`` (豆包 doubao-embedding 等)."""
        key, model = self._volcengine_key_and_model()
        client = openai.OpenAI(
            api_key=key,
            base_url=str(settings.VOLCENGINE_BASE_URL).rstrip("/"),
        )
        out: List[List[float]] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)
            for row in ordered:
                out.append([float(x) for x in row.embedding])
        return out

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        backend = (settings.EMBEDDING_BACKEND or "volcengine").strip().lower()
        if backend == "local":
            return self._embed_local(texts)
        if backend in ("volcengine", "ark", "doubao"):
            if settings.EMBEDDING_USE_MULTIMODAL_API:
                return self._embed_volcengine_multimodal_texts(texts)
            return self._embed_volcengine(texts)
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND={settings.EMBEDDING_BACKEND!r}; use 'volcengine' or 'local'."
        )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        result = self.embed_texts([text])
        return result[0] if result else []

    def embed_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Generate and assign embeddings to document chunks in-place."""
        if not chunks:
            return

        backend = (settings.EMBEDDING_BACKEND or "volcengine").strip().lower()
        if backend in ("volcengine", "ark", "doubao") and settings.EMBEDDING_USE_MULTIMODAL_API:
            for chunk in chunks:
                parts = _multimodal_input_parts_from_chunk(chunk)
                chunk.embedding = self._embed_volcengine_multimodal_one(parts)
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return settings.EMBEDDING_DIMENSION
