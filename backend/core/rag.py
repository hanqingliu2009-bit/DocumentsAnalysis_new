"""RAG (Retrieval-Augmented Generation) pipeline."""
from typing import List, Optional

import openai

from config import settings
from storage.vector_store import VectorStore


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for question answering."""

    def __init__(
        self,
        vector_store: VectorStore,
        openai_client: Optional[openai.OpenAI] = None,
    ):
        self.vector_store = vector_store
        self.client = openai_client or openai.OpenAI(
            api_key=settings.VOLCENGINE_API_KEY,
            base_url=settings.VOLCENGINE_BASE_URL,
        )
        self.model = settings.LLM_MODEL
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        self.top_k = settings.TOP_K_RETRIEVAL

    def query(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        retrieval_query: Optional[str] = None,
    ) -> dict:
        """
        Execute a RAG query: retrieve relevant chunks and generate answer.

        Args:
            question: Full text for the LLM (e.g. chat history + current turn).
            document_ids: Optional list of document IDs to search within
            top_k: Number of chunks to retrieve (default from settings)
            retrieval_query: If set, use this string only for embedding / vector search
                (avoids diluting retrieval when ``question`` includes long chat history).

        Returns:
            Dict with answer, sources, and metadata
        """
        try:
            # Step 1: Embed the search query (short, retrieval-focused)
            from storage.vector_store import EmbeddingGenerator

            embedder = EmbeddingGenerator()
            search_text = (retrieval_query or question).strip() or question.strip()
            query_embedding = embedder.embed_text(search_text)

            # Step 2: Retrieve relevant chunks
            k = top_k or self.top_k
            retrieved = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k,
                document_ids=document_ids,
            )

            if not retrieved:
                return {
                    "answer": self._no_context_answer(search_text, full_prompt=question),
                    "sources": [],
                    "confidence": 0.0,
                    "context_used": 0,
                }

            # Step 3: Build context from retrieved chunks
            context = self._build_context(retrieved)

            # Step 4: Generate answer using LLM (full ``question`` may include chat history)
            if not settings.VOLCENGINE_API_KEY or not str(settings.VOLCENGINE_API_KEY).strip():
                return {
                    "answer": (
                        "已检索到相关文档片段，但未配置 VOLCENGINE_API_KEY，无法调用大模型。"
                        "请在 backend/.env 中设置 VOLCENGINE_API_KEY（火山方舟 API Key）后重启服务。"
                    ),
                    "sources": self._format_sources(retrieved),
                    "confidence": round(
                        sum(score for _, score, _ in retrieved) / len(retrieved), 4
                    ),
                    "context_used": len(retrieved),
                }

            if not str(self.model).strip():
                return {
                    "answer": (
                        "未配置 LLM_MODEL。使用火山方舟时请在 backend/.env 中设置 LLM_MODEL 为控制台里的"
                        "接入点 Endpoint ID（通常以 ep- 开头）或控制台给出的 model 名称。"
                        "保存后重启后端。若已配置仍报 404，说明 ID 与当前 VOLCENGINE_BASE_URL 不匹配或无权访问。"
                    ),
                    "sources": self._format_sources(retrieved),
                    "confidence": round(
                        sum(score for _, score, _ in retrieved) / len(retrieved), 4
                    ),
                    "context_used": len(retrieved),
                }

            answer = self._generate_answer(question, context)

            # Step 5: Format sources
            sources = self._format_sources(retrieved)

            # Calculate confidence (average of similarity scores)
            confidence = sum(score for _, score, _ in retrieved) / len(retrieved)

            return {
                "answer": answer,
                "sources": sources,
                "confidence": round(confidence, 4),
                "context_used": len(retrieved),
            }

        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "error": str(e),
            }

    def _no_context_answer(self, question: str, full_prompt: Optional[str] = None) -> str:
        """User-facing text when retrieval returns nothing (empty index or no semantic match)."""
        # Greeting / small talk: use the short retrieval string so chat history does not hide "hello"
        probe = (question or "").strip()
        haystack = full_prompt if full_prompt else question
        q = probe.lower()
        compact = "".join(q.split())
        if any(
            g in haystack
            for g in ("你好", "您好", "在吗", "早上好", "晚上好", "hi", "hello", "hey")
        ) or q in ("hi", "hello", "hey", "thanks", "thank you") or compact in ("thankyou",):
            return (
                "您好！我是文档问答助手。"
                "（说明：当前没有在向量库中匹配到文档片段，因此这是预设说明，并未调用大模型。）"
                "请先上传并在 Documents 中处理完成的文档，再用与文档内容相关的问题提问。"
            )
        return (
            "在当前已索引的文档中没有找到与这个问题相关的内容。"
            "请确认已在「Documents」中上传且状态为 completed，并尽量用与文档主题相关的问题提问。"
        )

    def _build_context(self, retrieved: List[tuple]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []

        for i, (chunk_id, score, text) in enumerate(retrieved, 1):
            context_parts.append(f"[Source {i}]\n{text}\n")

        return "\n---\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using the configured OpenAI-compatible chat model."""
        messages = [
            {
                "role": "system",
                "content": "You are a technical support assistant. Answer the question based ONLY on the provided context. If the context doesn't contain enough information, say 'I don't have enough information to answer this.' Cite specific sources (e.g., 'According to Source 1...') when providing information."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].message.content

    def _format_sources(self, retrieved: List[tuple]) -> List[dict]:
        """Format retrieved chunks as sources."""
        sources = []
        for chunk_id, score, text in retrieved:
            sources.append({
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "text": text[:500] + "..." if len(text) > 500 else text,
            })
        return sources
