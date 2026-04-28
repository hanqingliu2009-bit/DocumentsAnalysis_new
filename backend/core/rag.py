"""RAG (Retrieval-Augmented Generation) pipeline."""
from typing import TYPE_CHECKING, List, Optional

import openai

from config import settings
from core.document import ProcessingStatus
from storage.vector_store import VectorStore

if TYPE_CHECKING:
    from storage.document_store import DocumentStore


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for question answering."""

    def __init__(
        self,
        vector_store: VectorStore,
        openai_client: Optional[openai.OpenAI] = None,
        document_store: Optional["DocumentStore"] = None,
    ):
        self.vector_store = vector_store
        self.document_store = document_store
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

        When retrieval returns no chunks but VOLCENGINE_API_KEY and LLM_MODEL are set,
        the pipeline calls the LLM without document context (no fixed canned replies).

        Returns:
            Dict with answer, sources, and metadata
        """
        try:
            backend = (getattr(settings, "RAG_BACKEND", "chromadb") or "chromadb").strip().lower()
            if backend == "external_graph":
                return self._query_external_graph(question, retrieval_query)

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
                if (
                    settings.VOLCENGINE_API_KEY
                    and str(settings.VOLCENGINE_API_KEY).strip()
                    and str(self.model).strip()
                ):
                    answer = self._generate_answer_without_context(
                        question,
                        system_prompt=self._system_prompt_for_no_retrieval_match(),
                    )
                    return {
                        "answer": answer,
                        "sources": [],
                        "confidence": 0.0,
                        "context_used": 0,
                        "answer_mode": "llm_direct",
                    }
                return {
                    "answer": (
                        "未配置大模型接口：请在 backend/.env 中设置 VOLCENGINE_API_KEY 与 LLM_MODEL 后重启。"
                        "知识库未命中片段时也需要这两项才能生成回答。"
                    ),
                    "sources": [],
                    "confidence": 0.0,
                    "context_used": 0,
                    "answer_mode": "system",
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
                        sum(score for _, score, _, _ in retrieved) / len(retrieved), 4
                    ),
                    "context_used": len(retrieved),
                    "answer_mode": "system",
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
                        sum(score for _, score, _, _ in retrieved) / len(retrieved), 4
                    ),
                    "context_used": len(retrieved),
                    "answer_mode": "system",
                }

            answer = self._generate_answer(question, context)

            # Step 5: Format sources
            sources = self._format_sources(retrieved)

            # Calculate confidence (average of similarity scores)
            confidence = sum(score for _, score, _, _ in retrieved) / len(retrieved)

            return {
                "answer": answer,
                "sources": sources,
                "confidence": round(confidence, 4),
                "context_used": len(retrieved),
                "answer_mode": "knowledge_base",
            }

        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "context_used": 0,
                "error": str(e),
                "answer_mode": "system",
            }

    def _query_external_graph(
        self,
        question: str,
        retrieval_query: Optional[str],
    ) -> dict:
        """Retrieve context from supplier graph HTTP API, then answer with the LLM."""
        from core.external_graph import (
            build_graph_system_prompt_no_hits,
            fetch_graph_context,
        )

        q = (retrieval_query or question).strip() or question.strip()
        try:
            context, sources, _ = fetch_graph_context(q)
        except Exception as e:
            return {
                "answer": f"图数据库请求失败：{str(e)}",
                "sources": [],
                "confidence": 0.0,
                "context_used": 0,
                "error": str(e),
                "answer_mode": "system",
            }

        if not context.strip():
            if (
                settings.VOLCENGINE_API_KEY
                and str(settings.VOLCENGINE_API_KEY).strip()
                and str(self.model).strip()
            ):
                answer = self._generate_answer_without_context(
                    question,
                    system_prompt=build_graph_system_prompt_no_hits(),
                )
                return {
                    "answer": answer,
                    "sources": [],
                    "confidence": 0.0,
                    "context_used": 0,
                    "answer_mode": "llm_direct",
                }
            return {
                "answer": (
                    "图数据库未返回三元组或摘要；未配置大模型时无法生成补充回答。"
                    "请在 backend/.env 中设置 VOLCENGINE_API_KEY 与 LLM_MODEL 后重启。"
                ),
                "sources": [],
                "confidence": 0.0,
                "context_used": 0,
                "answer_mode": "system",
            }

        if not settings.VOLCENGINE_API_KEY or not str(settings.VOLCENGINE_API_KEY).strip():
            return {
                "answer": (
                    "已从图数据库取回上下文，但未配置 VOLCENGINE_API_KEY，无法调用大模型。"
                    "请在 backend/.env 中设置后重启服务。"
                ),
                "sources": sources,
                "confidence": 1.0,
                "context_used": len(sources),
                "answer_mode": "system",
            }

        if not str(self.model).strip():
            return {
                "answer": (
                    "已从图数据库取回上下文，但未配置 LLM_MODEL。"
                    "请在 backend/.env 中设置接入点或模型名后重启。"
                ),
                "sources": sources,
                "confidence": 1.0,
                "context_used": len(sources),
                "answer_mode": "system",
            }

        answer = self._generate_answer_graph(question, context)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": 1.0,
            "context_used": len(sources),
            "answer_mode": "external_graph",
        }

    def _generate_answer_graph(self, question: str, context: str) -> str:
        """LLM answer when context comes from the external knowledge graph."""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是技术支持助手。请仅根据用户消息中提供的「图数据库检索结果」作答；"
                    "其中可能包含三元组（实体-关系-实体）、摘要与文本块。"
                    "若这些信息不足以回答，须明确说明信息不足，不要编造未出现的事实。"
                    "引用时可概括性说明来自图数据库检索结果。"
                    "回答语言必须与用户问题一致。"
                ),
            },
            {
                "role": "user",
                "content": f"图数据库检索结果:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def _document_display_name(self, document_id: str) -> str:
        """Human-readable document label for context and citations."""
        if not document_id:
            return ""
        if self.document_store is None:
            return document_id
        doc = self.document_store.get(document_id)
        if doc is None:
            return document_id
        title = (doc.title or "").strip()
        return title or doc.source_path or document_id

    def _build_context(self, retrieved: List[tuple]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []

        for i, (chunk_id, score, text, doc_id) in enumerate(retrieved, 1):
            label = self._document_display_name(doc_id) if doc_id else f"片段{i}"
            context_parts.append(f"[Source {i} — {label}]\n{text}\n")

        return "\n---\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using the configured OpenAI-compatible chat model."""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是技术支持助手。请仅根据用户消息中提供的 Context 作答；"
                    "若 Context 不足以回答，须明确说明信息不足，不要编造 Context 中没有的内容。"
                    "引用时请标明来源编号（例如「根据来源 1…」）。"
                    "回答语言必须与用户问题一致：用户使用中文提问时用中文作答；"
                    "用户使用英文或其他语言时，用与用户相同的语言作答。"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].message.content

    def _system_prompt_for_no_retrieval_match(self) -> str:
        """Instructions when semantic search returned no chunks (avoid false 'not uploaded' claims)."""
        chunk_total = int(self.vector_store.get_chunk_count())
        completed_docs = 0
        if self.document_store is not None:
            by_status = self.document_store.count_by_status()
            completed_docs = int(by_status.get(ProcessingStatus.COMPLETED.value, 0))

        core = (
            "你是通用助手。本轮「向量语义检索」没有返回任何可引用的文档片段，因此不要编造具体条文、页码或文件名；"
            "也不要假装阅读了用户磁盘上的文件内容。"
            "回答语言与用户问题一致：用户用中文提问时用中文作答。"
        )
        if chunk_total == 0:
            return (
                core
                + " 系统侧信息：当前向量索引中的片段数为 0，知识库可能尚未写入内容。"
                "可礼貌建议用户在 Documents 上传文件并确认状态为 completed。"
            )
        return (
            core
            + f" 系统侧信息：向量索引中已有 {chunk_total} 条文本片段；"
            f"Documents 中状态为 completed 的文档约 {completed_docs} 篇。"
            "因此当用户称已上传时，「不要」断言用户「没有可查询文档」或「未完成上传/处理」；"
            "应解释为：本轮提问与已索引正文的语义匹配不足，或未超过系统配置的相似度阈值。"
            "可建议用户换一种更贴近文档原文的提问、包含题干中的关键词，或联系管理员检查 SIMILARITY_THRESHOLD。"
            "若用户只想确认是否上传成功，可提示其查看 Documents 列表与 Dashboard 中的 chunk 统计。"
        )

    def _generate_answer_without_context(
        self, question: str, *, system_prompt: Optional[str] = None
    ) -> str:
        """LLM answer when no document chunks were retrieved (general assistant, no RAG)."""
        system = system_prompt or self._system_prompt_for_no_retrieval_match()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
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
        for chunk_id, score, text, doc_id in retrieved:
            title = self._document_display_name(doc_id) if doc_id else ""
            sources.append({
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "text": text[:500] + "..." if len(text) > 500 else text,
                "document_id": doc_id or None,
                "document_title": title or None,
            })
        return sources
