"""RAG (Retrieval-Augmented Generation) pipeline."""
from concurrent.futures import ThreadPoolExecutor
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
            api_key=settings.llm_openai_api_key() or None,
            base_url=settings.llm_openai_base_url(),
        )
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        self.top_k = settings.TOP_K_RETRIEVAL

    @staticmethod
    def _llm_model_id() -> str:
        """Chat model id sent to the OpenAI-compatible API (provider model name or id)."""
        return str(settings.LLM_MODEL or "").strip()

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

        When retrieval returns no chunks but LLM_API_KEY and LLM_MODEL are set,
        the pipeline calls the LLM without document context (no fixed canned replies).

        Returns:
            Dict with answer, sources, and metadata
        """
        try:
            backend = (getattr(settings, "RAG_BACKEND", "chromadb") or "chromadb").strip().lower()
            if backend == "hybrid":
                return self._query_hybrid(question, document_ids, top_k, retrieval_query)
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
                if settings.llm_openai_api_key() and self._llm_model_id():
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
                        "未配置大模型接口：请在 backend/.env 中设置 LLM_API_KEY、LLM_BASE_URL 与 LLM_MODEL 后重启。"
                        "知识库未命中片段时也需要密钥、兼容地址与模型名才能生成回答。"
                    ),
                    "sources": [],
                    "confidence": 0.0,
                    "context_used": 0,
                    "answer_mode": "system",
                }

            # Step 3: Build context from retrieved chunks
            context = self._build_context(retrieved)

            # Step 4: Generate answer using LLM (full ``question`` may include chat history)
            if not settings.llm_openai_api_key():
                return {
                    "answer": (
                        "已检索到相关文档片段，但未配置对话 API Key，无法调用大模型。"
                        "请在 backend/.env 中设置 LLM_API_KEY 后重启服务。"
                    ),
                    "sources": self._format_sources(retrieved),
                    "confidence": round(
                        sum(score for _, score, _, _ in retrieved) / len(retrieved), 4
                    ),
                    "context_used": len(retrieved),
                    "answer_mode": "system",
                }

            if not self._llm_model_id():
                return {
                    "answer": (
                        "未配置 LLM_MODEL。请在 backend/.env 中设置为 OpenAI 兼容接口所需的模型名（以服务商控制台为准）。"
                        "保存后重启后端。若已配置仍报 404，请核对 LLM_BASE_URL 与模型权限。"
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

    def _query_hybrid(
        self,
        question: str,
        document_ids: Optional[List[str]],
        top_k: Optional[int],
        retrieval_query: Optional[str],
    ) -> dict:
        """
        Parallel retrieval: supplier graph HTTP + splite BGE Chroma collection,
        merge contexts, then answer with the LLM.
        """
        from core.external_graph import build_graph_system_prompt_no_hits, fetch_graph_context
        from storage.vector_store import EmbeddingGenerator, VectorStore

        q = (retrieval_query or question).strip() or question.strip()

        def graph_call():
            try:
                ctx, src, _dbg = fetch_graph_context(q)
                return ("ok", ctx, src, None)
            except Exception as e:
                return ("err", "", [], str(e))

        def vector_call():
            try:
                mid = (settings.HYBRID_SPLITE_EMBEDDING_MODEL or "").strip() or "BAAI/bge-large-zh-v1.5"
                embedder = EmbeddingGenerator(model_name=mid)
                emb = embedder.embed_texts([q], embedding_backend="local")[0]
                splite_store = VectorStore(collection_name=settings.HYBRID_SPLITE_COLLECTION)
                vk = int(settings.HYBRID_VECTOR_TOP_K or top_k or self.top_k)
                rows = splite_store.search(
                    query_embedding=emb,
                    top_k=vk,
                    document_ids=document_ids,
                )
                return ("ok", rows, None)
            except Exception as e:
                return ("err", [], str(e))

        with ThreadPoolExecutor(max_workers=2) as pool:
            fg = pool.submit(graph_call)
            fv = pool.submit(vector_call)
            gr = fg.result()
            vr = fv.result()

        graph_context = ""
        graph_sources: List[dict] = []
        graph_err: Optional[str] = None
        if gr[0] == "ok":
            graph_context, graph_sources = gr[1], gr[2]
        else:
            graph_err = gr[3]

        retrieved: List[tuple] = []
        vec_err: Optional[str] = None
        if vr[0] == "ok":
            retrieved = vr[1]
        else:
            vec_err = vr[2]

        merged_context = self._build_hybrid_context(graph_context, retrieved)
        merged_context = self._truncate_hybrid_context(
            merged_context, int(getattr(settings, "HYBRID_CONTEXT_MAX_CHARS", 24000) or 24000)
        )

        sources: List[dict] = []
        for s in graph_sources:
            item = dict(s)
            item["source_channel"] = "graph"
            sources.append(item)
        sources.extend(self._format_sources(retrieved, source_channel="vector"))

        has_graph = bool((graph_context or "").strip())
        has_vec = bool(retrieved)
        if not has_graph and not has_vec:
            detail = []
            if graph_err:
                detail.append(f"图数据库: {graph_err}")
            if vec_err:
                detail.append(f"本地向量: {vec_err}")
            note = "；".join(detail) if detail else "两路均未返回可用内容。"
            if settings.llm_openai_api_key() and self._llm_model_id():
                answer = self._generate_answer_without_context(
                    question,
                    system_prompt=(
                        build_graph_system_prompt_no_hits()
                        + " 本轮混合检索中，" + note
                    ),
                )
                return {
                    "answer": answer,
                    "sources": sources,
                    "confidence": 0.0,
                    "context_used": 0,
                    "answer_mode": "llm_direct",
                }
            return {
                "answer": (
                    "混合检索未命中可用上下文；未配置大模型时无法生成补充回答。"
                    f"（{note}）请在 backend/.env 中设置 LLM_API_KEY 与 LLM_MODEL 后重启。"
                ),
                "sources": sources,
                "confidence": 0.0,
                "context_used": 0,
                "answer_mode": "system",
            }

        if not settings.llm_openai_api_key():
            return {
                "answer": (
                    "混合检索已取回部分上下文，但未配置对话 API Key，无法调用大模型。"
                    "请在 backend/.env 中设置 LLM_API_KEY 后重启服务。"
                ),
                "sources": sources,
                "confidence": self._hybrid_confidence(retrieved, has_graph),
                "context_used": len(sources),
                "answer_mode": "system",
            }

        if not self._llm_model_id():
            return {
                "answer": (
                    "混合检索已取回部分上下文，但未配置 LLM_MODEL。"
                    "请在 backend/.env 中设置接入点或模型名后重启。"
                ),
                "sources": sources,
                "confidence": self._hybrid_confidence(retrieved, has_graph),
                "context_used": len(sources),
                "answer_mode": "system",
            }

        answer = self._generate_answer_hybrid(question, merged_context)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": self._hybrid_confidence(retrieved, has_graph),
            "context_used": len(sources),
            "answer_mode": "hybrid_graph_vector",
        }

    @staticmethod
    def _truncate_hybrid_context(text: str, max_chars: int) -> str:
        text = (text or "").strip()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 24)].rstrip() + "\n…[上下文已截断]"

    def _build_hybrid_context(self, graph_context: str, retrieved: List[tuple]) -> str:
        parts: List[str] = []
        gc = (graph_context or "").strip()
        if gc:
            parts.append("### 图数据库检索结果\n" + gc)
        if retrieved:
            vc = self._build_context(retrieved)
            parts.append("### 本地手册向量检索\n" + vc)
        return "\n\n".join(parts).strip()

    @staticmethod
    def _hybrid_confidence(retrieved: List[tuple], has_graph: bool) -> float:
        if retrieved:
            return round(sum(score for _, score, _, _ in retrieved) / len(retrieved), 4)
        if has_graph:
            return 1.0
        return 0.0

    def _generate_answer_hybrid(self, question: str, context: str) -> str:
        """LLM answer when context comes from graph + local vector retrieval."""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是技术支持助手。用户消息中包含两路检索结果："
                    "「### 图数据库检索结果」来自供应方图数据库（可能含三元组、摘要、文本块）；"
                    "「### 本地手册向量检索」来自本地维护手册向量索引。"
                    "请综合两路信息作答；若仅一路有内容则主要依据该路。"
                    "若仍不足以回答，须明确说明信息不足，不要编造未出现的事实。"
                    "引用时可说明来自图数据库或本地手册检索。"
                    "回答语言必须与用户问题一致。"
                ),
            },
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ]
        response = self.client.chat.completions.create(
            model=self._llm_model_id(),
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

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
            if settings.llm_openai_api_key() and self._llm_model_id():
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
                    "请在 backend/.env 中设置 LLM_API_KEY 与 LLM_MODEL 后重启。"
                ),
                "sources": [],
                "confidence": 0.0,
                "context_used": 0,
                "answer_mode": "system",
            }

        if not settings.llm_openai_api_key():
            return {
                "answer": (
                    "已从图数据库取回上下文，但未配置对话 API Key，无法调用大模型。"
                    "请在 backend/.env 中设置 LLM_API_KEY 后重启服务。"
                ),
                "sources": sources,
                "confidence": 1.0,
                "context_used": len(sources),
                "answer_mode": "system",
            }

        if not self._llm_model_id():
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
            model=self._llm_model_id(),
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
            model=self._llm_model_id(),
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
            model=self._llm_model_id(),
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def _format_sources(
        self, retrieved: List[tuple], *, source_channel: Optional[str] = None
    ) -> List[dict]:
        """Format retrieved chunks as sources."""
        sources = []
        for chunk_id, score, text, doc_id in retrieved:
            title = self._document_display_name(doc_id) if doc_id else ""
            item = {
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "text": text[:500] + "..." if len(text) > 500 else text,
                "document_id": doc_id or None,
                "document_title": title or None,
            }
            if source_channel:
                item["source_channel"] = source_channel
            sources.append(item)
        return sources
