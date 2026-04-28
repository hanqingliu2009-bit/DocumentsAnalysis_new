# 外部图数据库 RAG 实现说明

本文档总结在分支 **`nanxing-db`** 上完成的改造：在启用 **`RAG_BACKEND=external_graph`** 时，问答不再使用本地 Embedding + Chroma 向量检索，改为调用供应方图数据库 HTTP 接口，将返回的三元组/摘要/文本块作为上下文，由现有火山兼容大模型生成回答。

---

## 1. 架构与数据流

```mermaid
flowchart LR
  subgraph client [本应用]
    Chat["POST /api/chat 或 /api/query"]
    RAG["RAGPipeline.query"]
    Graph["fetch_graph_context"]
    LLM["OpenAI 兼容 chat.completions"]
  end
  subgraph supplier [供应方]
    API["POST .../graph/info"]
    GDB[(图数据库检索)]
  end
  Chat --> RAG
  RAG --> Graph
  Graph -->|JSON: query, domain, search_type| API
  API --> GDB
  GDB -->|triplet, summary, chunk| Graph
  Graph -->|拼成上下文文本| LLM
  RAG --> LLM
```

- **图存储与检索逻辑**：在供应方；本仓库只发约定 JSON，并解析约定字段。
- **生成回答**：仍在本地，依赖 `VOLCENGINE_API_KEY`、`LLM_MODEL`（与原先 Chroma 方案相同）。

---

## 2. 涉及文件

| 文件 | 作用 |
|------|------|
| `backend/config.py` | `RAG_BACKEND`、图接口 URL、domain、`search_type`、超时 |
| `backend/core/external_graph.py` | HTTP 请求、解析 `message`、三元组字典格式化、拼上下文与 sources |
| `backend/core/rag.py` | `query()` 分支；`_query_external_graph`；图上下文专用 system prompt |
| `backend/main.py` | `external_graph` 模式下跳过本地 Embedding 预热 |
| `backend/api/query.py` | `answer_mode` 说明含 `external_graph`；chat 仍传 `retrieval_query` |
| `backend/tests/core/test_external_graph.py` | 图客户端单元测试 |
| `CLAUDE.md` | 约定使用 `backend/venv` 安装依赖与跑 pytest |

---

## 3. 环境变量（`backend/.env` 示例）

以下仅为说明，**不要**把真实 API Key 提交到 git。

```env
# 启用供应方图检索 + 本地大模型作答
RAG_BACKEND=external_graph

# 可选；不设则用 config.py 中的默认值
EXTERNAL_GRAPH_API_URL=http://14.103.133.160:8022/graph/info
EXTERNAL_GRAPH_DOMAIN=南兴装备
EXTERNAL_GRAPH_SEARCH_TYPE=triplet
EXTERNAL_GRAPH_TIMEOUT=60

# 仍必需：用于生成最终回答
VOLCENGINE_API_KEY=你的密钥
LLM_MODEL=你的接入点或模型名
```

修改 `.env` 后请在 **`backend/venv`** 环境中重启 `uvicorn`。

---

## 4. 供应方 HTTP 约定

**请求**：`POST`，`Content-Type: application/json`，body 形如：

```json
{
  "query": "用户当前问题文本",
  "domain": "南兴装备",
  "search_type": "triplet"
}
```

**响应**（已对接的一种形态）：顶层含 `success`；`message` 可为对象或 JSON 字符串，其中常见字段：

- `triplet`：字符串列表，或「头/关系/尾」字典列表（见下文代码中的字段映射）
- `summary`：摘要列表
- `chunk`：文本块列表

---

## 5. 配置项代码（`backend/config.py`）

```75:80:backend/config.py
    # RAG backend: "chromadb" = local embeddings + Chroma; "external_graph" = supplier graph HTTP + LLM.
    RAG_BACKEND: str = "chromadb"
    EXTERNAL_GRAPH_API_URL: str = "http://14.103.133.160:8022/graph/info"
    EXTERNAL_GRAPH_DOMAIN: str = "南兴装备"
    EXTERNAL_GRAPH_SEARCH_TYPE: str = "triplet"
    EXTERNAL_GRAPH_TIMEOUT: float = 60.0
```

---

## 6. 图 HTTP 客户端与上下文拼装（`backend/core/external_graph.py`）

### 6.1 三元组：字符串或字典 → 统一成行

字典时尝试常见键名（`head`/`subject`、`relation`/`predicate`、`tail`/`object` 等），格式化为 `头 -关系-> 尾`：

```15:42:backend/core/external_graph.py
def _triplet_item_to_line(item: Any) -> str:
    """Turn one graph triplet (string or common dict shape) into one human-readable line."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return str(item).strip()

    head_keys = ("head", "subject", "source", "h", "entity", "start", "from")
    rel_keys = ("relation", "predicate", "rel", "type", "edge", "r")
    tail_keys = ("tail", "object", "target", "t", "end", "to", "o")

    def pick(keys: tuple[str, ...]) -> str:
        for k in keys:
            v = item.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        return ""

    head = pick(head_keys)
    rel = pick(rel_keys)
    tail = pick(tail_keys)
    if head and rel and tail:
        return f"{head} -{rel}-> {tail}"
    if head and tail:
        return f"{head} -> {tail}"
    return json.dumps(item, ensure_ascii=False)
```

### 6.2 请求体、解析与 Markdown 小节

通过 **`import config`** 使用 **`config.settings`**，便于测试里 `patch("config.settings", ...)`。

```81:162:backend/core/external_graph.py
def fetch_graph_context(query: str) -> Tuple[str, List[dict], Dict[str, Any]]:
    """
    POST JSON {query, domain, search_type} to the supplier graph API.

    Expects a graph/RAG-style JSON body (e.g. ``success`` + ``message`` with
    ``triplet``, ``summary``, ``chunk``). Triplets may be strings or dicts with
    common keys (head/subject + relation/predicate + tail/object).

    Returns:
        (context_text_for_llm, sources_for_api_response, raw_body_for_debug)
    """
    cfg = config.settings
    url = (cfg.EXTERNAL_GRAPH_API_URL or "").strip()
    if not url:
        raise ValueError("EXTERNAL_GRAPH_API_URL is empty")

    payload = {
        "query": query.strip(),
        "domain": cfg.EXTERNAL_GRAPH_DOMAIN,
        "search_type": cfg.EXTERNAL_GRAPH_SEARCH_TYPE,
    }
    timeout = float(cfg.EXTERNAL_GRAPH_TIMEOUT or 60.0)

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        body: Dict[str, Any] = response.json()

    if not body.get("success", True):
        msg = body.get("message", "unknown error")
        raise RuntimeError(f"Graph API reported failure: {msg}")

    message = _normalize_message(body.get("message"))
    triplets = _as_str_list(message.get("triplet"))
    summaries = _as_str_list(message.get("summary"))
    chunks = _as_str_list(message.get("chunk"))

    parts: List[str] = []
    if triplets:
        lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(triplets))
        parts.append(f"## 图数据库三元组\n{lines}")
    if summaries:
        lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(summaries))
        parts.append(f"## 摘要\n{lines}")
    if chunks:
        lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(chunks))
        parts.append(f"## 文本块\n{lines}")

    context = "\n\n".join(parts).strip()
    meta: Dict[str, Any] = {
        "triplet_count": len(triplets),
        "summary_count": len(summaries),
        "chunk_count": len(chunks),
    }

    sources: List[dict] = []
    for i, text in enumerate(triplets):
        sources.append({
            "chunk_id": f"graph-triplet-{i + 1}",
            "score": 1.0,
            "text": text[:500] + "..." if len(text) > 500 else text,
            "document_id": None,
            "document_title": "图数据库·三元组",
        })
    for i, text in enumerate(summaries):
        sources.append({
            "chunk_id": f"graph-summary-{i + 1}",
            "score": 1.0,
            "text": text[:500] + "..." if len(text) > 500 else text,
            "document_id": None,
            "document_title": "图数据库·摘要",
        })
    for i, text in enumerate(chunks):
        sources.append({
            "chunk_id": f"graph-chunk-{i + 1}",
            "score": 1.0,
            "text": text[:500] + "..." if len(text) > 500 else text,
            "document_id": None,
            "document_title": "图数据库·文本块",
        })

    return context, sources, {"response": body, "meta": meta}
```

### 6.3 图库无命中时的系统提示（无上下文走 LLM）

```164:169:backend/core/external_graph.py
def build_graph_system_prompt_no_hits() -> str:
    return (
        "你是通用助手。本轮「图数据库检索」没有返回可用的三元组、摘要或文本块；"
        "不要编造具体设备条款或内部数据。回答语言与用户问题一致。"
    )
```

---

## 7. RAG 流水线分支（`backend/core/rag.py`）

### 7.1 `query()` 入口：按 `RAG_BACKEND` 分流

```57:60:backend/core/rag.py
        try:
            backend = (getattr(settings, "RAG_BACKEND", "chromadb") or "chromadb").strip().lower()
            if backend == "external_graph":
                return self._query_external_graph(question, retrieval_query)
```

### 7.2 外部图路径：取上下文 → 校验 LLM 配置 → 生成回答

```164:248:backend/core/rag.py
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
```

### 7.3 图上下文专用对话模板

```250:274:backend/core/rag.py
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
```

---

## 8. 启动时跳过本地 Embedding 预热（`backend/main.py`）

在 `RAG_BACKEND=external_graph` 时不再调用 `EmbeddingGenerator().embed_text("warmup")`，避免无嵌入配置时的无意义启动开销：

```34:60:backend/main.py
    # Warm up embedding so first chat/upload does not appear to "hang" with no server log.
    try:
        rag_backend = (getattr(settings, "RAG_BACKEND", "chromadb") or "chromadb").strip().lower()
        if rag_backend == "external_graph":
            print("RAG_BACKEND=external_graph: skipping local embedding warmup.")
        else:
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
                mm = bool(getattr(settings, "EMBEDDING_USE_MULTIMODAL_API", False))
                api_note = ", multimodal /embeddings/multimodal" if mm else ""
                print(
                    f"Warming up Ark embeddings (backend={settings.EMBEDDING_BACKEND}, "
                    f"model={settings.EMBEDDING_MODEL or 'unset'}{api_note})..."
                )
            EmbeddingGenerator().embed_text("warmup")
            print("Embedding backend ready.")
```

---

## 9. Chat 与图检索 query 的对应关系（`backend/api/query.py`）

多轮对话时，**整段历史 + 当前问句** 仍传给大模型作为 `question`；**仅当前用户一句** 作为 `retrieval_query` 发给图接口（与原先向量检索「只 embed 最新一句」一致）：

```215:224:backend/api/query.py
        # Combine history with current question
        full_question = f"{history_context}\nCurrent question: {request.message}"

        result = get_rag_pipeline().query(
            question=full_question,
            document_ids=request.document_ids,
            top_k=request.top_k,
            # Embed only the latest user turn; long history wrecks semantic search.
            retrieval_query=request.message.strip(),
        )
```

---

## 10. 未改动的行为说明

- **`RAG_BACKEND` 默认 `chromadb`**：未改 `.env` 时行为与改造前一致（本地 Embedding + Chroma）。
- **`POST /api/search`**：仍为本地向量语义检索，不调用图接口。
- **文档上传与索引**：仍写入本地存储与向量库；图方案下问答不依赖这些路径，但应用启动仍会初始化 `DocumentStore` / `VectorStore`。

---

## 11. 测试

在 **`backend/venv`** 中执行（若尚未安装 pytest，需 `pip install pytest`；若 `pytest.ini` 带 `--cov` 但未装 `pytest-cov`，可加 `-o addopts=`）：

```bash
cd backend
.\venv\Scripts\Activate.ps1
python -m pytest tests/core/test_external_graph.py -q -o addopts=
```

---

## 12. 与仓库内代码的同步

文档中的「`startLine:endLine:path`」代码块与仓库文件行号一致，便于在 IDE 中对照。若你后续改动上述文件，可用 diff 检查本节是否需更新行号或片段。
