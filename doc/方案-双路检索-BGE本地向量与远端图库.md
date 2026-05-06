# 方案：双路并行检索（远端图数据库 + 本地 BGE 向量库）

本文档描述在 **`system2`** 分支上要完成的目标、推荐数据形态、技术选型与**分阶段实施步骤**。实施时按章节顺序推进，每阶段结束可单独验收。

---

## 1. 目标（What）

1. **合并数据**：将 `files/` 目录下多份 `*_splite_plan.json` / `*_splite.json` 合并为**一份**结构化 JSON，便于批处理与追溯来源。
2. **中文向量**：选用对中文友好的 **BGE** 系列模型，对合并后的文本单元做 **embedding**，并写入**本地向量数据库**（当前工程为 **Chroma**，路径见 `backend/config.py` 的 `VECTOR_DB_PATH` / `CHROMADB_COLLECTION`）。
3. **双路检索**：在线问答时**并行**请求：
   - **远端**：供应方图数据库 HTTP（已有 `fetch_graph_context`，配置见 `EXTERNAL_GRAPH_*`）；
   - **本地**：Chroma 语义检索（BGE 向量）；
4. **合并与生成**：将两路检索结果**去重、排序、截断**后拼成统一上下文，调用 **LLM** 生成最终回答，并在响应中区分/标注来源（图库 vs 本地向量）。

非目标（本阶段默认不做，除非后续单列需求）：

- 替换火山方舟 LLM；仅讨论检索侧与上下文拼装。
- 把合并后的巨型 JSON 当作「用户上传文档」再走现有 PDF 流水线（可与本方案并存，但非必须）。

---

## 2. 现状与衔接点（Where）

| 能力 | 现状 | 本方案中的角色 |
|------|------|------------------|
| 图数据库 HTTP | `backend/core/external_graph.py` → `fetch_graph_context` | 双路之一，保持不变或小幅增强（超时、调试字段） |
| 本地向量 | `EmbeddingGenerator` + `VectorStore`（Chroma） | 双路之二；嵌入后端切到 **local + BGE** |
| RAG 入口 | `RAGPipeline.query()`，`RAG_BACKEND=external_graph` 时只走图 | 需新增**混合模式**（图 + 本地向量并行），不再二选一互斥 |
| API | `/api/query`、`/api/chat` | 行为不变或增加 `answer_mode` / `sources` 里 `source_type` 字段 |

---

## 3. 合并 JSON 的设计（Merge）

### 3.1 单文件结构（已观察）

每份文件大致包含：

- `document_name`、`document_id`、`total_pages`
- `chunks[]`：`chunk_id`、`group_id`、`group_title`、`content`、`pages` 等

### 3.2 合并后推荐顶层结构

建议合并为**一个根对象**，避免「多个根」或纯数组难以扩展元数据：

```json
{
  "version": 1,
  "generated_at": "ISO-8601 时间戳",
  "sources": [
    {
      "file": "原始文件名.json",
      "document_id": "F_pdf_xxx",
      "document_name": "xxx.pdf",
      "total_pages": 30,
      "chunks": [ ... ]
    }
  ]
}
```

**要点**：

- 每个元素对应原文件一份 **`document_id` + `chunks`**，避免不同文档的 `chunk_id` 在全局冲突时难以区分；向量库 metadata 中务必写入 **`document_id`** 与可选 **`chunk_id`**（或生成全局 `chunk_uid = document_id + '#' + chunk_id`）。
- 若希望扁平列表便于遍历，可在脚本里再导出一份 `all_chunks[]` 视图，但**仍以带 `sources` 的版本为权威**，便于审计。

### 3.3 落地位置（建议）

- **脚本**：`backend/scripts/merge_splite_json.py`（或 `tools/`），输入目录 `files/`，输出例如 `files/merged_splite_corpus.json`。
- **版本策略**：合并结果文件若较大，是否提交 git 由团队决定；常见做法是**脚本提交、大 JSON gitignore + 产物放构建机/对象存储**。

---

## 4. BGE 嵌入与向量库（Embed + Index）

### 4.1 模型选型（中文）

在 `sentence-transformers` 生态下，可选用例如（按资源与效果权衡）：

- **`BAAI/bge-large-zh-v1.5`**：中文检索质量较好，维数较大、算力占用高；
- **`BAAI/bge-base-zh-v1.5`**：折中；
- **`BAAI/bge-m3`**：多语言/多粒度，若后续有中英混合可优先考虑。

具体以团队显存与延迟测试为准。

### 4.2 与现有配置对齐

工程已支持 `EMBEDDING_BACKEND=local` 走 Hugging Face / 本地缓存（见 `backend/config.py` 与 `EmbeddingGenerator`）。实施步骤包括：

1. 在 `backend/.env` 设置例如：`EMBEDDING_BACKEND=local`、`EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5`（或所选 BGE id）。
2. **`EMBEDDING_DIMENSION`** 必须与所选模型输出维度一致（否则 Chroma 会报错或检索异常）。
3. **换模型或维度时**：使用**新集合名** `CHROMADB_COLLECTION` 或清空 `VECTOR_DB_PATH`，避免与旧向量混维度（代码注释里已有类似约定）。

### 4.3 建库内容（从合并 JSON 写什么）

对每个 chunk 建议写入向量 metadata 至少：

- `document_id`、`document_name`（或 title）
- `chunk_id`（原文件内）或全局 `chunk_uid`
- `group_title`（可选，用于展示）
- `text`：用于 embedding 的字符串，推荐模板例如：  
  `【标题】{group_title}\n【正文】{content}`  
  以提升检索对章节语义的区分度。

**批处理**：对 `merged_splite_corpus.json` 遍历 `sources[].chunks[]`，批量 `embed_text` + `vector_store.add`（若现有 API 无批量入口，可在脚本中直接操作 `VectorStore` 或封装 CLI）。

---

## 5. 双路并行检索与合并（Retrieve + Fuse）

### 5.1 并行方式

- **推荐**：`asyncio` 包装两路 IO（图 HTTP + 本地 Chroma 查询），或 `concurrent.futures.ThreadPoolExecutor`（若嵌入/ chroma 客户端为同步 API）。
- **超时**：图侧沿用 `EXTERNAL_GRAPH_TIMEOUT`；本地侧设合理上限，避免一路拖死整请求。

### 5.2 合并策略（需实现时定参）

两路结果形态不同（图：三元组/摘要/块；向量：chunk + score），建议统一为内部 **`EvidenceItem`** 列表，再：

1. **归一化分数**：图侧若无分数，可给固定权重或按类型加权；向量侧用相似度。
2. **去重**：同一 `content` 或高 Jaccard/字符重叠度阈值视为重复，保留高分或保留「图 + 向量」各一条。
3. **排序**：加权分数降序；或 **RRF（倒数排名融合）** 对两路各自排名做融合（实现简单、对分数尺度不敏感）。
4. **截断**：按字符数或 token 估算上限，保证进入 LLM 的上下文不超过模型窗口（可与 `MAX_TOKENS` 区分：上下文预算单独配置更清晰）。

### 5.3 LLM 提示词

- System：明确「上下文来自两部分：图数据库检索结果、本地手册向量检索结果」，要求引用时区分来源或编号。
- User：分段给出 `### 图数据库` 与 `### 本地手册` 两块，便于模型解析。

### 5.4 `answer_mode` 建议

新增例如：`hybrid_graph_vector`，便于日志与前端展示；`sources` 中每条增加 `source: "graph" | "vector"`。

---

## 6. 配置项扩展（Config）

建议在 `backend/config.py` / `.env` 增加（名称可微调）：

| 变量 | 含义 |
|------|------|
| `RAG_BACKEND` 新取值或子开关 | 例如 `hybrid` 或与 `external_graph` 组合：`HYBRID_LOCAL_VECTOR=true` |
| `HYBRID_GRAPH_TOP_K` / `HYBRID_VECTOR_TOP_K` | 两路各自取多少条再融合 |
| `HYBRID_FUSE_STRATEGY` | `rrf` / `weighted` 等 |
| `HYBRID_CONTEXT_MAX_CHARS` | 拼进 LLM 前的总字符上限 |

保持向后兼容：`RAG_BACKEND=chromadb` 与 `external_graph` 现有行为不变。

---

## 7. 分阶段实施步骤（How — 一步一步做）

下列顺序为建议执行顺序；每步完成后可打勾验收。

### 阶段 A：数据合并（无代码依赖变更）

- [x] **A1**：确认 `files/` 下所有待合并 JSON 列表与编码（UTF-8）。
- [x] **A2**：实现合并脚本 `backend/scripts/merge_splite_json.py`，默认输出 `files/merged_splite_corpus.json`（结构见 §3.2）。运行示例（仓库根目录）：
  - `backend\venv\Scripts\python.exe backend\scripts\merge_splite_json.py`
  - 可选：`--input <目录> --output <路径>`
- [x] **A3**：校验：`document_id` 在 7 份源中互不重复；`chunk_count` 等于各 `sources[].chunks` 长度之和；若需更严验收可再对单文件 diff 抽查 `content`。

**当前一次合并结果（示例）**：7 个源文件、651 条 `chunks`（以你本地 `files/` 为准，重跑脚本后会变）。

### 阶段 B：BGE 建索引（本地向量库）

- [x] **B1**：模型 **`BAAI/bge-large-zh-v1.5`**，输出维度 **1024**（sentence-transformers）。
- [x] **B2**：灌库脚本在运行前会强制 `EMBEDDING_BACKEND=local` 及 `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION`（不依赖 `.env` 里火山嵌入）；日常用 API 检索该集合时需在 `.env` 对齐，见 `backend/.env.example` 末尾「Splite corpus」示例块。
- [x] **B3**：独立集合名 **`splite_bge_zh_v15`**（与默认 `document_chunks` 分离，避免维度混用）。
- [x] **B4**：脚本 **`backend/scripts/ingest_merged_corpus_bge.py`**：读 `files/merged_splite_corpus.json` → 文本模板 `【group_title】\ncontent` → 批量 embed → `VectorStore.add_chunks`。脚本会将 **`cwd` 设为 `backend/`**，使 Chroma 落在 **`backend/data/vector_db`**（与 uvicorn 一致）。仓库根执行示例：
  - `backend\venv\Scripts\python.exe backend\scripts\ingest_merged_corpus_bge.py --recreate`
  - 可选：`--corpus files\merged_splite_corpus.json`、`--collection splite_bge_zh_v15`、`--batch-size 32`
- [x] **B5**：灌库后冒烟：同一模型嵌入查询 + `VectorStore(..., splite_bge_zh_v15).search`，`SIMILARITY_THRESHOLD` 默认下可返回命中（具体条数随 query 而变）。

### 阶段 C：RAG 混合检索逻辑

- [x] **C1**：在 `RAGPipeline.query()` 中识别 **`RAG_BACKEND=hybrid`**，走 `_query_hybrid`：并行调用 `fetch_graph_context` 与「Splite 专用 `VectorStore` + 强制本地 BGE `EmbeddingGenerator(..., embed_texts(..., embedding_backend=\"local\"))`」。
- [x] **C2**：`ThreadPoolExecutor(max_workers=2)` 并行；单侧异常时该路为空，另一路仍参与（未单独加超时包装，图侧仍遵循 `EXTERNAL_GRAPH_TIMEOUT`）。
- [x] **C3**：融合策略为 **两节 Markdown 拼接**（`### 图数据库检索结果` + `### 本地手册向量检索`）+ **`HYBRID_CONTEXT_MAX_CHARS` 硬截断**（未做 RRF，可后续增强）。
- [x] **C4**：专用 system prompt `_generate_answer_hybrid`；`sources` 中图条目带 `source_channel=graph`，向量条目带 `source_channel=vector`；API `SourceCitation` 增加可选字段 `source_channel`；`answer_mode=hybrid_graph_vector`。
- [x] **C5**：`tests/core/test_rag_hybrid.py`（mock 图 + mock 向量 + LLM）。

**启用方式**：`backend/.env` 中 `RAG_BACKEND=hybrid`，并保证 LLM、图接口、Splite 集合与 `HYBRID_*` 配置就绪（见 `backend/.env.example`）。

### 阶段 D：配置、文档与运维

- [x] **D1**：`backend/.env.example` 已含 Splite / Hybrid 块；`doc/外部图数据库RAG实现说明.md` 已增 **§14 混合模式** 与文件表、测试命令引用。
- [x] **D2**：`backend/.env.example` 末尾 **Ops** 小节：合并脚本、灌库、HF 缓存、`backend/data/vector_db`、重灌说明。
- [x] **D3**：新增 **`POST /api/splite_search`**（请求体同 `/api/search`），仅查 Splite BGE 集合、无 LLM，便于 Swagger 调试。

### 阶段 E：联调与性能

- [ ] **E1**：端到端延迟：并行是否优于串行；调整 `top_k` 与上下文上限。
- [ ] **E2**：与供应方图库联调：索引名/域名等问题与对方确认；失败时不拖垮整请求。

---

## 8. 风险与注意事项

- **维度与集合**：换 BGE 必须同步维度与集合，否则 silent bug 或写入失败。
- **许可证与模型分发**：BGE 权重遵循其模型卡许可；内网需提前缓存。
- **上下文膨胀**：双路合并后极易超长，**必须有硬截断**，否则 LLM 慢且贵。
- **图侧不稳定**：并行 + 超时 + 单侧降级是必选项。

---

## 9. 下一步（立即要做的第一件事）

进入 **阶段 D/E**：在真实环境打开 `RAG_BACKEND=hybrid` 做端到端联调，按需调 `HYBRID_VECTOR_TOP_K`、`HYBRID_CONTEXT_MAX_CHARS`、`SIMILARITY_THRESHOLD`；可选为图侧增加显式超时/日志与 RRF 融合。

---

## 10. 文档维护

本文件路径：`doc/方案-双路检索-BGE本地向量与远端图库.md`。  
实施过程中若变更合并 schema、集合名或融合算法，请同步更新本节与相关 `doc/` 说明。
