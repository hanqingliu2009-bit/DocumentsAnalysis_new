# `backend/.env` 配置示例（三种常用场景）

实际运行时请将内容保存为 **`backend/.env`**（与 `backend/config.py` 同级），并把占位符换成你自己的密钥与模型名。**勿将含真实 Key 的 `.env` 提交到 git。**

字段说明还可对照仓库根目录的 **`backend/.env.example`**（精简注释版）与 **`backend/config.py`** 中的默认值。

---

## 方案 A：本地上传文档向量 + 云端对话（最常用）

适用：`RAG_BACKEND` 默认 **`chromadb`** —— 上传的 PDF 等在本地用 sentence-transformers 建向量索引，问答走 OpenAI 兼容的 `chat.completions`（例如阿里云百炼）。

```env
# ========== 必配：对话 LLM（OpenAI 兼容）==========
LLM_API_KEY=sk-你的DashScope_API_Key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo

# ========== 嵌入：本地 MiniLM（与 config 默认一致）==========
EMBEDDING_BACKEND=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
CHROMADB_COLLECTION=document_chunks

# 可选：把 Hugging Face 权重缓存在 backend 下，便于备份/离线
# EMBEDDING_CACHE_DIR=./data/models/hf_home
# TRANSFORMERS_OFFLINE=false

# ========== 可选：检索与分块 ==========
# TOP_K_RETRIEVAL=15
# SIMILARITY_THRESHOLD=0.7
# CHUNK_SIZE=512
# CHUNK_OVERLAP=50

# ========== 可选：服务 ==========
# HOST=0.0.0.0
# PORT=8000
# DEBUG=false
```

**说明**

- `LLM_MODEL` 以服务商控制台为准（如 `qwen-plus`、`qwen-turbo` 等）。
- 若你曾用其他维度或集合灌过 Chroma，**不要与当前 `EMBEDDING_DIMENSION` / `CHROMADB_COLLECTION` 混用**：要么清空 `backend/data/vector_db` 后重新上传，要么保持与建库时一致的嵌入配置。

---

## 方案 B：混合检索（图库 + Splite BGE 向量 + 同一套 LLM）

在**方案 A** 的 LLM 与（可选）本地文档嵌入配置基础上，启用并行图检索 + Splite 专用向量库。**需先**按项目文档完成 merge、ingest（例如 `splite_bge_zh_v15` 集合）。

```env
# --- 先保留方案 A 中的 LLM_* 与（按需）EMBEDDING_* ---

RAG_BACKEND=hybrid
HYBRID_SPLITE_COLLECTION=splite_bge_zh_v15
HYBRID_SPLITE_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
HYBRID_VECTOR_TOP_K=5
HYBRID_CONTEXT_MAX_CHARS=24000

# 图接口可按环境修改
# EXTERNAL_GRAPH_API_URL=http://14.103.133.160:8022/graph/info
# EXTERNAL_GRAPH_DOMAIN=南兴装备
```

**说明**

- 「上传文档」默认索引仍由方案 A 的 `EMBEDDING_BACKEND=local` + `CHROMADB_COLLECTION` 等控制；Splite 一路在代码中会单独使用 BGE，与 `HYBRID_*` 对齐即可。

---

## 方案 C：嵌入也走云端 OpenAI 兼容 `/embeddings`

不使用本地 sentence-transformers 建上传文档的向量时，改为 HTTP 调用兼容的 `embeddings` 接口。密钥与基址通常可与对话共用；若不同，可单独设置 `EMBEDDING_OPENAI_*`。

```env
# ========== 对话 LLM ==========
LLM_API_KEY=sk-你的DashScope_API_Key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo

# ========== 远程嵌入 ==========
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_DIMENSION=1024
CHROMADB_COLLECTION=document_chunks

# 若嵌入与对话使用不同 Key 或不同 Base URL，再取消注释并填写：
# EMBEDDING_OPENAI_API_KEY=
# EMBEDDING_OPENAI_BASE_URL=

# 部分多模态嵌入模型需走 /embeddings/multimodal（按控制台说明）
# EMBEDDING_USE_MULTIMODAL_API=false
```

**说明**

- `EMBEDDING_MODEL` 与 **`EMBEDDING_DIMENSION` 必须与实际模型一致**；不确定维度时，可用同一 Key 调用一次 embeddings，看返回向量长度。
- 切换嵌入模型或维度后，应更换 `CHROMADB_COLLECTION` 或清空 `backend/data/vector_db`，避免不同维度混写。

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `backend/.env` | 本地真实配置（不提交） |
| `backend/.env.example` | 仓库内精简条目与注释 |
| `backend/config.py` | 全部环境变量名与默认值 |
| `PROJECT.md` | 含 Hybrid 部署索引与文档链接 |
