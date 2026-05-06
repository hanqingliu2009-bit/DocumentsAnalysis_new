# 后端调试：从 Swagger UI 直接发消息

本文档用于在**不启动前端**的情况下，直接通过后端 Swagger UI 调试问答/检索接口。

---

## 1. 打开 Swagger UI

后端启动后，默认 Swagger UI 地址为：

- `http://<HOST>:<PORT>/docs`

例如本机默认端口：

- `http://127.0.0.1:8000/docs`

> 也可用 `http://127.0.0.1:8000/openapi.json` 检查 OpenAPI 是否正常生成。

---

## 2. 选哪个接口？（是否经过 LLM）

### 2.1 会经过 LLM（生成“回答”）

- `POST /api/query`：单轮问答（字段名是 `question`）
- `POST /api/chat`：多轮问答（字段名是 `message` + 可选 `history`）

这两个接口会进入 `RAGPipeline.query()`，进行检索后调用大模型生成回答（或按配置降级）。

### 2.2 不经过 LLM（只做“检索”）

- `POST /api/search`：**默认 Chroma 集合**的向量检索（由全局 `EMBEDDING_BACKEND` / `CHROMADB_COLLECTION` 决定），不生成回答
- `POST /api/splite_search`：**仅 Splite 手册集合**（`HYBRID_SPLITE_*`，本地 BGE + 专用集合），不生成回答；用于调试混合模式中的「本地一路」
- `POST /api/external_search`：**供应商图数据库检索**（HTTP 调用），不生成回答

> 若你希望“完全排除 embedding 与 LLM”，并且“只用供应商图数据库”，应使用 **`/api/external_search`**。

启用 **`RAG_BACKEND=hybrid`** 后，端到端问答仍用 **`/api/query`** 或 **`/api/chat`**（两路检索在服务端并行后再调 LLM）。详见 `doc/外部图数据库RAG实现说明.md` §14。

---

## 3. 在 Swagger UI 里如何发请求

Swagger UI 操作要点：

- 先展开对应 endpoint
- 点击 **Try it out**
- 在 Request body 中填入 JSON
- 点击 **Execute**

---

## 4. 示例：`POST /api/query`（单轮问答，走 LLM）

请求体最小示例：

```json
{
  "question": "我将要对一台高速智能封边机进行开机，开机前我要如何检查电源开关？",
  "top_k": 5
}
```

字段说明：

- `question`：必填
- `top_k`：可选，默认 5，范围 1–20
- `document_ids`：可选（限定只在指定文档内检索）

---

## 5. 示例：`POST /api/chat`（多轮问答，走 LLM）

请求体最小示例：

```json
{
  "message": "我将要对一台高速智能封边机进行开机，开机前我要如何检查电源开关？",
  "history": [],
  "top_k": 5
}
```

字段说明：

- `message`：必填（当前这轮用户消息）
- `history`：可选；如传入，数组元素形如 `{"role":"user|assistant|system","content":"..."}`  
- `document_ids`：可选
- `top_k`：可选，默认 5，范围 1–20

---

## 6. 示例：`POST /api/search`（本地向量检索，不走 LLM）

请求体最小示例：

```json
{
  "query": "开机前如何检查电源开关",
  "top_k": 5
}
```

常见报错与含义：

- `EMBEDDING_BACKEND=openai requires EMBEDDING_MODEL ...`
  - 说明你调用的是**本地向量检索链路**，需要配置 embedding 接入点（`EMBEDDING_MODEL` 等）
  - 若你的目标是“只走供应商图数据库”，请改用 `POST /api/external_search`

---

## 7. 示例：`POST /api/external_search`（供应商图数据库检索，不走 LLM）

请求体最小示例：

```json
{
  "query": "高速智能封边机维护保养手册适用于哪些产品型号？"
}
```

返回字段：

- `results`：归一化后的 sources（便于 UI 展示）
- `total`：results 数量
- `meta`：统计信息（如 triplet/summary/chunk 数量）

常见报错与含义：

- `The specified vector index name does not exist...`
  - 这是**供应商服务端**返回的业务错误，通常表示对方环境中缺少默认索引/索引名配置不匹配
  - 可把请求体（`query/domain/search_type`）和返回的 `detail` 原样发给供应商排查

---

## 8. 常见排查清单

- Swagger 能打开但接口报 500：先看响应体 `detail`（后端会透传错误信息）
- 要验证“是否经过 LLM”：
  - 用 `/api/query` 或 `/api/chat`：会生成 `answer/message`（通常会调用 LLM）
  - 用 `/api/search` 或 `/api/external_search`：只返回检索结果，不生成回答
- 要验证“是否在走供应商图数据库”：
  - 直接调用 `/api/external_search`
  - 或在 `.env` 配置 `RAG_BACKEND=external_graph` 后调用 `/api/query`/`/api/chat`（图检索 + LLM）

