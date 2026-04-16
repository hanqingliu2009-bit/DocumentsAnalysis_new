# Technical Manual Q&A System - Design Document

## Overview
A prototype system that ingests technical manuals and maintenance documents, indexes them for semantic search, and answers questions using retrieval-augmented generation (RAG).

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Web Interface             │
                    │  (React + FastAPI Backend)          │
                    │  - Document Upload Dashboard        │
                    │  - Chat Interface                   │
                    │  - Search & Browse                  │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │         FastAPI Backend API         │
                    │  - /api/documents (CRUD)            │
                    │  - /api/query (Q&A)                   │
                    │  - /api/chat (Conversational)       │
                    └──────────────┬──────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐      ┌───────────▼──────────┐    ┌───────────▼──────────┐
│   Document     │      │   Document           │    │   Vector             │
│   Ingestion    │──────▶│   Processor          │───▶│   Store/Index        │
│   (PDF/DOCX)   │      │   (Chunk/Embed)      │    │   (ChromaDB)         │
└────────────────┘      └──────────────────────┘    └──────────────────────┘
                                                                  │
                                                                  │
        ┌──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐  │
│  │   Query     │───▶│   Context    │───▶│   LLM (Claude)      │  │
│  │   Embed     │    │   Assembly   │    │   Answer Generation │  │
│  └─────────────┘    └──────────────┘    └─────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Document Ingestion Layer
- **Purpose**: Accept documents in various formats
- **Supported Formats**: PDF, DOCX, TXT, Markdown
- **Key Libraries**: `pypdf`, `python-docx`, `unstructured`

#### PDF Parser Options
The system supports pluggable PDF parsers via configuration:

**Option 1: Default (pypdf)**
- **Library**: `pypdf>=4.0.0`
- **Best for**: Simple text-based PDFs, fast processing
- **Pros**: Pure Python, no external dependencies, good performance
- **Cons**: Limited support for complex layouts, tables, scanned PDFs

**Option 2: Advanced (opendataloader-pdf)** ⚙️ Configurable
- **Repository**: https://github.com/yiliangbetter/opendataloader-pdf
- **Best for**: Complex technical manuals, tables, mixed layouts
- **Installation**: 
  ```bash
  pip install git+https://github.com/yiliangbetter/opendataloader-pdf.git
  ```
- **Configuration**: Set `PDF_PARSER=opendataloader` in `.env`
- **Pros**: Better handling of complex layouts, tables, figure captions
- **Cons**: Additional dependency, may be slower for simple PDFs

**Parser Selection Logic**:
```python
if config.PDF_PARSER == "opendataloader":
    from opendataloader_pdf import PDFLoader
    return PDFLoader()
else:
    from pypdf import PdfReader
    return PyPDFLoader()
```

### 2. Document Processor
- **Text Extraction**: Extract clean text from documents
- **Metadata Extraction**: Document title, author, date, page numbers
- **Chunking Strategy**:
  - Chunk size: 512 tokens (configurable)
  - Overlap: 50 tokens between chunks
  - Preserve paragraph boundaries when possible
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, fast and good quality)

### 3. Vector Store
- **Technology**: ChromaDB (embedded, no external dependencies)
- **Storage**: Local filesystem
- **Features**: Similarity search, metadata filtering
- **Index Type**: HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search

### 4. Query Engine
- **Query Processing**: Convert natural language questions to embeddings
- **Retrieval Strategy**:
  - Top-k retrieval (k=5 by default)
  - Similarity threshold filtering (optional)
  - Metadata filters (e.g., search only specific document types)

### 5. Answer Generation (RAG Pipeline)
- **LLM**: Claude via Anthropic API
- **Context Window**: Up to 200K tokens (more than enough for retrieved chunks)
- **Prompt Template**:
  ```
  You are a technical support assistant. Answer the question based ONLY on the provided context.
  If the context doesn't contain enough information, say "I don't have enough information to answer this."
  Cite specific document names and sections when possible.

  Context:
  {retrieved_chunks}

  Question: {user_question}

  Answer:
  ```

## Web Interface

### Frontend (React + TypeScript)

#### Pages & Components

1. **Dashboard (`/`)**
   - Document statistics (total docs, chunks, recent uploads)
   - Quick upload button
   - Recent queries history
   - Quick search bar

2. **Document Manager (`/documents`)**
   - List view: Document cards with metadata (title, type, date, size)
   - Upload zone: Drag-and-drop multi-file upload
   - Actions: View, Download, Delete
   - Filters: By type, date range, search by title

3. **Chat Interface (`/chat`)**
   - Chat-style interface (similar to ChatGPT)
   - Message history with user questions and AI answers
   - Source citations (expandable to see which documents were used)
   - Follow-up question suggestions
   - Clear conversation button

4. **Search & Browse (`/search`)**
   - Advanced search with filters
   - Highlighted results with context snippets
   - Faceted search (by document, date, type)

#### Technology Stack

```
Frontend:
- React 18 + TypeScript
- Vite (build tool)
- Chakra UI or Tailwind CSS (component library)
- React Query (server state management)
- React Router (routing)
- Axios (HTTP client)
```

### Backend API (FastAPI)

#### REST Endpoints

```python
# Documents
GET    /api/documents              # List all documents
POST   /api/documents              # Upload document(s)
GET    /api/documents/{id}         # Get document details
DELETE /api/documents/{id}         # Delete document
GET    /api/documents/{id}/download # Download original file

# Query/Chat
POST   /api/query                  # Single question → answer
POST   /api/chat                   # Chat with history
POST   /api/search                 # Semantic search (no LLM)

# System
GET    /api/stats                  # System stats (doc count, etc.)
GET    /api/health                 # Health check
```

#### WebSocket Support (Optional)
- `/ws/chat` - Real-time streaming responses for chat

#### Technology Stack

```
Backend:
- FastAPI (web framework)
- Uvicorn (ASGI server)
- Pydantic (validation)
- Python-multipart (file uploads)
- WebSocket support for streaming
```

### Shared Models (Pydantic)

```python
# Request/Response Models
class DocumentUploadResponse(BaseModel):
    id: str
    title: str
    status: "processing" | "completed" | "failed"

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    filters: dict | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    confidence: float

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None  # For continuing chats
    history: list[ChatMessage] | None = None

class ChatResponse(BaseModel):
    message: str
    sources: list[SourceCitation]
    conversation_id: str
```

## Updated Project Structure

See `PROJECT.md` for detailed project structure, implementation phases, dependencies, and configuration.

## Summary of Web Interface Features

| Feature | Description |
|---------|-------------|
| **Dashboard** | Overview stats, quick upload, recent queries |
| **Document Manager** | Upload, view, delete documents; drag-and-drop support |
| **Chat Interface** | ChatGPT-style Q&A with source citations |
| **Search & Browse** | Advanced semantic search with filters |
| **Real-time** | WebSocket streaming for chat responses |

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/documents` | List documents |
| POST | `/api/documents` | Upload document(s) |
| DELETE | `/api/documents/{id}` | Delete document |
| POST | `/api/query` | Single question → answer |
| POST | `/api/chat` | Chat with history |
| POST | `/api/search` | Semantic search |
| GET | `/api/stats` | System statistics |

---

## Next Steps

1. **Review the updated design** - Provide feedback on the web interface, API design, or features
2. **Approve the design** - I'll proceed with Phase 1 implementation
3. **Request modifications** - Change scope, add features, or adjust priorities

What would you like to change or proceed with?
