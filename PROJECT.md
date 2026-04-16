# Technical Manual Q&A System - Project Implementation Details

This document contains the detailed project structure, implementation phases, dependencies, and configuration for the Document Q&A system.

## Project Structure

```
docqa/
├── README.md
├── requirements.txt
├── DESIGN.md              # Architecture and design decisions
├── PROJECT.md             # This file - implementation details
├── .env.example           # Example environment variables
│
├── backend/               # FastAPI Backend
│   ├── __init__.py
│   ├── main.py            # FastAPI app entry point
│   ├── config.py          # Backend configuration
│   ├── dependencies.py    # FastAPI dependencies
│   │
│   ├── api/               # API Routes
│   │   ├── __init__.py
│   │   ├── documents.py   # Document endpoints
│   │   ├── query.py       # Query/Chat endpoints
│   │   └── system.py      # Stats/Health endpoints
│   │
│   ├── core/              # Core Business Logic
│   │   ├── __init__.py
│   │   ├── document.py    # Document models
│   │   ├── processor.py   # Text extraction & chunking
│   │   ├── indexer.py     # Vector indexing
│   │   └── rag.py         # Retrieval + Generation
│   │
│   ├── storage/           # Storage Layer
│   │   ├── __init__.py
│   │   ├── vector_store.py    # ChromaDB wrapper
│   │   ├── document_store.py  # Document metadata storage
│   │   └── file_store.py      # Raw file storage
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── frontend/              # React Frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   │
│   ├── src/
│   │   ├── main.tsx           # React entry point
│   │   ├── App.tsx            # Main app component
│   │   ├── App.css
│   │   │
│   │   ├── components/        # Reusable components
│   │   │   ├── Layout.tsx
│   │   │   ├── Navbar.tsx
│   │   │   ├── DocumentCard.tsx
│   │   │   ├── DocumentList.tsx
│   │   │   ├── UploadZone.tsx
│   │   │   ├── ChatMessage.tsx
│   │   │   ├── ChatInput.tsx
│   │   │   ├── SearchResults.tsx
│   │   │   ├── StatCard.tsx
│   │   │   └── SourceCitation.tsx
│   │   │
│   │   ├── pages/             # Page components
│   │   │   ├── Dashboard.tsx
│   │   │   ├── DocumentManager.tsx
│   │   │   ├── ChatInterface.tsx
│   │   │   └── SearchBrowse.tsx
│   │   │
│   │   ├── hooks/             # Custom React hooks
│   │   │   ├── useDocuments.ts
│   │   │   ├── useChat.ts
│   │   │   ├── useQuery.ts
│   │   │   └── useApi.ts
│   │   │
│   │   ├── services/          # API services
│   │   │   └── api.ts
│   │   │
│   │   ├── store/             # State management (Zustand)
│   │   │   ├── useAppStore.ts
│   │   │   └── useChatStore.ts
│   │   │
│   │   └── types/             # TypeScript types
│   │       └── index.ts
│   │
│   └── public/
│       └── favicon.ico
│
└── data/                      # Data storage (gitignored)
    ├── vector_db/             # ChromaDB files
    ├── documents/             # Document metadata
    └── files/                 # Uploaded raw files
```

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Project setup (requirements, structure)
- [ ] Document models and data classes
- [ ] Configuration management
- [ ] Backend FastAPI skeleton

### Phase 2: Ingestion Pipeline
- [ ] Document loaders (PDF, DOCX, TXT)
- [ ] Text extraction
- [ ] Metadata extraction
- [ ] File storage system

### Phase 3: Processing & Indexing
- [ ] Text chunking with overlap
- [ ] Embedding generation
- [ ] ChromaDB integration
- [ ] Vector storage

### Phase 4: Query & RAG
- [ ] Similarity search
- [ ] Context assembly
- [ ] Claude API integration
- [ ] Answer generation

### Phase 5: Backend API
- [ ] Document endpoints (CRUD)
- [ ] Query endpoints
- [ ] Chat endpoints
- [ ] WebSocket streaming support

### Phase 6: Frontend
- [ ] React + TypeScript setup
- [ ] Dashboard page
- [ ] Document Manager page
- [ ] Chat Interface page
- [ ] API integration

### Phase 7: Polish & CLI
- [ ] CLI implementation
- [ ] Error handling
- [ ] Documentation
- [ ] Testing

## Dependencies

### Backend Dependencies (`backend/requirements.txt`)

```txt
# Web Framework
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
websockets>=12.0

# Core
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-dotenv>=1.0.0

# Document Processing
pypdf>=4.0.0
python-docx>=1.1.0
unstructured>=0.12.0
python-magic>=0.4.27
# Optional: Advanced PDF parsing
# git+https://github.com/yiliangbetter/opendataloader-pdf.git

# Embeddings & Vector DB
sentence-transformers>=2.5.0
chromadb>=0.4.0

# LLM
anthropic>=0.18.0

# Utilities
tqdm>=4.66.0
aiofiles>=23.2.0
```

### Frontend Dependencies (`frontend/package.json`)

```json
{
  "name": "docqa-frontend",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.21.0",
    "@tanstack/react-query": "^5.17.0",
    "axios": "^1.6.5",
    "zustand": "^4.4.7",
    "@chakra-ui/react": "^2.8.2",
    "@emotion/react": "^11.11.3",
    "@emotion/styled": "^11.11.0",
    "framer-motion": "^10.18.0",
    "react-dropzone": "^14.2.3",
    "react-markdown": "^9.0.1",
    "react-icons": "^4.12.0",
    "date-fns": "^3.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@typescript-eslint/eslint-plugin": "^6.14.0",
    "@typescript-eslint/parser": "^6.14.0",
    "@vitejs/plugin-react": "^4.2.1",
    "eslint": "^8.55.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.5",
    "typescript": "^5.2.2",
    "vite": "^5.0.8"
  }
}
```

### CLI Dependencies (`requirements-cli.txt`)

```txt
# CLI-only dependencies (for CLI usage without backend)
click>=8.1.0
rich>=13.0.0
typer>=0.9.0

# Shared with backend
pydantic>=2.5.0
python-dotenv>=1.0.0
pypdf>=4.0.0
python-docx>=1.1.0
sentence-transformers>=2.5.0
chromadb>=0.4.0
anthropic>=0.18.0
tqdm>=4.66.0
```

## Configuration

### Backend Environment Variables (`backend/.env`)

```bash
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=info

# LLM Configuration
ANTHROPIC_API_KEY=your_key_here
LLM_MODEL=claude-3-sonnet-20240229
MAX_TOKENS=4096
TEMPERATURE=0.0

# Paths
VECTOR_DB_PATH=./data/vector_db
DOCUMENT_STORE_PATH=./data/documents
FILE_STORAGE_PATH=./data/files

# Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=50
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# PDF Parser Configuration (options: pypdf, opendataloader)
PDF_PARSER=pypdf

# Search
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.7

# CORS (for frontend)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Upload Limits
MAX_FILE_SIZE=104857600  # 100MB
MAX_FILES_PER_UPLOAD=10
```

### Frontend Environment Variables (`frontend/.env`)

```bash
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## API Endpoints Reference

### Documents API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/documents` | List documents (paginated) |
| POST | `/api/documents` | Upload new document(s) |
| GET | `/api/documents/{id}` | Get document details |
| DELETE | `/api/documents/{id}` | Delete document |
| GET | `/api/documents/{id}/download` | Download original file |
| GET | `/api/documents/{id}/chunks` | Get document chunks |

### Query API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/query` | Single question → answer |
| POST | `/api/search` | Semantic search (no LLM) |

### Chat API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Chat with history |
| WS | `/ws/chat` | WebSocket chat stream |
| GET | `/api/chat/history` | Get chat history |
| DELETE | `/api/chat/{id}` | Delete chat session |

### System API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stats` | System statistics |
| GET | `/api/health` | Health check |

## Database Schema

### ChromaDB Collections

1. **`documents`** - Document metadata
   - `id`: string (UUID)
   - `title`: string
   - `source_path`: string
   - `doc_type`: string
   - `created_at`: timestamp
   - `metadata`: JSON object

2. **`chunks`** - Document chunks with embeddings
   - `id`: string (UUID)
   - `document_id`: string
   - `text`: string
   - `chunk_index`: integer
   - `start_char`: integer
   - `end_char`: integer
   - `embedding`: vector (384-dim)

### File Storage Structure

```
data/
├── vector_db/          # ChromaDB files
│   └── chroma.sqlite3
├── documents/          # Document metadata (JSON)
│   ├── doc_001.json
│   └── doc_002.json
└── files/             # Uploaded raw files
    ├── doc_001.pdf
    └── doc_002.docx
```

## Running the System

### Development Mode

```bash
# Terminal 1: Backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

### Production Mode

```bash
# Build frontend
cd frontend
npm run build

# Copy frontend build to backend
cp -r dist ../backend/static

# Run backend
cd ../backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

## CLI Usage (Standalone)

```bash
# Install CLI-only dependencies
pip install -r requirements-cli.txt

# Ingest documents
python -m docqa ingest /path/to/manuals/*.pdf

# Query
python -m docqa query "How do I reset the password on Model X?"

# List documents
python -m docqa list

# Delete document
python -m docqa delete <doc_id>
```
