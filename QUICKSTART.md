# Quick Start Guide

## 1. Install Dependencies

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## 2. Configure Environment

```bash
# Copy the example environment file
cp backend/.env.example backend/.env

# Edit backend/.env and add your API key
# ANTHROPIC_API_KEY=your_key_here
```

## 3. Run the Application

### Terminal 1 - Backend
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload
```

### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
```

## 4. Access the Application

- Web UI: http://localhost:5173
- API Docs: http://localhost:8000/docs

## 5. Upload Documents

1. Go to http://localhost:5173
2. Click on "Documents" in the sidebar
3. Drag and drop PDF, DOCX, or TXT files
4. Wait for processing to complete
5. Go to "Chat" to ask questions

## API Usage

### Upload a document
```bash
curl -X POST "http://localhost:8000/api/documents/" \
  -F "file=@document.pdf"
```

### Ask a question
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```
