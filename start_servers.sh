#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PY="$SCRIPT_DIR/backend/venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "❌ backend/venv not found. Create with: cd backend && python -m venv venv"
    exit 1
fi

# Start Backend
cd backend
"$VENV_PY" -m uvicorn main:app --host 0.0.0.0 --port 8000 &
echo "Backend started on http://localhost:8000"
cd ..

# Start Frontend
cd frontend
npm run dev &
echo "Frontend started on http://localhost:5173"

echo ""
echo "Both servers are starting up..."
echo "- API Docs: http://localhost:8000/docs"
echo "- Web UI: http://localhost:5173"
