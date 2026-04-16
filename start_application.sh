#!/bin/bash
set -e

# Repo root = directory containing this script (works on any machine / path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Document Q&A Application..."
echo "===================================="
echo "Project root: $SCRIPT_DIR"
echo ""

# Virtualenv lives under backend/ (see README: cd backend && python -m venv venv)
VENV_PY="$SCRIPT_DIR/backend/venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "❌ Backend venv not found at backend/venv/bin/python"
    echo "   Create it from README: cd backend && python -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Volcengine Ark: prefer backend/.env (loaded by FastAPI Settings when cwd is backend).
# Optional: export VOLCENGINE_API_KEY / VOLCENGINE_BASE_URL in your shell before running this script.
if [ -n "${VOLCENGINE_API_KEY:-}" ] || [ -n "${VOLCENGINE_BASE_URL:-}" ]; then
    echo "🔧 Shell env overrides (optional):"
    [ -n "${VOLCENGINE_API_KEY:-}" ] && echo "  - VOLCENGINE_API_KEY: set (${#VOLCENGINE_API_KEY} chars)"
    [ -n "${VOLCENGINE_BASE_URL:-}" ] && echo "  - VOLCENGINE_BASE_URL: $VOLCENGINE_BASE_URL"
    echo ""
fi

# Start Backend
echo "🚀 Starting Backend Server..."
cd backend
"$VENV_PY" -m uvicorn main:app --host 0.0.0.0 --port 8000 &
cd ..
BACKEND_PID=$!

echo "  Backend PID: $BACKEND_PID"
echo "  API URL: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""

# Wait for backend to start
sleep 3

# Start Frontend
echo "🚀 Starting Frontend Server..."
cd frontend
npm run dev &
cd ..
FRONTEND_PID=$!

echo "  Frontend PID: $FRONTEND_PID"
echo "  Web UI: http://localhost:5173"
echo ""

# Check if services are running
sleep 2
echo "🔍 Checking service status..."

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✅ Backend: Running"
else
    echo "  ⚠️  Backend: Starting... (may take a moment)"
fi

if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "  ✅ Frontend: Running"
else
    echo "  ⚠️  Frontend: Starting... (may take a moment)"
fi

echo ""
echo "===================================="
echo "✨ Application Started Successfully!"
echo ""
echo "📱 Access Points:"
echo "   • Web UI:       http://localhost:5173"
echo "   • API:          http://localhost:8000"
echo "   • API Docs:     http://localhost:8000/docs"
echo ""
echo "⚠️  Note: Set VOLCENGINE_API_KEY / VOLCENGINE_BASE_URL / LLM_MODEL in backend/.env — see backend/config.py."
echo ""
echo "Press Ctrl+C to stop all servers"
echo "===================================="

# Keep script running
wait
