#!/bin/bash

echo "===================================================="
echo "  Starting Wine Classification XAI Dashboard Suite"
echo "===================================================="
echo

# 1. Launch Backend API
echo "[1/2] Launching Flask API Backend..."
# Check if running on Windows path or unix path for virtual environment activation
if [ -d ".venv/Scripts" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

python server.py &
BACKEND_PID=$!

# 2. Launch Frontend
echo "[2/2] Launching React Frontend (Vite)..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo
echo "===================================================="
echo "  Servers launched successfully!"
echo "  - Backend: http://localhost:5000"
echo "  - Frontend: http://localhost:5173"
echo "  Press Ctrl+C to stop both servers."
echo "===================================================="
echo

# Graceful cleanup on Ctrl+C
cleanup() {
    echo -e "\nShutting down servers (PIDs: Backend=$BACKEND_PID, Frontend=$FRONTEND_PID)..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for background jobs to finish
wait
