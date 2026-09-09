@echo off
title Multimodal Early Heart Attack XAI System Launcher
color 0B

echo =========================================================================
echo  CARDIO-FUSION XAI: Multimodal Early Heart Attack Risk System
echo =========================================================================
echo.
echo  [1/2] Starting Flask PyTorch Neural API Server (Port 5000)...
start "Flask API Server" cmd /k "python server.py"

echo  [2/2] Starting React Vite Frontend Console (Port 5173)...
cd frontend
start "React Vite Console" cmd /k "npm run dev"

echo.
echo =========================================================================
echo  System Initialized Successfully!
echo  API Endpoint:  http://localhost:5000/api/health
echo  Web Console:   http://localhost:5173
echo =========================================================================
echo.
pause
