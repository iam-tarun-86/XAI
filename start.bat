@echo off
setlocal enabledelayedexpansion
title XAI Dashboard Launcher

echo ====================================================
echo   Wine Classification XAI Dashboard (Windows)
echo ====================================================
echo.

:: 1. Check Python virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [Setup] Creating Python virtual environment...
    python -m venv .venv
    echo [Setup] Installing Python dependencies...
    .venv\Scripts\pip install -r requirements.txt
)

:: 2. Check Frontend node_modules
if not exist "frontend\node_modules" (
    echo [Setup] Installing Frontend npm dependencies...
    cd frontend
    call npm install
    cd ..
)

echo [1/2] Launching Flask Backend (http://localhost:5000)...
echo [2/2] Launching React Frontend (http://localhost:5173)...
echo.
echo ====================================================
echo   Servers are running!
echo   - Frontend: http://localhost:5173
echo   - Backend:  http://localhost:5000
echo.
echo   [!] Closing this window or pressing Ctrl+C will
echo       automatically stop both servers.
echo ====================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "^
    $backend = Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList 'server.py' -PassThru -NoNewWindow; ^
    Set-Location frontend; ^
    $frontend = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c npm run dev' -PassThru -NoNewWindow; ^
    Set-Location ..; ^
    try { ^
        while ($true) { Start-Sleep -Seconds 1 } ^
    } finally { ^
        Write-Host '`n[Shutdown] Stopping servers...' -ForegroundColor Yellow; ^
        if ($backend) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }; ^
        if ($frontend) { Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue }; ^
        Get-Process -Name 'node', 'python' -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*SHAPELIME*' } | Stop-Process -Force -ErrorAction SilentlyContinue; ^
        Write-Host '[Shutdown] Servers stopped cleanly.' -ForegroundColor Green; ^
    }"

pause
