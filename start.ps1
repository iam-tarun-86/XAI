Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "  Starting Wine Classification XAI Dashboard Suite" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Start backend Flask API
Write-Host "[1/2] Launching Flask API Backend..."
$backendProcess = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "server.py" -NoNewWindow -PassThru -ErrorAction SilentlyContinue

if (-not $backendProcess) {
    Write-Host "Error: Could not start Flask backend. Make sure .venv is initialized." -ForegroundColor Red
    exit 1
}

# 2. Start frontend React app
Write-Host "[2/2] Launching React Frontend (Vite)..."
Set-Location frontend
$frontendProcess = Start-Process -FilePath "npm.cmd" -ArgumentList "run dev" -NoNewWindow -PassThru -ErrorAction SilentlyContinue
Set-Location ..

if (-not $frontendProcess) {
    Write-Host "Error: Could not start Vite frontend. Make sure Node.js is installed." -ForegroundColor Red
    Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host ""
Write-Host "====================================================" -ForegroundColor Green
Write-Host "  Servers launched successfully!" -ForegroundColor Green
Write-Host "  - Backend: http://localhost:5000" -ForegroundColor Green
Write-Host "  - Frontend: http://localhost:5173" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop both servers." -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
Write-Host ""

# Block and capture Ctrl+C to stop both processes
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host "`nStopping servers..." -ForegroundColor Yellow
    if ($backendProcess) {
        Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($frontendProcess) {
        Stop-Process -Id $frontendProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
