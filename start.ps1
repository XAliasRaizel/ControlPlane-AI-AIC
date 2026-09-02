# start.ps1 - ControlPlane.ai One-Command Startup
# Usage: .\start.ps1
Write-Host '==========================================' -ForegroundColor Cyan
Write-Host '  ControlPlane.ai - Professional Demo    ' -ForegroundColor Cyan
Write-Host '==========================================' -ForegroundColor Cyan

$mlBase = "$PSScriptRoot\ml\artifacts"
$modelsActive = 0

function Get-LatestModelPath($taskName, $v1Name) {
    $v4Path = "$mlBase\$taskName-v4\model"
    if (Test-Path $v4Path) { return $v4Path }
    $v3Path = "$mlBase\$taskName-v3\model"
    if (Test-Path $v3Path) { return $v3Path }
    $v2Path = "$mlBase\$taskName-v2\model"
    if (Test-Path $v2Path) { return $v2Path }
    return "$mlBase\$v1Name\model"
}

$models = @{
    'CONTROLPLANE_MODEL_INJECTION'        = (Get-LatestModelPath 'injection' 'injection-v1')
    'CONTROLPLANE_MODEL_SAFETY'           = (Get-LatestModelPath 'toxicity' 'toxicity-v1')
    'CONTROLPLANE_MODEL_FAIRNESS'         = (Get-LatestModelPath 'fairness' 'fairness-v1')
    'CONTROLPLANE_MODEL_GROUNDING'        = "$mlBase\grounding-nli-large\model"
    'CONTROLPLANE_MODEL_SENSITIVE_INTENT' = "$mlBase\sensitive-intent\model"
}

Write-Host "
[1/4] Checking ML model artifacts..." -ForegroundColor Yellow
foreach ($envVar in $models.Keys) {
    $path = $models[$envVar]
    if (Test-Path $path) {
        Set-Item "env:$envVar" $path
        Write-Host "  [OK] $envVar" -ForegroundColor Green
        $modelsActive++
    } else {
        Write-Host "  [--] $envVar (no artifact - regex fallback)" -ForegroundColor DarkGray
    }
}
Write-Host "  -> $modelsActive ML model(s) activated" -ForegroundColor Cyan

Write-Host "
[2/4] Enabling Session Accumulator..." -ForegroundColor Yellow
$env:CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED = 'true'
$calibPath = "$PSScriptRoot\ml\artifacts\session-accumulator\calibration.json"
if (Test-Path $calibPath) {
    $env:CONTROLPLANE_SESSION_ACCUMULATOR_CONFIG = $calibPath
    Write-Host '  [OK] Session Accumulator ON (calibrated)' -ForegroundColor Green
} else {
    Write-Host '  [OK] Session Accumulator ON (default config)' -ForegroundColor Green
}

if (-not $env:GROQ_API_KEY -and (Test-Path "$PSScriptRoot\.env")) {
    Get-Content "$PSScriptRoot\.env" | ForEach-Object {
        if ($_ -match '^([^#=]+)=(.*)$') {
            $k = $matches[1].Trim(); $v = $matches[2].Trim()
            if ($k -and -not [System.Environment]::GetEnvironmentVariable($k)) { Set-Item "env:$k" $v }
        }
    }
}

$pythonExe = 'python'
if (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe") {
    $pythonExe = "$PSScriptRoot\.venv\Scripts\python.exe"
} elseif (Test-Path "$PSScriptRoot\.ci_venv\Scripts\python.exe") {
    $pythonExe = "$PSScriptRoot\.ci_venv\Scripts\python.exe"
}

# Free ports 8000 and 8501 if held by leftover processes
function Stop-PortProcess([int]$port) {
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    foreach ($conn in $conns) {
        if ($conn.OwningProcess -and $conn.OwningProcess -ne $PID) {
            Write-Host "  Stopping leftover process on port $port (PID: $($conn.OwningProcess))..." -ForegroundColor DarkGray
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}
Stop-PortProcess 8000
Stop-PortProcess 8501

Write-Host "
[3/4] Starting Backend (FastAPI on :8000)..." -ForegroundColor Yellow
$backend = Start-Process -NoNewWindow -PassThru -FilePath $pythonExe `
    -ArgumentList '-m uvicorn backend.main:app --host 127.0.0.1 --port 8000' `
    -WorkingDirectory $PSScriptRoot

Write-Host "  Waiting for Backend to initialize and warm up..." -NoNewline
$maxAttempts = 30
$attempt = 0
$healthy = $false
while ($attempt -lt $maxAttempts -and -not $healthy) {
    Start-Sleep -Seconds 1
    $attempt++
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2 -ErrorAction Stop
        if ($resp.status -eq "ok" -or $resp.status -eq "healthy" -or $resp) {
            $healthy = $true
        }
    } catch {
        Write-Host "." -NoNewline
    }
}
if ($healthy) {
    Write-Host " [READY]" -ForegroundColor Green
} else {
    Write-Host " [TIMEOUT - Proceeding anyway]" -ForegroundColor Red
}

Write-Host "[4/4] Starting Frontend (Streamlit on :8501)..." -ForegroundColor Yellow
$frontend = Start-Process -NoNewWindow -PassThru -FilePath $pythonExe `
    -ArgumentList '-m streamlit run frontend/streamlit_app.py --server.port 8501 --server.headless true' `
    -WorkingDirectory $PSScriptRoot
Start-Sleep -Seconds 2

Write-Host '==========================================' -ForegroundColor Green
Write-Host '  ControlPlane.ai is RUNNING' -ForegroundColor Green
Write-Host '==========================================' -ForegroundColor Green
Write-Host '  Frontend:   http://localhost:8501'
Write-Host '  Backend:    http://localhost:8000'
Write-Host '  API Docs:   http://localhost:8000/docs'
Write-Host '  Health:     http://localhost:8000/health'
Write-Host "  ML Models Active:    $modelsActive / 5"
Write-Host '  Session Accumulator: ON (EWMA + Peak-Decay)'
Write-Host '  Press Ctrl+C to stop.'
Write-Host '==========================================' -ForegroundColor Green

try {
    Wait-Process -Id $backend.Id, $frontend.Id
} finally {
    if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
    if ($frontend -and -not $frontend.HasExited) { Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue }
}
