# start.ps1 - ControlPlane.ai One-Command Startup
# Usage: .\start.ps1
Write-Host '==========================================' -ForegroundColor Cyan
Write-Host '  ControlPlane.ai - Professional Demo    ' -ForegroundColor Cyan
Write-Host '==========================================' -ForegroundColor Cyan

$mlBase = "$PSScriptRoot\ml\artifacts"
$modelsActive = 0

$models = @{
    'CONTROLPLANE_MODEL_INJECTION'        = "$mlBase\injection-v1\model"
    'CONTROLPLANE_MODEL_SAFETY'           = "$mlBase\toxicity-v1\model"
    'CONTROLPLANE_MODEL_FAIRNESS'         = "$mlBase\fairness-v1\model"
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

Write-Host "
[3/4] Starting Backend (FastAPI on :8000)..." -ForegroundColor Yellow
$backend = Start-Process -NoNewWindow -PassThru -FilePath 'python' `
    -ArgumentList '-m uvicorn backend.main:app --host 127.0.0.1 --port 8000' `
    -WorkingDirectory $PSScriptRoot
Start-Sleep -Seconds 4

Write-Host "[4/4] Starting Frontend (Streamlit on :8501)..." -ForegroundColor Yellow
$frontend = Start-Process -NoNewWindow -PassThru -FilePath 'python' `
    -ArgumentList '-m streamlit run frontend/streamlit_app.py --server.port 8501 --server.headless true' `
    -WorkingDirectory $PSScriptRoot
Start-Sleep -Seconds 3

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

Wait-Process -Id $backend.Id, $frontend.Id
