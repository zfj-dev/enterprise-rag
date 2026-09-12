# Run golden-set eval against running service (localhost:8000). ASCII only.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\evaluate.ps1
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
$py = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv not found."; exit 1 }
Set-Location $backend
& $py evaluate.py
Write-Host ""
Write-Host "Report: backend\logs\eval-report.log"
