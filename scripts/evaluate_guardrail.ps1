# Compression quality guardrail: same golden set, compression off vs on (offline, no service).
# ASCII only. Usage: powershell -ExecutionPolicy Bypass -File scripts\evaluate_guardrail.ps1
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
$py = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv not found."; exit 1 }
Set-Location $backend
& $py evaluate_guardrail.py
Write-Host ""
Write-Host "Report: backend\logs\compression-guardrail.log"
