# Agent link vs deterministic link on the same golden set (offline, no running service).
# ASCII only. Usage: powershell -ExecutionPolicy Bypass -File scripts\evaluate_agent.ps1
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
$py = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv not found."; exit 1 }
Set-Location $backend
& $py evaluate_agent.py
Write-Host ""
Write-Host "Report: backend\logs\agent-vs-baseline.log"
