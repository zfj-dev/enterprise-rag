# Run latency eval (segment buckets + concurrency P50/P95/TTFT) against a running service.
# Needs: service running on localhost:8000.
# NOTE: the service caps concurrent streams PER USER (MAX_CONCURRENT_STREAMS_PER_USER,
#       default 2). Raise it (e.g. 8) and restart before measuring 3-5 concurrency.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\evaluate_latency.ps1
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
$py = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv not found."; exit 1 }
Set-Location $backend
& $py evaluate_latency.py
Write-Host ""
Write-Host "Report: backend\logs\latency-report.log"
