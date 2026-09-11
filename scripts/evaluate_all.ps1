# One-page eval report: run every implemented metric and write a single report.
# Prereqs: service running on localhost:8000 for the generation + latency sections.
#          RGB data in backend\data\rgb\ for the RGB section (skipped loudly if absent).
# Usage:   powershell -ExecutionPolicy Bypass -File scripts\evaluate_all.ps1
# Skip:    $env:EVAL_SKIP="latency,rgb"   (section keys: generation / retrieval / rgb / latency)
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
$py = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv not found."; exit 1 }
Set-Location $backend
& $py evaluate_all.py
Write-Host ""
Write-Host "Report: backend\logs\eval-summary.log"
