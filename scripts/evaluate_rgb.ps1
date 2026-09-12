# Run RGB Chinese four-ability eval (offline: builds runtime locally, no running service).
# Data: put zh.json / zh_int.json / zh_fact.json from https://github.com/chen700564/RGB
#       into backend\data\rgb\ first. Missing files are reported, never faked.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\evaluate_rgb.ps1
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
$py = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv not found."; exit 1 }
Set-Location $backend
& $py evaluate_rgb.py
Write-Host ""
Write-Host "Report: backend\logs\rgb-eval-report.log"
