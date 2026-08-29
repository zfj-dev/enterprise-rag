# Diagnose retrieval for a table query under current code. (ASCII only)
# Usage: powershell -ExecutionPolicy Bypass -File scripts\diagnose_retrieval.ps1
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
$py = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv not found."; exit 1 }
$env:USE_REAL="true"; $env:EMBEDDING_PROVIDER="bge"; $env:RERANKER_PROVIDER="bge"
$env:EMBEDDING_DEVICE="cuda"; $env:RERANKER_DEVICE="cuda"; $env:VECTOR_STORE="inmemory"
$env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_SYMLINKS="1"; $env:HF_HUB_DISABLE_XET="1"
Set-Location $backend
& $py diagnose_retrieval.py
Write-Host ""
Write-Host "Report: backend\logs\retrieval-diagnose.log"
