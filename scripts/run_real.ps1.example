# Real semantic retrieval: local bge embed/rerank (GPU) + DashScope LLM  (ASCII only)
# Run:  powershell -ExecutionPolicy Bypass -File scripts\run_real.ps1
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
Set-Location $backend
$py = Join-Path $backend ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "venv not found. First run scripts\setup_real.ps1"
    exit 1
}

# ===== EDIT: your DashScope (Aliyun Bailian) API key =====
$env:LLM_API_KEY = "YOUR_DASHSCOPE_API_KEY_HERE"
$env:SECRET_KEY    = "dev-rag-secret-0123456789abcdef0123456789"
$env:LLM_MODEL   = "qwen-plus"

# HF mirror for downloading bge models in China
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_SYMLINKS = "1"
$env:HF_HUB_DISABLE_XET = "1"

$env:USE_REAL="true"
$env:EMBEDDING_PROVIDER="bge"
$env:RERANKER_PROVIDER="bge"
$env:LLM_PROVIDER="dashscope"
$env:EMBEDDING_DEVICE="cuda"
$env:DOCLING_DEVICE="cuda"   # docling heron/table models on GPU
$env:RERANKER_DEVICE="cuda"
$env:VECTOR_STORE="inmemory"

Write-Host "Starting real semantic retrieval: bge embed/rerank (GPU) + qwen-plus"
& $py -m uvicorn app.main:app --host 0.0.0.0 --port 8000
