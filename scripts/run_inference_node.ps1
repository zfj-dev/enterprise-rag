# Private inference node (bge embedding + reranker) on the GPU box, PRIVATE network only.
# Run on the GPU machine:  powershell -ExecutionPolicy Bypass -File scripts\run_inference_node.ps1
# First time setup:
#   cd inference_service
#   py -3.12 -m venv .venv
#   .\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
$ErrorActionPreference = "Stop"
$svc = Join-Path $PSScriptRoot "..\inference_service"
Set-Location $svc
$py = Join-Path $svc ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "node venv not found. Create it first (see comment above)."
    exit 1
}

# ===== EDIT: shared secret must match the orchestrator's *_API_KEY =====
$env:INFERENCE_TOKEN = "replace_with_inference_token"
$env:INFER_DEVICE = "cuda"
$env:INFER_CONCURRENCY = "4"
$env:EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
$env:RERANKER_MODEL = "BAAI/bge-reranker-large"

Write-Host "Starting private inference node on :9000 (PRIVATE network only)"
& $py -m uvicorn app:app --host 0.0.0.0 --port 9000
