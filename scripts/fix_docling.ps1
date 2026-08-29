# Fix Docling on Windows: install onnxruntime (needed by layout/table models). (ASCII only)
# Usage: powershell -ExecutionPolicy Bypass -File scripts\fix_docling.ps1
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
$py = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "venv not found. Run scripts\setup_real.ps1 first."
    exit 1
}
Write-Host "Installing onnxruntime (CPU) for docling layout/table models ..."
& $py -m pip install "onnxruntime>=1.17" -i https://pypi.tuna.tsinghua.edu.cn/simple
Write-Host ""
Write-Host "Done. Now re-run:  scripts\verify_docling.ps1"
