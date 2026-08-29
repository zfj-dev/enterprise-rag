# Verify Docling can parse a PDF and extract tables. (ASCII only - PS5.1 ANSI/GBK)
# Usage:
#   1) copy the paper PDF to backend\paper.pdf
#   2) powershell -ExecutionPolicy Bypass -File scripts\verify_docling.ps1
#   optional: pass a PDF path as argument, e.g. ...verify_docling.ps1 D:\papers\thesis.pdf
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
$py = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "venv not found. Run scripts\setup_real.ps1 first."
    exit 1
}
$env:PYTHONIOENCODING = "utf-8"
Set-Location $backend
if ($args.Count -gt 0) {
    & $py verify_docling.py $args[0]
} else {
    & $py verify_docling.py
}
Write-Host ""
Write-Host "Report: backend\logs\docling-verify.log"
