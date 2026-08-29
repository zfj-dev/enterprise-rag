# One-time setup: Python 3.12 venv + deps + bge  (ASCII only)
# CPU mode works out of the box; GPU (CUDA torch) is optional, see note below.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\setup_real.ps1
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
Set-Location $backend

# torch cuXXX needs Python <=3.12 (default python here may be 3.14)
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'py' launcher not found. Install Python 3.12 (python.org)."
    exit 1
}
Write-Host "Removing old venv..."
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
& py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: 'py -3.12' failed. Install Python 3.12 first."
    exit 1
}

$py = Join-Path $backend ".venv\Scripts\python.exe"
$env:HF_HUB_DISABLE_SYMLINKS = "1"   # Windows: copy instead of symlink in HF cache
$env:HF_HUB_DISABLE_XET = "1"        # mirror has no Xet/CAS backend, use plain HTTP
Write-Host ("venv python: " + (& $py --version))
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# NOTE: on Windows, PyPI 'torch' is the CPU build. For GPU, after setup run:
#   powershell -ExecutionPolicy Bypass -File scripts\enable_gpu.ps1
#   (installs torch 2.13.0+cu130 from Aliyun pytorch-wheels mirror, CN-accessible)
& $py -m pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
& $py -m pip install sentence-transformers transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host ""
Write-Host "SETUP OK (Python 3.12 + bge, CPU mode). Edit scripts\run_real.ps1 key, then run it."
Write-Host "GPU optional: install CUDA torch manually if you want faster embedding."
