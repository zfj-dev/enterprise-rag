# Switch venv torch to CUDA build for NVIDIA GPU (bge embed/rerank on GPU). (ASCII only)
# Uses Aliyun pytorch-wheels mirror (download.pytorch.org is blocked in CN).
# Run:  powershell -ExecutionPolicy Bypass -File scripts\enable_gpu.ps1
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
Set-Location $backend
$py = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "venv not found. Run scripts\setup_real.ps1 first."
    exit 1
}

Write-Host "Step 1/3: uninstall CPU torch/torchvision ..."
& $py -m pip uninstall -y torch torchvision
if ($LASTEXITCODE -ne 0) { Write-Host "uninstall failed"; exit 1 }

Write-Host "Step 2/3: install torch 2.13.0+cu130 + torchvision 0.28.0+cu130 (Aliyun mirror, several GB, may take 10-30 min) ..."
& $py -m pip install `
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu130/ `
  -i https://pypi.tuna.tsinghua.edu.cn/simple `
  "torch==2.13.0+cu130" "torchvision==0.28.0+cu130"
if ($LASTEXITCODE -ne 0) { Write-Host "install failed"; exit 1 }

Write-Host "Step 3/3: verify CUDA ..."
& $py -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host ""
Write-Host "Done. Restart run_real.ps1 to use GPU (EMBEDDING_DEVICE=cuda)."
