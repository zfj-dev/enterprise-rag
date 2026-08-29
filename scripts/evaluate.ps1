# Run golden-set evaluation against the running service (localhost:8000).  (ASCII only)
# Prereq: run_real.ps1 must be running.
$py = Join-Path $PSScriptRoot "..\backend\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "ERROR: venv python not found. Run scripts\setup_real.ps1 first."
    exit 1
}
& $py (Join-Path $PSScriptRoot "..\backend\evaluate.py")
$report = Join-Path $PSScriptRoot "..\backend\logs\eval-report.log"
Write-Host "Report file: $report"
