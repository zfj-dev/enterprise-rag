# End-to-end self-test against the running service (localhost:8000).  (ASCII only)
# Prereq: run_real.ps1 must be running. Then:  powershell -ExecutionPolicy Bypass -File scripts\selftest.ps1
$py = Join-Path $PSScriptRoot "..\backend\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "ERROR: venv python not found. Run scripts\setup_real.ps1 first."
    exit 1
}
& $py (Join-Path $PSScriptRoot "..\backend\selftest.py")
$report = Join-Path $PSScriptRoot "..\backend\logs\selftest-report.log"
Write-Host "Report file: $report"
