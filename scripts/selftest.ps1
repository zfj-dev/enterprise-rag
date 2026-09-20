# End-to-end self-test against the running service (localhost:8000).  (ASCII only)
# Prereq: run_real.ps1 must be running. Then:  powershell -ExecutionPolicy Bypass -File scripts\selftest.ps1
$py = Join-Path $PSScriptRoot "..\backend\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "ERROR: venv python not found. Run scripts\setup_real.ps1 first."
    exit 1
}
# Run from backend\ so settings read backend\.env (pydantic resolves .env against the CWD).
# ADMIN_PASSWORD / SECRET_KEY live there -- without this the self-test would fall back to the
# demo password and fail to log in against a real-mode service. Same as the other scripts.
Set-Location (Join-Path $PSScriptRoot "..\backend")
& $py (Join-Path $PSScriptRoot "..\backend\selftest.py")
$report = Join-Path $PSScriptRoot "..\backend\logs\selftest-report.log"
Write-Host "Report file: $report"
