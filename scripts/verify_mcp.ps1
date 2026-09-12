# Verify the MCP roundtrip for real: spawn a stdio server subprocess, list tools, call Calculator.
# Unit tests stub the transport on purpose; this script proves the wire actually works.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\verify_mcp.ps1
$ErrorActionPreference = "Stop"
$backend = Join-Path $PSScriptRoot "..\backend"
$py = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv not found."; exit 1 }
Set-Location $backend
& $py verify_mcp.py
Write-Host ""
Write-Host "Report: backend\logs\mcp-verify.log"
