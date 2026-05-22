# Moondream Demo - Quick Launcher
# Run in PowerShell: .\run_demo.ps1
# If blocked by execution policy, run first:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[ERROR] Virtual environment not found. Run setup_windows.ps1 first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

& ".venv\Scripts\Activate.ps1"

Write-Host "Starting Moondream demo..." -ForegroundColor Cyan
Write-Host "(First run loads model ~15s. Two windows will appear: LIVE FEED + ANALYSIS)" -ForegroundColor Yellow
Write-Host "Keys: SPACE=pause  N=next clip  Q=quit" -ForegroundColor Yellow
Write-Host ""

python demo.py --folder videos --max-width 512
