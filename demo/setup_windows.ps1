# Moondream Demo - Windows Setup
# Run in PowerShell: .\setup_windows.ps1
# If blocked by execution policy, run first:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Moondream Behavior Detection Demo - Windows Setup"         -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Check Python ──────────────────────────────────────────────
try {
    $pyver = python --version 2>&1
    Write-Host "[OK] $pyver" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found." -ForegroundColor Red
    Write-Host "        Install Python 3.10+ from https://python.org"
    Write-Host "        Make sure to check 'Add Python to PATH' during install."
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# ── Virtual environment ───────────────────────────────────────
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "[SKIP] .venv already exists." -ForegroundColor Yellow
} else {
    Write-Host "[1/4] Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}
& ".venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip --quiet

# ── PyTorch ───────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/4] PyTorch installation" -ForegroundColor Cyan
Write-Host ""
Write-Host "      Do you have an NVIDIA GPU?"
Write-Host "        Y = PyTorch + CUDA 12.1  (recommended, ~10x faster)"
Write-Host "        N = PyTorch CPU-only      (slow: ~30-60s per frame)"
Write-Host ""
$gpuChoice = Read-Host "Your choice [Y/N]"

if ($gpuChoice -match "^[Yy]$") {
    Write-Host "Installing PyTorch + CUDA 12.1..." -ForegroundColor Cyan
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "Installing PyTorch CPU..." -ForegroundColor Cyan
    pip install torch torchvision
}

# ── Other dependencies ────────────────────────────────────────
Write-Host ""
Write-Host "[3/4] Installing other dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet

# ── Model pre-download ────────────────────────────────────────
Write-Host ""
Write-Host "[4/4] Pre-download Moondream2 model? (~2 GB, saves time on first run)"
$dlChoice = Read-Host "Download now? [Y/N]"

if ($dlChoice -match "^[Yy]$") {
    Write-Host "Downloading vikhyatk/moondream2 (revision 2025-01-09)..." -ForegroundColor Cyan
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('vikhyatk/moondream2', revision='2025-01-09', trust_remote_code=True); print('[OK] tokenizer')"
    python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('vikhyatk/moondream2', revision='2025-01-09', trust_remote_code=True); print('[OK] model')"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the demo:"
Write-Host "  .\run_demo.ps1"
Write-Host ""
Read-Host "Press Enter to exit"
