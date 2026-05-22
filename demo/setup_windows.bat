@echo off
chcp 65001 >nul
echo ============================================================
echo  Moondream Behavior Detection Demo - Windows Setup
echo ============================================================
echo.

REM ── Check Python ─────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Install Python 3.10+ from https://python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found:
python --version
echo.

REM ── Create virtual environment ────────────────────────────────
if exist ".venv\" (
    echo [SKIP] .venv already exists, skipping creation.
) else (
    echo [1/4] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 ( echo [ERROR] venv creation failed. & pause & exit /b 1 )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

REM ── PyTorch (CUDA vs CPU) ─────────────────────────────────────
echo.
echo [2/4] PyTorch installation
echo.
echo       Do you have an NVIDIA GPU?
echo         Y = install PyTorch with CUDA 12.1  (recommended, ~10x faster)
echo         N = install PyTorch CPU-only         (slow: ~30-60s per frame)
echo.
set /p GPU_CHOICE="Your choice [Y/N]: "

if /I "%GPU_CHOICE%"=="Y" (
    echo Installing PyTorch + CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Installing PyTorch CPU...
    pip install torch torchvision
)
if errorlevel 1 ( echo [ERROR] PyTorch install failed. & pause & exit /b 1 )

REM ── Other dependencies ────────────────────────────────────────
echo.
echo [3/4] Installing other dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 ( echo [ERROR] pip install failed. & pause & exit /b 1 )

REM ── Model pre-download (optional) ────────────────────────────
echo.
echo [4/4] Pre-download Moondream2 model? (~2 GB, saves time on first run)
set /p DL_CHOICE="Download now? [Y/N]: "
if /I "%DL_CHOICE%"=="Y" (
    echo Downloading vikhyatk/moondream2 (revision 2025-01-09)...
    python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('vikhyatk/moondream2', revision='2025-01-09', trust_remote_code=True); print('[OK] tokenizer done')"
    python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('vikhyatk/moondream2', revision='2025-01-09', trust_remote_code=True); print('[OK] model done')"
)

echo.
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo To run the demo, double-click run_demo.bat
echo   OR open a terminal and run:
echo.
echo   .venv\Scripts\activate
echo   python demo.py --folder videos
echo.
pause
