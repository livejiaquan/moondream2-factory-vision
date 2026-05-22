@echo off
chcp 65001 >nul

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo         Run setup_windows.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo Starting Moondream demo...
echo (First run loads model ~15s. Two windows will appear: LIVE FEED + ANALYSIS)
echo.
echo Keys:  SPACE=pause   N=next clip   Q=quit
echo.
python demo.py --folder videos --max-width 512
pause
