@echo off
title Activity Detection - Live Demo
echo ============================================================
echo   ACTIVITY DETECTION SYSTEM - LIVE DEMO
echo ============================================================
echo.

REM Try to activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo [*] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [*] Using system Python...
)

echo [*] Starting Activity Detection...
echo [*] Press Q or ESC in the window to quit.
echo.

python main.py

echo.
echo [*] Application closed.
pause
