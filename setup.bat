@echo off
title Activity Detection - Setup
echo ============================================================
echo   ACTIVITY DETECTION SYSTEM - SETUP
echo ============================================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
if exist venv (
    echo       Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)
python -m venv venv
if errorlevel 1 (
    echo [NOTE] venv creation had issues. Continuing with system Python...
    goto :install_torch
)
echo       Done.
echo.

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo       Done.
echo.

:install_torch
echo [3/5] Installing PyTorch with CUDA support...
echo       This may take several minutes...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
if errorlevel 1 (
    echo [WARNING] CUDA PyTorch install failed. Installing CPU version...
    pip install torch torchvision --quiet
)
echo       Done.
echo.

echo [4/5] Installing remaining dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo       Done.
echo.

echo [5/5] Downloading MediaPipe Pose model...
if not exist models mkdir models
if not exist models\pose_landmarker_full.task (
    python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task', 'models/pose_landmarker_full.task'); print('       Model downloaded.')"
) else (
    echo       Model already exists.
)
echo       Done.
echo.

echo ============================================================
echo   SETUP COMPLETE!
echo.
echo   To run the application:
echo     run.bat
echo ============================================================
echo.
pause
