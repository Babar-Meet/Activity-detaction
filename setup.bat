@echo off
setlocal EnableExtensions

pushd "%~dp0" >nul

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
    popd >nul
    pause
    exit /b 1
)

REM Validate Python major/minor version >= 3.10
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10+ is required.
    python --version
    popd >nul
    pause
    exit /b 1
)

REM Warn for newer Python versions where CUDA wheels may lag behind.
python -c "import sys; sys.exit(0 if sys.version_info < (3,13) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Python 3.13+ detected. CUDA wheels may be unavailable for some packages.
    echo [WARNING] For best GPU compatibility, prefer Python 3.10-3.12.
    echo.
)

echo [1/5] Creating virtual environment...
if exist "venv" (
    echo       Virtual environment already exists. Removing old one...
    rmdir /s /q "venv"
    if errorlevel 1 (
        echo [ERROR] Could not remove old virtual environment.
        popd >nul
        pause
        exit /b 1
    )
)

python -m venv "venv"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    popd >nul
    pause
    exit /b 1
)
echo       Done.
echo.

echo [2/5] Activating virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment activation script not found.
    popd >nul
    pause
    exit /b 1
)
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    popd >nul
    pause
    exit /b 1
)
echo       Done.
echo.

echo [3/5] Installing PyTorch with CUDA support...
echo       This may take several minutes...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Could not upgrade pip. Continuing...
)

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [WARNING] CUDA PyTorch install failed. Installing CPU version...
    python -m pip install torch torchvision
    if errorlevel 1 (
        echo [ERROR] Failed to install PyTorch.
        popd >nul
        pause
        exit /b 1
    )
)
echo       Verifying torch CUDA runtime...
python -c "import sys,torch; print('       torch=' + str(torch.__version__)); print('       torch.cuda=' + str(torch.version.cuda)); print('       cuda_available=' + str(torch.cuda.is_available())); print('       cuda_device_count=' + str(torch.cuda.device_count())); sys.exit(0 if (torch.version.cuda and torch.cuda.is_available() and torch.cuda.device_count() > 0) else 1)"
if errorlevel 1 (
    echo [WARNING] CUDA runtime is not active in this environment.
    echo [WARNING] App will fall back to CPU unless CUDA-enabled torch and NVIDIA drivers are available.
)
echo       Done.
echo.

echo [4/5] Installing remaining dependencies...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found.
    popd >nul
    pause
    exit /b 1
)

python -m pip install -r "requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    popd >nul
    pause
    exit /b 1
)
echo       Done.
echo.

echo [5/5] Ensuring model files are present...
if not exist "models" mkdir "models"

if not exist "models\pose_landmarker_full.task" (
    echo       Downloading MediaPipe pose model...
    python -c "import os,urllib.request; os.makedirs('models', exist_ok=True); urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task','models/pose_landmarker_full.task')"
    if errorlevel 1 (
        echo [ERROR] Failed to download pose_landmarker_full.task
        popd >nul
        pause
        exit /b 1
    )
) else (
    echo       pose_landmarker_full.task already exists.
)

if not exist "yolov8n.pt" (
    echo       Downloading YOLO model yolov8n.pt...
    python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt','yolov8n.pt')"
    if errorlevel 1 (
        echo [ERROR] Failed to download yolov8n.pt
        popd >nul
        pause
        exit /b 1
    )
) else (
    echo       yolov8n.pt already exists.
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

popd >nul
pause
exit /b 0
