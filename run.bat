@echo off
setlocal EnableExtensions

pushd "%~dp0" >nul

title Activity Detection - Live Demo
echo ============================================================
echo   ACTIVITY DETECTION SYSTEM - LIVE DEMO
echo ============================================================
echo.

if not exist "main.py" (
    echo [ERROR] main.py not found. Please run this script from the project folder.
    popd >nul
    pause
    exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Run setup.bat first.
    popd >nul
    pause
    exit /b 1
)

echo [*] Activating virtual environment...
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    popd >nul
    pause
    exit /b 1
)

if not exist "yolov8n.pt" (
    echo [ERROR] Missing file: yolov8n.pt
    echo Run setup.bat first.
    popd >nul
    pause
    exit /b 1
)

if not exist "models\pose_landmarker_full.task" (
    echo [ERROR] Missing file: models\pose_landmarker_full.task
    echo Run setup.bat first.
    popd >nul
    pause
    exit /b 1
)

python -c "import cv2, ultralytics, mediapipe" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required Python packages are missing in this environment.
    echo Please run setup.bat to install dependencies.
    popd >nul
    pause
    exit /b 1
)

echo [*] Starting Activity Detection...
echo [*] Press Q or ESC in the window to quit.
echo.

python "main.py"
set "APP_EXIT=%ERRORLEVEL%"

echo.
if not "%APP_EXIT%"=="0" (
    echo [ERROR] Application exited with code %APP_EXIT%.
) else (
    echo [*] Application closed.
)

popd >nul
pause
exit /b %APP_EXIT%
