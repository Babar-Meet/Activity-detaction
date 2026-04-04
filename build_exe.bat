@echo off
setlocal EnableExtensions

pushd "%~dp0" >nul

title Build Standalone Executable
echo ============================================================
echo   BUILDING STANDALONE ACTIVITY DETECTION APP
echo ============================================================
echo.

if not exist "main.py" (
    echo [ERROR] main.py not found. Please run from the project folder.
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

echo [*] Activating virtual environment...
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    popd >nul
    pause
    exit /b 1
)

echo [*] Installing PyInstaller...
python -m pip install pyinstaller
if errorlevel 1 (
    echo [ERROR] Failed to install PyInstaller.
    popd >nul
    pause
    exit /b 1
)

echo [*] Cleaning old build outputs...
if exist "build" rmdir /s /q "build"
if exist "dist\ActivityDetection" rmdir /s /q "dist\ActivityDetection"

echo [*] Building executable with PyInstaller...
echo     This may take a few minutes as it packages models and dependencies...

REM Use --onedir for fast startup and stable loading of packaged model files.
python -m PyInstaller --name "ActivityDetection" --onedir --add-data "models;models" --add-data "yolov8n.pt;." --hidden-import "ultralytics" --hidden-import "mediapipe" --noconfirm "main.py"

if errorlevel 1 (
    echo [ERROR] PyInstaller failed during compilation.
    popd >nul
    pause
    exit /b 1
)

if not exist "dist\ActivityDetection\ActivityDetection.exe" (
    echo [ERROR] Build finished but executable not found.
    popd >nul
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   BUILD COMPLETE!
echo   The executable is located at:
echo   dist\ActivityDetection\ActivityDetection.exe
echo.
echo   You can copy the entire "dist\ActivityDetection" folder
echo   and run the .exe anywhere on Windows.
echo ============================================================

popd >nul
pause
exit /b 0
