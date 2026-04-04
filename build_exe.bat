@echo off
title Build Standalone Executable
echo ============================================================
echo   BUILDING STANDALONE ACTIVITY DETECTION APP
echo ============================================================
echo.

echo [*] Installing PyInstaller...
pip install pyinstaller --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install PyInstaller.
    pause
    exit /b 1
)

echo [*] Building the executable with PyInstaller...
echo     This may take a few minutes as it packages the models and dependencies...

REM Use --onedir instead of --onefile so startup is fast instead of taking minutes to unpack
pyinstaller --name "ActivityDetection" --onedir --add-data "models;models" --add-data "yolov8n.pt;." --hidden-import="ultralytics" --hidden-import="mediapipe" --noconfirm main.py

if errorlevel 1 (
    echo [ERROR] PyInstaller failed during compilation.
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
pause
