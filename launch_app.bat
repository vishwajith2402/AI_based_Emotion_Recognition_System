@echo off
title EmotiSync - Multimodal Emotion Recognition Software
echo =====================================================================
echo    Launching EmotiSync (Multimodal Emotion Recognition Software)...
echo =====================================================================
cd /d "%~dp0"

if exist "dist\EmotiSync.exe" (
    echo Launching EmotiSync standalone executable...
    start "" "dist\EmotiSync.exe"
    exit /b 0
)

where npx >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Node.js not detected in PATH. Launching via Python Desktop Launcher...
    python launch_software.py
    exit /b 0
)

if not exist "node_modules\electron" (
    echo Installing software desktop runtime (one-time setup)...
    call npm install electron --save-dev
)

echo Starting EmotiSync desktop application window...
call npx electron .
