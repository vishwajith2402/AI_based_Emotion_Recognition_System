@echo off
title EmotiX - Emotion Recognition System
echo =====================================================================
echo    Launching EmotiX (Multimodal Emotion Recognition Software)...
echo =====================================================================
cd /d "%~dp0"

if exist "dist\EmotiX.exe" (
    echo Launching EmotiX standalone software...
    start "" "dist\EmotiX.exe"
    exit /b 0
)
if exist "dist\EmotiX - Emotion Recognition System 1.0.0.exe" (
    echo Launching EmotiX standalone software...
    start "" "dist\EmotiX - Emotion Recognition System 1.0.0.exe"
    exit /b 0
)

where npx >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js and npm are required to run this software.
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

if not exist "node_modules\electron" (
    echo Installing software desktop runtime (one-time setup)...
    call npm install electron --save-dev
)

echo Starting desktop application window...
call npx electron .
