@echo off
title Human Emotion Recognition AI - Desktop Software
echo =====================================================================
echo    Launching Human Emotion Recognition AI (Desktop Software)...
echo =====================================================================
cd /d "%~dp0"

if exist "dist\Human Emotion Recognition AI 1.0.0.exe" (
    echo Launching standalone executable...
    start "" "dist\Human Emotion Recognition AI 1.0.0.exe"
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
