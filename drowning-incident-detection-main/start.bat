@echo off
title Drowning Incident Detection System
cd /d "%~dp0"

echo ===================================================
echo     Drowning Incident Detection System Launcher
echo ===================================================
echo.

echo [1/2] Starting Docker Desktop in background...
start "" "C:\Users\User\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe"

echo [2/2] Launching Drowning Detection Application...
if exist venv\Scripts\activate (
    call venv\Scripts\activate
)
python run.py

pause
