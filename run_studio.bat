@echo off
title Piano Cover AI Studio
echo ======================================================================
echo             PIANO COVER AI STUDIO — MASTER ACOUSTIC ENGINE
echo ======================================================================
echo.

set PYTHON_CMD=python
if exist "..\backend\venv\Scripts\python.exe" (
    set PYTHON_CMD=..\backend\venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set PYTHON_CMD=venv\Scripts\python.exe
)

echo [1/2] Using Python: %PYTHON_CMD%
echo [2/2] Starting Studio Server on http://localhost:8055...
echo.

start "" "http://localhost:8055"
%PYTHON_CMD% main.py

pause
