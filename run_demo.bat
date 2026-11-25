@echo off
rem GapLink Live - double-click to start the demo console (Windows).
rem First run: creates .venv and installs the packages from requirements.txt (needs internet, Python 3.12).
setlocal
cd /d "%~dp0"
title GapLink Live

if exist ".venv\.installed" goto run

echo [setup] First run: creating .venv and installing packages (needs internet, about 5-15 minutes)...
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo [error] Python 3.12 was not found. Install it from https://www.python.org/downloads/ and run this file again.
    pause
    exit /b 1
)
rem An existing environment that already has the packages is used as is.
".venv\Scripts\python.exe" -c "import torch, cv2, fastapi, uvicorn, ultralytics, yaml, scipy" 2>nul && goto mark
rem Some environments (e.g. created by uv) have no pip: bootstrap it first.
".venv\Scripts\python.exe" -m pip --version >nul 2>nul || ".venv\Scripts\python.exe" -m ensurepip --upgrade
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [error] Package installation failed. Check the internet connection and run this file again.
    pause
    exit /b 1
)
:mark
echo ok> ".venv\.installed"

:run
if not exist "release\yolov8n.pt" echo [note] release\yolov8n.pt is missing; Ultralytics will try to download yolov8n.pt.
echo.
echo GapLink Live is starting at http://127.0.0.1:8001  (close this window to stop it)
echo Demo videos to upload: %CD%\demo\showcase
start "" cmd /c "timeout /t 10 /nobreak >nul & start http://127.0.0.1:8001"
".venv\Scripts\python.exe" -m demo.gaplink_live --fresh
echo.
echo GapLink Live has stopped.
pause
