@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Planet X ^& SPHEREx - NASA Space Apps Challenge 2026
echo   Team Outer Space
echo ============================================================
echo.

rem ------------------------------------------------------------
rem  1. Create the virtual environment on first run
rem ------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment...
  where py >nul 2>nul
  if errorlevel 1 (
    python -m venv .venv
  ) else (
    py -3 -m venv .venv
  )
  if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: Python 3 was not found.
    echo Install Python 3.11+ from https://www.python.org/downloads/
    echo and enable "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
  )
) else (
  echo [1/3] Virtual environment found.
)

rem ------------------------------------------------------------
rem  2. Install / check the pinned dependencies
rem ------------------------------------------------------------
echo [2/3] Installing dependencies (first run may take a few minutes)...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
  echo.
  echo ERROR: Could not install the dependencies.
  echo Check your internet connection and try again.
  echo.
  pause
  exit /b 1
)

rem ------------------------------------------------------------
rem  3. Start the server and open the browser
rem ------------------------------------------------------------
echo [3/3] Starting server at http://127.0.0.1:8000
echo.
echo Press Ctrl+C in this window to stop the server.
echo.

start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process 'http://127.0.0.1:8000'"

".venv\Scripts\python.exe" -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
