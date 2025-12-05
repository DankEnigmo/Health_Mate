@echo off
SETLOCAL EnableDelayedExpansion

echo ================================================
echo    Health_Mate - Environment Setup Script
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo         Please install Python 3.8+ and try again.
    exit /b 1
)

echo [INFO] Python found:
python --version
echo.

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Node.js is not installed. Frontend dependencies will not be installed.
    echo           Install Node.js from https://nodejs.org/ if you need the frontend.
    set NODE_AVAILABLE=0
) else (
    echo [INFO] Node.js found:
    node --version
    set NODE_AVAILABLE=1
)
echo.

REM Remove old venv if exists
if exist "venv" (
    echo [INFO] Removing existing virtual environment...
    rmdir /s /q venv
)

echo [1/5] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    exit /b 1
)
echo       Done.
echo.

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo       Done.
echo.

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip
echo       Done.
echo.

echo [4/5] Installing Python dependencies...
echo       This may take several minutes (PyTorch is large)...
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some packages may have failed to install.
    echo           Check the output above for errors.
)
echo       Done.
echo.

if !NODE_AVAILABLE!==1 (
    echo [5/5] Installing JavaScript dependencies for frontend...
    cd HM_Frontend
    call npm install
    cd ..
    echo       Done.
) else (
    echo [5/5] Skipping JavaScript dependencies (Node.js not available)
)

echo.
echo ================================================
echo    Setup Complete!
echo ================================================
echo.
echo To activate the Python virtual environment:
echo     venv\Scripts\activate
echo.
echo To run the Fall Detection backend:
echo     cd Fall_Detection
echo     python api_server.py
echo.
echo To run the Gaze Tracking API:
echo     cd gaze_tracking
echo     python gaze_api.py
echo.
echo To run the frontend (requires Node.js):
echo     cd HM_Frontend
echo     npm run dev
echo.
echo ================================================
ENDLOCAL
