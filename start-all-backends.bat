@echo off
echo ========================================
echo Starting HealthMate Backend Services
echo ========================================
echo.
echo This will start TWO backend services:
echo   1. Fall Detection API (port 8000)
echo   2. Gaze Tracking API (port 8001)
echo.

REM Store the root directory
set ROOT_DIR=%~dp0
cd /d "%ROOT_DIR%"

REM Activate virtual environment
echo [1/5] Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found at venv\Scripts\activate.bat
    echo Please run: python -m venv venv
    echo Then run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if .env exists
echo [2/5] Checking configuration...
if not exist "Fall_Detection\.env" (
    echo.
    echo WARNING: Fall_Detection\.env not found!
    echo Please copy Fall_Detection\.env.example to Fall_Detection\.env
    echo and configure your settings.
    echo.
    pause
    exit /b 1
)

REM Check if model weights exist
echo [3/5] Checking model weights...
if not exist "Fall_Detection\yolov7-w6-pose.pt" (
    echo.
    echo WARNING: YOLOv7-Pose model weights not found!
    echo Please download from:
    echo https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
    echo and place it in the Fall_Detection directory.
    echo.
    pause
    exit /b 1
)

REM Start Fall Detection API in a new window
echo [4/5] Starting Fall Detection API (port 8000)...
start "Fall Detection API" cmd /k "cd /d "%ROOT_DIR%Fall_Detection" && python api_server.py"

REM Wait a moment
timeout /t 2 /nobreak > nul

REM Start Gaze Tracking API in a new window
echo [5/5] Starting Gaze Tracking API (port 8001)...
start "Gaze Tracking API" cmd /k "cd /d "%ROOT_DIR%gaze_tracking" && python gaze_server.py"

echo.
echo ========================================
echo Backend services are starting...
echo ========================================
echo.
echo Fall Detection API:
echo   API Base:      http://localhost:8000
echo   API Docs:      http://localhost:8000/docs
echo   Health Check:  http://localhost:8000/health
echo   Video Stream:  http://localhost:8000/api/video/stream?patient_id=default
echo   WebSocket:     ws://localhost:8000/api/ws/alerts?patient_id=default
echo.
echo Gaze Tracking API:
echo   API Base:      http://localhost:8001
echo   Health Check:  http://localhost:8001/health
echo   WebSocket:     ws://localhost:8001/ws/gaze
echo.
echo Press any key to close this window (services will keep running)
echo To stop services, close their respective windows
echo.
pause
