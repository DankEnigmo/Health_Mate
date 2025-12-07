#!/bin/bash

echo "========================================"
echo "Starting HealthMate Backend Services"
echo "========================================"
echo ""
echo "This will start TWO backend services:"
echo "  1. Fall Detection API (port 8000)"
echo "  2. Gaze Tracking API (port 8001)"
echo ""

# Store the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Activate virtual environment
echo "[1/5] Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: Virtual environment not found at venv/bin/activate"
    echo "Please run: python -m venv venv"
    echo "Then run: pip install -r requirements.txt"
    exit 1
fi

# Check if .env exists
echo "[2/5] Checking configuration..."
if [ ! -f "Fall_Detection/.env" ]; then
    echo ""
    echo "WARNING: Fall_Detection/.env not found!"
    echo "Please copy Fall_Detection/.env.example to Fall_Detection/.env"
    echo "and configure your settings."
    echo ""
    exit 1
fi

# Check if model weights exist
echo "[3/5] Checking model weights..."
if [ ! -f "Fall_Detection/yolov7-w6-pose.pt" ]; then
    echo ""
    echo "WARNING: YOLOv7-Pose model weights not found!"
    echo "Please download from:"
    echo "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt"
    echo "and place it in the Fall_Detection directory."
    echo ""
    exit 1
fi

# Start Fall Detection API in background
echo "[4/5] Starting Fall Detection API (port 8000)..."
cd "$ROOT_DIR/Fall_Detection"
python api_server.py > ../fall_detection.log 2>&1 &
FALL_PID=$!
echo "Fall Detection API started (PID: $FALL_PID)"

# Wait a moment
sleep 2

# Start Gaze Tracking API in background
echo "[5/5] Starting Gaze Tracking API (port 8001)..."
cd "$ROOT_DIR/gaze_tracking"
python gaze_server.py > ../gaze_tracking.log 2>&1 &
GAZE_PID=$!
echo "Gaze Tracking API started (PID: $GAZE_PID)"

cd "$ROOT_DIR"

echo ""
echo "========================================"
echo "Backend services are running!"
echo "========================================"
echo ""
echo "Fall Detection API:"
echo "  API Base:      http://localhost:8000"
echo "  API Docs:      http://localhost:8000/docs"
echo "  Health Check:  http://localhost:8000/health"
echo "  Video Stream:  http://localhost:8000/api/video/stream?patient_id=default"
echo "  WebSocket:     ws://localhost:8000/api/ws/alerts?patient_id=default"
echo ""
echo "Gaze Tracking API:"
echo "  API Base:      http://localhost:8001"
echo "  Health Check:  http://localhost:8001/health"
echo "  WebSocket:     ws://localhost:8001/ws/gaze"
echo ""
echo "Process IDs:"
echo "  Fall Detection: $FALL_PID"
echo "  Gaze Tracking:  $GAZE_PID"
echo ""
echo "Logs:"
echo "  Fall Detection: fall_detection.log"
echo "  Gaze Tracking:  gaze_tracking.log"
echo ""
echo "To stop services, run:"
echo "  kill $FALL_PID $GAZE_PID"
echo ""
