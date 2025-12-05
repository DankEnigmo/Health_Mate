#!/bin/bash

echo "================================================"
echo "   Health_Mate - Environment Setup Script"
echo "================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed or not in PATH"
    echo "        Please install Python 3.8+ and try again."
    exit 1
fi

echo "[INFO] Python found:"
python3 --version
echo ""

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "[WARNING] Node.js is not installed. Frontend dependencies will not be installed."
    echo "          Install Node.js from https://nodejs.org/ if you need the frontend."
    NODE_AVAILABLE=0
else
    echo "[INFO] Node.js found:"
    node --version
    NODE_AVAILABLE=1
fi
echo ""

# Remove old venv if exists
if [ -d "venv" ]; then
    echo "[INFO] Removing existing virtual environment..."
    rm -rf venv
fi

echo "[1/5] Creating Python virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create virtual environment"
    exit 1
fi
echo "      Done."
echo ""

echo "[2/5] Activating virtual environment..."
source venv/bin/activate
echo "      Done."
echo ""

echo "[3/5] Upgrading pip..."
pip install --upgrade pip
echo "      Done."
echo ""

echo "[4/5] Installing Python dependencies..."
echo "      This may take several minutes (PyTorch is large)..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[WARNING] Some packages may have failed to install."
    echo "          Check the output above for errors."
fi
echo "      Done."
echo ""

if [ $NODE_AVAILABLE -eq 1 ]; then
    echo "[5/5] Installing JavaScript dependencies for frontend..."
    cd HM_Frontend
    npm install
    cd ..
    echo "      Done."
else
    echo "[5/5] Skipping JavaScript dependencies (Node.js not available)"
fi

echo ""
echo "================================================"
echo "   Setup Complete!"
echo "================================================"
echo ""
echo "To activate the Python virtual environment:"
echo "    source venv/bin/activate"
echo ""
echo "To run the Fall Detection backend:"
echo "    cd Fall_Detection"
echo "    python api_server.py"
echo ""
echo "To run the Gaze Tracking API:"
echo "    cd gaze_tracking"
echo "    python gaze_api.py"
echo ""
echo "To run the frontend (requires Node.js):"
echo "    cd HM_Frontend"
echo "    npm run dev"
echo ""
echo "================================================"
