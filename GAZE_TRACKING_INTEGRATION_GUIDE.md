# Gaze Tracking Integration Guide

## Complete Integration Steps for Health Monitoring Frontend

This guide provides step-by-step instructions for integrating the optimized gaze tracking system with your React frontend.

---

## Part 1: Backend Setup

### Step 1: Install Python Dependencies

```bash
cd gaze_tracking
pip install mediapipe opencv-python numpy
```

### Step 2: Verify Backend Integration

The backend API server has been updated with the gaze tracking endpoint. Start the server:

```bash
cd Fall_Detection
python -m Fall_Detection.api_server
```

You should see in the logs:
```
INFO - Gaze tracking module loaded successfully
```

The endpoint will be available at: `ws://localhost:8000/ws/gaze-tracking`

### Step 3: Test Standalone (Optional)

Test the optimized tracker independently:

```bash
cd gaze_tracking
python test_optimized.py
```

Choose option 1 for performance test, 2 for calibration test, or 3 for real-time tracking.

---

## Part 2: Frontend Setup

### Step 1: Create the Gaze Tracking Hook

Create file: `HM_Frontend/src/hooks/use-gaze-tracking.ts`

This hook handles:
- WebSocket connection to backend
- Webcam access and frame capture
- Sending frames to backend
- Receiving gaze coordinates
- Calibration management

### Step 2: Update OnScreenKeyboard Component

The OnScreenKeyboard component needs these additions:
- Accept `gazeCoords` prop
- Display gaze cursor
- Implement dwell-time selection
- Show visual feedback during dwell

### Step 3: Create Calibration Component

Create file: `HM_Frontend/src/components/patient/GazeCalibration.tsx`

This component:
- Shows 9-point calibration grid
- Guides user through calibration
- Collects samples at each point
- Shows progress

### Step 4: Update PatientMessageInput Page

Integrate everything in the PatientMessageInput page.

---

## Part 3: Configuration

### Backend Configuration

Edit `Fall_Detection/.env` or create if it doesn't exist:

```env
# Gaze Tracking Settings
GAZE_TRACKING_ENABLED=true
GAZE_TRACKING_FPS_LIMIT=30
GAZE_TRACKING_SMOOTHING=0.40
GAZE_TRACKING_PREPROCESSING=true
```

### Frontend Configuration

Edit `HM_Frontend/.env.local`:

```env
VITE_BACKEND_API_URL=http://localhost:8000
VITE_GAZE_TRACKING_ENABLED=true
VITE_GAZE_TRACKING_FPS=15
VITE_GAZE_DWELL_TIME=800
```

---

## Testing the Integration

### 1. Start Backend
```bash
cd Fall_Detection
python -m Fall_Detection.api_server
```

### 2. Start Frontend
```bash
cd HM_Frontend
npm run dev
```

### 3. Test Flow
1. Navigate to Patient Message Input page
2. Enable gaze tracking
3. Complete 9-point calibration
4. Start typing with your eyes!

---

## Performance Tips

### Optimize Frame Rate
- Frontend: Send 15-20 FPS (configurable)
- Backend: Process up to 30 FPS
- Lower for slower devices

### Reduce Latency
- Use JPEG quality 0.7-0.8
- Resize frames to 640x480 before sending
- Deploy backend and frontend on same machine

### Improve Accuracy
- Good lighting (avoid backlighting)
- Camera at eye level
- 30-60cm from camera
- Recalibrate if you move

---

## Troubleshooting

See `gaze_tracking/README.md` for detailed troubleshooting guide.

---

## What's Next?

After integration, you can:
1. Save calibration profiles per user
2. Add voice feedback
3. Implement gesture controls (blink detection)
4. Add accessibility shortcuts
5. Create training mode for new users
