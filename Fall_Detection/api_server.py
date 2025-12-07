"""
FastAPI-based API Server for Fall Detection

This server provides:
- MJPEG video streaming with real-time fall detection overlays
- WebSocket endpoint for real-time fall alerts
- REST endpoints for stats and health checks
- Supabase integration for persisting fall events

Endpoints:
- GET  /health              - Health check
- GET  /api/stats           - Get current detection statistics
- GET  /api/video/stream    - MJPEG video stream with fall detection
- WS   /api/ws/alerts       - WebSocket for real-time fall alerts
- POST /api/video/source    - Configure video source (camera index or URL)
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from supabase import Client, create_client

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fall_core import FallDetectorMulti

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("fall_detection_api.log")],
)
LOG = logging.getLogger(__name__)


# =============================================================================
# Helper Functions for Environment Variables
# =============================================================================


def get_env_str(var_name: str, default: str) -> str:
    """Get string environment variable."""
    return os.getenv(var_name, default)


def get_env_int(var_name: str, default: int) -> int:
    """Get integer environment variable with fallback."""
    try:
        return int(os.getenv(var_name, str(default)))
    except ValueError:
        LOG.warning(f"{var_name} is not a valid int, using default {default}")
        return default


def get_env_float(var_name: str, default: float) -> float:
    """Get float environment variable with fallback."""
    try:
        return float(os.getenv(var_name, str(default)))
    except ValueError:
        LOG.warning(f"{var_name} is not a valid float, using default {default}")
        return default


# =============================================================================
# Configuration (All values from .env file)
# =============================================================================


@dataclass
class ServerConfig:
    """Server configuration from environment variables."""

    # Supabase
    supabase_url: str = field(default_factory=lambda: get_env_str("SUPABASE_URL", ""))
    supabase_key: str = field(
        default_factory=lambda: get_env_str("SUPABASE_SERVICE_KEY", "")
    )

    # Video source
    video_source: str = field(default_factory=lambda: get_env_str("VIDEO_SOURCE", "0"))

    # Server settings
    host: str = field(default_factory=lambda: get_env_str("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: get_env_int("API_PORT", 8000))

    # Model path
    model_path: str = field(
        default_factory=lambda: get_env_str("MODEL_PATH", "yolov7-w6-pose.pt")
    )

    # Fall detection parameters (all from .env)
    fps: int = field(default_factory=lambda: get_env_int("FPS", 30))
    window_size: int = field(default_factory=lambda: get_env_int("WINDOW_SIZE", 10))
    v_thresh: float = field(default_factory=lambda: get_env_float("V_THRESH", 60.0))
    dy_thresh: float = field(default_factory=lambda: get_env_float("DY_THRESH", 20.0))
    ar_thresh: float = field(
        default_factory=lambda: get_env_float("ASPECT_RATIO_THRESH", 0.35)
    )

    # Streaming settings
    jpeg_quality: int = field(default_factory=lambda: get_env_int("JPEG_QUALITY", 80))
    max_fps: int = field(default_factory=lambda: get_env_int("MAX_STREAM_FPS", 30))

    # Alert settings
    fall_cooldown_seconds: float = field(
        default_factory=lambda: get_env_float("FALL_COOLDOWN", 5.0)
    )


config = ServerConfig()

# Log configuration on startup
LOG.info("=== Fall Detection API Configuration ===")
LOG.info(f"Model Path: {config.model_path}")
LOG.info(f"Video Source: {config.video_source}")
LOG.info(f"Server: {config.host}:{config.port}")
LOG.info(f"FPS: {config.fps}, Window Size: {config.window_size}")
LOG.info(
    f"Thresholds - V: {config.v_thresh}, DY: {config.dy_thresh}, AR: {config.ar_thresh}"
)
LOG.info(f"Supabase configured: {bool(config.supabase_url)}")
LOG.info("=========================================")


# =============================================================================
# Global State
# =============================================================================


@dataclass
class DetectionState:
    """Global state for fall detection system."""

    # Detector instance
    detector: Optional[FallDetectorMulti] = None

    # Video capture
    video_capture: Optional[cv2.VideoCapture] = None
    video_source: int | str = 0

    # Current frame (thread-safe access)
    current_frame: Optional[Any] = None
    processed_frame: Optional[Any] = None
    frame_lock: threading.Lock = field(default_factory=threading.Lock)

    # Statistics
    frames_processed: int = 0
    fps_actual: float = 0.0
    last_frame_time: float = 0.0
    processing_latency: float = 0.0

    # Fall tracking
    total_falls: int = 0
    falls_by_person: Dict[str, int] = field(default_factory=dict)
    last_fall_time: Dict[str, float] = field(default_factory=dict)

    # WebSocket connections (stores tuples of (websocket, patient_id))
    ws_connections: Dict[WebSocket, str] = field(default_factory=dict)
    ws_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Processing state
    is_processing: bool = False
    should_stop: bool = False
    processing_thread: Optional[threading.Thread] = None

    # Supabase client
    supabase: Optional[Client] = None


state = DetectionState()


# =============================================================================
# Supabase Integration
# =============================================================================


def init_supabase() -> Optional[Client]:
    """Initialize Supabase client."""
    if not config.supabase_url or not config.supabase_key:
        LOG.warning(
            "Supabase credentials not configured. Fall events will not be persisted."
        )
        return None

    try:
        client = create_client(config.supabase_url, config.supabase_key)
        LOG.info("Supabase client initialized successfully")
        return client
    except Exception as e:
        LOG.error(f"Failed to initialize Supabase client: {e}")
        return None


async def store_fall_event(
    patient_id: str,
    person_tracking_id: int,
    fall_count: int,
    metadata: Optional[Dict] = None,
) -> Optional[str]:
    """
    Store a fall event in Supabase.

    Args:
        patient_id: The ID of the patient
        person_tracking_id: The tracking ID of the person who fell
        fall_count: Total fall count for this person
        metadata: Additional metadata (bounding box, confidence, etc.)

    Returns:
        The ID of the created event, or None on failure
    """
    if not state.supabase:
        LOG.warning("Supabase not configured, skipping event storage")
        return None

    try:
        event_data = {
            "patient_id": patient_id,
            "person_tracking_id": person_tracking_id,
            "fall_count": fall_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "status": "new",
        }

        result = state.supabase.table("fall_events").insert(event_data).execute()

        if result.data and len(result.data) > 0:
            event_record = result.data[0]
            if isinstance(event_record, dict):
                event_id = event_record.get("id")
                LOG.info(f"Fall event stored: {event_id}")
                return str(event_id) if event_id else None

        return None

    except Exception as e:
        LOG.error(f"Failed to store fall event: {e}")
        return None


# =============================================================================
# Video Processing
# =============================================================================


def init_video_capture(source: int | str = 0) -> Optional[cv2.VideoCapture]:
    """Initialize video capture from camera or file."""
    try:
        # Try to parse as integer (camera index)
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            LOG.error(f"Failed to open video source: {source}")
            return None

        # Set camera properties for LOW LATENCY
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, config.max_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
        
        # Additional low-latency settings
        if isinstance(source, int):  # For webcams
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Auto exposure can cause lag
        
        LOG.info(f"Video capture initialized: {source}")
        return cap

    except Exception as e:
        LOG.error(f"Error initializing video capture: {e}")
        return None


def init_detector() -> Optional[FallDetectorMulti]:
    """Initialize the fall detection model."""
    try:
        detector = FallDetectorMulti(
            model_path=config.model_path,
            window_size=config.window_size,
            fps=config.fps,
            v_thresh=config.v_thresh,
            ar_thresh=config.ar_thresh,
            dy_thresh=config.dy_thresh,
        )
        LOG.info(
            f"Fall detector initialized successfully with model: {config.model_path}"
        )
        return detector

    except Exception as e:
        LOG.error(f"Failed to initialize fall detector: {e}")
        return None


def process_frame_sync(frame) -> tuple[Any, list]:
    """
    Process a single frame synchronously.

    Returns:
        Tuple of (processed_frame, fall_events)
    """
    if state.detector is None:
        return frame, []

    start_time = time.time()

    # Process frame with fall detection
    processed_frame = state.detector.process_frame(frame)

    # Calculate processing latency
    state.processing_latency = (time.time() - start_time) * 1000  # ms

    # Check for falls in the results
    fall_events = []

    for tid, tracker in state.detector.trackers.items():
        if tracker.is_ready():
            is_fall, bbox, debug, tag = tracker.check_fall()

            if is_fall and tag:
                # Check cooldown
                current_time = time.time()
                last_fall = state.last_fall_time.get(tid, 0)

                if current_time - last_fall > config.fall_cooldown_seconds:
                    state.last_fall_time[tid] = current_time
                    state.falls_by_person[tid] = state.falls_by_person.get(tid, 0) + 1
                    state.total_falls += 1

                    fall_events.append(
                        {
                            "person_tracking_id": int(tid),
                            "fall_count": state.falls_by_person[tid],
                            "tag": tag,
                            "debug": debug,
                            "bbox": bbox,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                    LOG.warning(f"FALL DETECTED - Person {tid}: {tag} ({debug})")

    return processed_frame, fall_events


def video_processing_loop():
    """Main video processing loop (runs in separate thread) - OPTIMIZED FOR LOW LATENCY."""
    LOG.info("Video processing loop started")

    fps_window = deque(maxlen=30)
    prev_time = time.time()
    frame_skip_counter = 0
    SKIP_FRAMES = 0  # Process every frame for best detection, 0 = no skip

    while not state.should_stop:
        if state.video_capture is None or not state.video_capture.isOpened():
            time.sleep(0.1)
            continue

        # Clear camera buffer to get latest frame (reduces latency)
        for _ in range(2):  # Grab 2 frames to clear buffer
            ret, frame = state.video_capture.read()
            if not ret:
                break

        if not ret:
            LOG.warning("Failed to read frame from video source")
            time.sleep(0.1)
            continue

        # Frame skipping for performance (if enabled)
        frame_skip_counter += 1
        if SKIP_FRAMES > 0 and frame_skip_counter % (SKIP_FRAMES + 1) != 0:
            continue

        # Store raw frame (no copy to save time)
        with state.frame_lock:
            state.current_frame = frame

        # Process frame
        processed_frame, fall_events = process_frame_sync(frame)

        # Store processed frame
        with state.frame_lock:
            state.processed_frame = processed_frame

        # Update statistics
        current_time = time.time()
        fps_window.append(1.0 / max(current_time - prev_time, 0.001))
        state.fps_actual = sum(fps_window) / len(fps_window)
        state.frames_processed += 1
        state.last_frame_time = current_time
        prev_time = current_time

        # Queue fall events for WebSocket broadcast
        if fall_events:
            asyncio.run(broadcast_fall_events(fall_events))

        # NO RATE LIMITING - Process as fast as possible for low latency

    LOG.info("Video processing loop stopped")


async def broadcast_fall_events(fall_events: list):
    """Broadcast fall events to all connected WebSocket clients."""
    if not fall_events:
        return

    async with state.ws_lock:
        disconnected: Set[WebSocket] = set()

        for ws, patient_id in state.ws_connections.items():
            for event in fall_events:
                try:
                    # Convert bounding box from [x1, y1, x2, y2] to {x, y, width, height}
                    bbox = event.get("bbox")
                    bounding_box = None
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        bounding_box = {
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                        }

                    message = {
                        "type": "fall_detected",
                        "data": {
                            "patient_id": patient_id,
                            "person_tracking_id": event["person_tracking_id"],
                            "fall_count": event["fall_count"],
                            "timestamp": event["timestamp"],
                            "metadata": {
                                "tag": event.get("tag"),
                                "debug": event.get("debug"),
                                "bounding_box": bounding_box,
                                "fps": state.fps_actual,
                            },
                        },
                    }

                    await ws.send_json(message)

                    # Store in Supabase
                    await store_fall_event(
                        patient_id=patient_id,
                        person_tracking_id=event["person_tracking_id"],
                        fall_count=event["fall_count"],
                        metadata=message["data"]["metadata"],
                    )

                except Exception as e:
                    LOG.error(f"Error broadcasting to WebSocket: {e}")
                    disconnected.add(ws)

        # Remove disconnected clients
        for ws in disconnected:
            state.ws_connections.pop(ws, None)


# =============================================================================
# MJPEG Streaming
# =============================================================================


async def generate_mjpeg_stream():
    """Generate MJPEG stream frames - OPTIMIZED FOR LOW LATENCY."""
    boundary = "frame"
    last_frame_time = 0.0
    min_frame_interval = 1.0 / config.max_fps  # Minimum time between frames

    while True:
        current_time = time.time()
        
        # Rate limiting - but non-blocking
        if current_time - last_frame_time < min_frame_interval:
            await asyncio.sleep(0.001)  # Very short sleep to yield control
            continue
        
        last_frame_time = current_time

        with state.frame_lock:
            frame = state.processed_frame

        if frame is None:
            # Send a blank frame if no video
            frame = create_blank_frame("Waiting for video...")

        # Encode frame as JPEG with optimized settings
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # Enable optimization
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0,  # Disable progressive (faster)
        ]
        _, jpeg = cv2.imencode(".jpg", frame, encode_params)

        # Yield MJPEG frame
        yield (
            (
                f"--{boundary}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg)}\r\n"
                f"\r\n"
            ).encode()
            + jpeg.tobytes()
            + b"\r\n"
        )


def create_blank_frame(message: str = "No Video") -> Any:
    """Create a blank frame with a message."""
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)  # Dark gray background

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 1, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2

    cv2.putText(frame, message, (text_x, text_y), font, 1, (255, 255, 255), 2)

    return frame


# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    LOG.info("Starting Fall Detection API Server...")

    # Initialize Supabase
    state.supabase = init_supabase()

    # Initialize detector
    state.detector = init_detector()

    # Initialize video capture (parse video source from string)
    video_source = (
        int(config.video_source)
        if config.video_source.isdigit()
        else config.video_source
    )
    state.video_capture = init_video_capture(video_source)
    state.video_source = video_source

    # Start processing thread
    state.should_stop = False
    state.is_processing = True
    state.processing_thread = threading.Thread(
        target=video_processing_loop, daemon=True
    )
    state.processing_thread.start()

    LOG.info("Fall Detection API Server started successfully")

    yield

    # Cleanup
    LOG.info("Shutting down Fall Detection API Server...")

    state.should_stop = True
    state.is_processing = False

    if state.processing_thread:
        state.processing_thread.join(timeout=5.0)

    if state.video_capture:
        state.video_capture.release()

    # Close all WebSocket connections
    async with state.ws_lock:
        for ws in state.ws_connections:
            try:
                await ws.close()
            except Exception:
                pass
        state.ws_connections.clear()

    LOG.info("Fall Detection API Server shutdown complete")


app = FastAPI(
    title="Fall Detection API",
    description="Real-time fall detection API with video streaming and WebSocket alerts",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REST Endpoints
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detector_loaded": state.detector is not None,
        "video_source_active": state.video_capture is not None
        and state.video_capture.isOpened(),
        "is_processing": state.is_processing,
        "supabase_connected": state.supabase is not None,
    }


@app.get("/api/stats")
async def get_stats():
    """Get current detection statistics."""
    return {
        "fps": round(state.fps_actual, 2),
        "latency": round(state.processing_latency, 2),
        "fall_count": state.total_falls,
        "falls_by_person": state.falls_by_person,
        "frames_processed": state.frames_processed,
        "is_processing": state.is_processing,
        "active_trackers": len(state.detector.trackers) if state.detector else 0,
        "video_source": str(state.video_source),
        "ws_connections": len(state.ws_connections),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/video/stream")
async def video_stream(patient_id: str = Query(default="default")):
    """
    MJPEG video stream endpoint.

    Args:
        patient_id: Optional patient ID for tracking purposes
    """
    LOG.info(f"Video stream requested for patient: {patient_id}")

    return StreamingResponse(
        generate_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/video/source")
async def set_video_source(
    source: str = Query(..., description="Camera index (0, 1, ...) or video URL/path"),
):
    """
    Configure the video source.

    Args:
        source: Camera index or video file path/URL
    """
    LOG.info(f"Changing video source to: {source}")

    # Parse source
    try:
        if source.isdigit():
            new_source: int | str = int(source)
        else:
            new_source = source
    except ValueError:
        new_source = source

    # Initialize new capture
    new_capture = init_video_capture(new_source)

    if new_capture is None:
        raise HTTPException(
            status_code=400, detail=f"Failed to open video source: {source}"
        )

    # Replace old capture
    old_capture = state.video_capture
    state.video_capture = new_capture
    state.video_source = new_source

    if old_capture:
        old_capture.release()

    # Reset detector trackers
    if state.detector:
        state.detector.trackers = {}
        state.detector.next_id = 1

    # Reset fall counters
    state.total_falls = 0
    state.falls_by_person = {}
    state.last_fall_time = {}

    return {
        "status": "success",
        "message": f"Video source changed to: {source}",
        "source": str(new_source),
    }


@app.post("/api/detector/reset")
async def reset_detector():
    """Reset the fall detector state."""
    if state.detector:
        state.detector.trackers = {}
        state.detector.next_id = 1

    state.total_falls = 0
    state.falls_by_person = {}
    state.last_fall_time = {}
    state.frames_processed = 0

    LOG.info("Detector state reset")

    return {"status": "success", "message": "Detector state reset successfully"}


@app.get("/api/config")
async def get_config():
    """Get current detection configuration."""
    return {
        "model_path": config.model_path,
        "video_source": config.video_source,
        "fps": config.fps,
        "window_size": config.window_size,
        "v_thresh": config.v_thresh,
        "dy_thresh": config.dy_thresh,
        "ar_thresh": config.ar_thresh,
        "fall_cooldown_seconds": config.fall_cooldown_seconds,
        "jpeg_quality": config.jpeg_quality,
        "max_stream_fps": config.max_fps,
    }


# =============================================================================
# WebSocket Endpoints
# =============================================================================


@app.websocket("/api/ws/alerts")
async def websocket_alerts(
    websocket: WebSocket, patient_id: str = Query(default="default")
):
    """
    WebSocket endpoint for real-time fall alerts.

    Args:
        patient_id: The patient ID to associate with this connection
    """
    await websocket.accept()

    # Store websocket with patient_id in the connections dict
    async with state.ws_lock:
        state.ws_connections[websocket] = patient_id

    LOG.info(
        f"WebSocket connected for patient: {patient_id} (total: {len(state.ws_connections)})"
    )

    try:
        # Send initial connection confirmation
        await websocket.send_json(
            {
                "type": "connected",
                "patient_id": patient_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages with timeout for heartbeat
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)

                # Handle different message types
                msg_type = message.get("type")

                if msg_type == "ping":
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                elif msg_type == "get_stats":
                    stats = {
                        "type": "stats",
                        "data": {
                            "fps": round(state.fps_actual, 2),
                            "latency": round(state.processing_latency, 2),
                            "fall_count": state.total_falls,
                            "is_processing": state.is_processing,
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await websocket.send_json(stats)

                elif msg_type == "subscribe":
                    # Client wants to subscribe to a specific patient
                    new_patient_id = message.get("patient_id", patient_id)
                    async with state.ws_lock:
                        state.ws_connections[websocket] = new_patient_id
                    await websocket.send_json(
                        {
                            "type": "subscribed",
                            "patient_id": new_patient_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json(
                        {
                            "type": "heartbeat",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "stats": {
                                "fps": round(state.fps_actual, 2),
                                "fall_count": state.total_falls,
                                "is_processing": state.is_processing,
                            },
                        }
                    )
                except Exception:
                    break

    except WebSocketDisconnect:
        LOG.info(f"WebSocket disconnected for patient: {patient_id}")
    except Exception as e:
        LOG.error(f"WebSocket error for patient {patient_id}: {e}")
    finally:
        async with state.ws_lock:
            state.ws_connections.pop(websocket, None)

        LOG.info(
            f"WebSocket removed for patient: {patient_id} (remaining: {len(state.ws_connections)})"
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    LOG.info(f"Starting server on {config.host}:{config.port}")

    uvicorn.run(
        "api_server:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
        access_log=True,
    )
