"""
FastAPI Server for Fall Detection Backend

This module provides HTTP and WebSocket endpoints for the fall detection system,
enabling real-time video streaming with fall detection overlays and real-time alerts.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from supabase import create_client, Client

# Relative imports for package structure
from .config_manager import ConfigManager, FallDetectionConfig
from .stream_manager import VideoStreamManager
from .websocket_manager import WebSocketManager
from .fall_detector import FallDetectionOrchestrator
from .medication_scheduler import MedicationScheduler
from . import network
from . import decoder

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fall_detection_api.log')
    ]
)

LOG = logging.getLogger(__name__)

# Import gaze tracking (with fallback if not available)
try:
    import os
    # Add gaze_tracking to path
    gaze_tracking_path = os.path.join(os.path.dirname(__file__), '..', 'gaze_tracking')
    if os.path.exists(gaze_tracking_path):
        sys.path.insert(0, gaze_tracking_path)
    from gaze_api import gaze_tracking_websocket
    GAZE_TRACKING_AVAILABLE = True
    LOG.info("Gaze tracking module loaded successfully")
except ImportError as e:
    LOG.warning(f"Gaze tracking not available: {e}")
    GAZE_TRACKING_AVAILABLE = False
    gaze_tracking_websocket = None

# Global state
config: Optional[FallDetectionConfig] = None
stream_manager: Optional[VideoStreamManager] = None
ws_manager: Optional[WebSocketManager] = None
supabase_client: Optional[Client] = None
medication_scheduler: Optional[MedicationScheduler] = None
fall_detector: Optional[FallDetectionOrchestrator] = None


async def initialize_fall_detection():
    """Initialize the fall detection system components."""
    global config, stream_manager, ws_manager, supabase_client, medication_scheduler, fall_detector
    
    # Initialize to None to prevent NameError later
    stream_manager = None
    fall_detector = None

    try:
        # 1. Load configuration (CRITICAL)
        LOG.info("Loading configuration...")
        config = ConfigManager.load_config()
        LOG.info("Configuration loaded successfully")
        
        # 2. Initialize Supabase client (CRITICAL)
        LOG.info("Initializing Supabase client...")
        supabase_client = create_client(config.supabase_url, config.supabase_key)
        LOG.info("Supabase client initialized")
        
        # 3. Initialize WebSocket manager (CRITICAL)
        LOG.info("Initializing WebSocket manager...")
        ws_manager = WebSocketManager(heartbeat_interval=30)
        LOG.info("WebSocket manager initialized")
        
        # 4. Initialize medication scheduler (CRITICAL)
        # We initialize this BEFORE the AI model so it works even if AI fails
        LOG.info("Initializing medication scheduler...")
        medication_scheduler = MedicationScheduler(supabase_client)
        await medication_scheduler.start()
        LOG.info("Medication scheduler initialized and started")
        
        # 5. Initialize AI & Video Stream (OPTIONAL / CAN FAIL GRACEFULLY)
        LOG.info("Attempting to load OpenPifPaf model...")
        try:
            device = torch.device('cuda' if config.enable_cuda and torch.cuda.is_available() else 'cpu')
            LOG.info(f"Using device: {device}")
            
            # Load model
            # This line was causing the crash due to the broken URL
            model, _ = network.factory(checkpoint=config.model_checkpoint)
            model = model.to(device)
            model.eval()
            LOG.info("OpenPifPaf model loaded successfully")
            
            # Create processor - only if model has head networks
            processor = None
            if hasattr(model, 'head_nets') and model.head_nets:
                LOG.info("Creating decoder processor with head networks...")
                processor = decoder.factory_decode(
                    model.head_nets,
                    basenet_stride=model.base_net.stride if hasattr(model, 'base_net') else 16
                )
                LOG.info("Processor created successfully")
            else:
                LOG.warning("Model does not have head networks. Processor will be None.")
                LOG.warning("This is expected for freshly-built untrained models.")
            
            # Initialize fall detection orchestrator
            LOG.info("Initializing fall detection orchestrator...")
            fall_detector = FallDetectionOrchestrator()
            LOG.info("Fall detection orchestrator initialized")
            
            # Define fall detection callback
            async def on_fall_detected(alert_data: dict):
                """Callback function when a fall is detected."""
                try:
                    # Broadcast via WebSocket
                    if ws_manager:
                        await ws_manager.broadcast_fall_alert(alert_data)
                    
                    # Store in Supabase
                    await store_fall_event(alert_data)
                    
                except Exception as e:
                    LOG.error(f"Error in fall detection callback: {e}", exc_info=True)
            
            # Initialize video stream manager
            LOG.info(f"Initializing video stream from source: {config.video_source}")
            stream_manager = VideoStreamManager(
                source=config.video_source,
                detector=fall_detector,
                processor=processor,
                model=model,
                device=device,
                scale=1.0,
                max_reconnect_attempts=5,
                on_fall_detected=on_fall_detected,
                patient_id="default_patient"
            )
            
            if stream_manager.is_connected:
                LOG.info("Video stream initialized successfully")
            else:
                LOG.warning("Video stream initialized but not connected")
                
        except Exception as ai_error:
            # Catches 404 errors, model loading errors, etc.
            LOG.error(f"CRITICAL: Failed to initialize AI Model or Video Stream: {ai_error}")
            LOG.warning("Server will continue running WITHOUT Fall Detection capabilities.")
            # We DO NOT raise here, allowing the server to continue running
        
        LOG.info("System initialization sequence completed")
        
    except Exception as e:
        # Only fatal errors (Config/DB/Auth) reach here
        LOG.error(f"Fatal error during system initialization: {e}", exc_info=True)
        raise


async def store_fall_event(alert_data: dict):
    """Store fall event in Supabase database."""
    if not supabase_client:
        LOG.warning("Supabase client not initialized, cannot store fall event")
        return

    try:
        # Prepare fall event record
        fall_event = {
            "patient_id": alert_data.get("patient_id"),
            "person_tracking_id": alert_data.get("person_tracking_id"),
            "fall_count": alert_data.get("fall_count"),
            "timestamp": alert_data.get("timestamp"),
            "metadata": alert_data.get("metadata", {}),
            "status": "new"
        }
        
        # Insert into database
        result = supabase_client.table("fall_events").insert(fall_event).execute()
        LOG.info(f"Fall event stored in database: {result.data}")
        
    except Exception as e:
        LOG.error(f"Failed to store fall event in database: {e}", exc_info=True)


async def shutdown_fall_detection():
    """Cleanup resources on shutdown."""
    global stream_manager, ws_manager, medication_scheduler
    
    LOG.info("Shutting down fall detection system...")
    
    try:
        if medication_scheduler:
            await medication_scheduler.stop()
            LOG.info("Medication scheduler stopped")
        
        if stream_manager:
            stream_manager.release()
            LOG.info("Video stream released")
        
        if ws_manager:
            await ws_manager.shutdown()
            LOG.info("WebSocket manager shut down")
        
        LOG.info("Fall detection system shut down successfully")
        
    except Exception as e:
        LOG.error(f"Error during shutdown: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    await initialize_fall_detection()
    yield
    # Shutdown
    await shutdown_fall_detection()


# Create FastAPI application
app = FastAPI(
    title="Fall Detection API",
    description="Real-time fall detection system with video streaming and alerts",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        is_healthy = (
            config is not None and
            ws_manager is not None and
            supabase_client is not None
        )
        
        video_status = stream_manager.is_connected if stream_manager else False
        ai_status = "running" if fall_detector else "failed_to_load"
        
        return JSONResponse(
            status_code=200 if is_healthy else 503,
            content={
                "status": "healthy" if is_healthy else "unhealthy",
                "video_stream_connected": video_status,
                "ai_system_status": ai_status,
                "websocket_connections": ws_manager.get_connection_count() if ws_manager else 0,
                "version": "1.0.0"
            }
        )
    except Exception as e:
        LOG.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/api/video/stream")
async def video_stream(patient_id: Optional[str] = None):
    """MJPEG video stream endpoint with fall detection overlays."""
    if stream_manager is None or not stream_manager.is_connected:
        return JSONResponse(
            status_code=503,
            content={"error": "Video stream not available (AI Model failed to load or Camera disconnected)"}
        )
    
    # Update the stream manager's patient_id if provided
    if patient_id and stream_manager:
        stream_manager.patient_id = patient_id
        LOG.info(f"Video stream requested for patient: {patient_id}")
    
    return StreamingResponse(
        stream_manager.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/api/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time fall alerts."""
    if ws_manager is None:
        await websocket.close(code=1011, reason="WebSocket manager not initialized")
        return
    
    await ws_manager.connect(websocket)
    
    try:
        await ws_manager.send_personal_message(
            '{"type": "connected", "message": "Connected to fall detection alerts"}',
            websocket
        )
        while True:
            try:
                data = await websocket.receive_text()
                LOG.debug(f"Received message from client: {data}")
            except WebSocketDisconnect:
                break
            except Exception as e:
                LOG.error(f"Error receiving WebSocket message: {e}")
                break
    finally:
        ws_manager.disconnect(websocket)


@app.get("/api/stats")
async def get_statistics():
    """Get current system statistics."""
    if stream_manager is None:
        return JSONResponse(
            status_code=200,
            content={
                "fps": 0,
                "latency": 0,
                "fall_count": 0,
                "status": "AI Model Failed to Load"
            }
        )
        
    try:
        stats = stream_manager.get_stats()
        return JSONResponse(status_code=200, content=stats)
    except Exception as e:
        LOG.error(f"Error getting statistics: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    if config is None:
        return JSONResponse(status_code=503, content={"error": "Configuration not loaded"})
    
    return JSONResponse(
        status_code=200,
        content={
            "video_source": str(config.video_source),
            "fall_threshold": config.fall_threshold,
            "enable_cuda": config.enable_cuda,
            "host": config.host,
            "port": config.port
        }
    )


@app.websocket("/ws/gaze-tracking")
async def websocket_gaze_endpoint(websocket: WebSocket):
    """WebSocket endpoint for gaze tracking."""
    if not GAZE_TRACKING_AVAILABLE or gaze_tracking_websocket is None:
        await websocket.close(code=1011, reason="Gaze tracking not available")
        LOG.warning("Gaze tracking WebSocket request rejected - module not available")
        return
    
    LOG.info("Gaze tracking WebSocket connection request received")
    await gaze_tracking_websocket(websocket)


if __name__ == "__main__":
    # Run the server
    # Note: using 'Fall_Detection.api_server:app' string to help Uvicorn find the app in the module
    try:
        cfg = ConfigManager.load_config()
        uvicorn.run(
            "Fall_Detection.api_server:app",
            host=cfg.host,
            port=cfg.port,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        LOG.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)