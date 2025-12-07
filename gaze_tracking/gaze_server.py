"""
Gaze Tracking API Server

FastAPI server for gaze tracking WebSocket endpoint.
Runs on port 8001 alongside the Fall Detection API (port 8000).
"""

import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from gaze_api import gaze_tracking_websocket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOG = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Gaze Tracking API",
    description="WebSocket API for real-time gaze tracking",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "gaze_tracking",
    }


@app.websocket("/ws/gaze")
async def websocket_gaze_endpoint(websocket: WebSocket):
    """WebSocket endpoint for gaze tracking."""
    await gaze_tracking_websocket(websocket)


if __name__ == "__main__":
    import uvicorn
    
    LOG.info("Starting Gaze Tracking API Server on port 8001...")
    
    uvicorn.run(
        "gaze_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
    )
