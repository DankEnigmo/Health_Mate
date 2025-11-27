"""
Example: WebSocket Integration with Fall Detection

This example demonstrates how to integrate the WebSocketManager with the 
VideoStreamManager to broadcast fall alerts in real-time.

This is a reference implementation showing how the FastAPI server should
integrate these components.
"""

import asyncio
import logging
from websocket_manager import WebSocketManager
from stream_manager import VideoStreamManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

LOG = logging.getLogger(__name__)


async def example_integration():
    """
    Example showing how to integrate WebSocketManager with VideoStreamManager.
    """
    
    # 1. Create WebSocketManager instance
    ws_manager = WebSocketManager(heartbeat_interval=30)
    LOG.info("WebSocketManager created")
    
    # 2. Define fall detection callback
    async def on_fall_detected(fall_alert_data):
        """
        Callback function that gets called when a fall is detected.
        This broadcasts the fall alert to all connected WebSocket clients.
        
        Args:
            fall_alert_data: Dictionary containing fall alert information
        """
        LOG.info(f"Fall detected! Broadcasting to {ws_manager.get_connection_count()} clients")
        await ws_manager.broadcast_fall_alert(fall_alert_data)
    
    # 3. Create VideoStreamManager with the callback
    # Note: In a real implementation, you would initialize the detector, processor, model, etc.
    # This is just showing the integration pattern
    
    """
    video_manager = VideoStreamManager(
        source=0,  # or RTSP URL
        detector=fall_detector_instance,
        processor=openpifpaf_processor,
        model=openpifpaf_model,
        device=torch_device,
        on_fall_detected=on_fall_detected,  # Pass the callback
        patient_id="patient_123"  # Patient ID for tracking
    )
    """
    
    LOG.info("VideoStreamManager would be created with fall detection callback")
    
    # 4. In FastAPI, you would:
    # - Accept WebSocket connections and add them to ws_manager
    # - Stream video frames using video_manager.generate_frames()
    # - When a fall is detected, the callback automatically broadcasts to all clients
    
    LOG.info("Integration example complete")
    
    # Cleanup
    await ws_manager.shutdown()


# FastAPI endpoint example (pseudo-code)
"""
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse

app = FastAPI()
ws_manager = WebSocketManager()
video_manager = VideoStreamManager(..., on_fall_detected=ws_manager.broadcast_fall_alert)

@app.websocket("/api/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.get("/api/video/stream")
async def video_stream():
    return StreamingResponse(
        video_manager.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
"""


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_integration())
