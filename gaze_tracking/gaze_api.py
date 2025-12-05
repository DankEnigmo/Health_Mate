"""
FastAPI WebSocket endpoint for gaze tracking.

Optimized for frontend integration with:
- Efficient base64 decoding
- Frame rate limiting
- Connection management
- Error recovery
- Calibration persistence
"""

import asyncio
import base64
import logging
import time
from typing import Optional, Dict
import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
import json

from gaze_tracker_optimized import GazeTrackerOptimized, TrackingQuality

LOG = logging.getLogger(__name__)


class GazeTrackingService:
    """
    Service for managing gaze tracking sessions.
    """
    
    def __init__(
        self,
        frame_rate_limit: int = 30,
        calibration_samples_per_point: int = 10
    ):
        """
        Initialize gaze tracking service.
        
        Args:
            frame_rate_limit: Maximum frames per second to process
            calibration_samples_per_point: Number of samples per calibration point
        """
        self.tracker = GazeTrackerOptimized(
            static_image_mode=False,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smoothing_alpha=0.40,
            max_frames_lost=15,
            enable_preprocessing=True
        )
        
        self.frame_rate_limit = frame_rate_limit
        self.min_frame_interval = 1.0 / frame_rate_limit
        self.last_frame_time = 0.0
        
        self.calibration_samples_per_point = calibration_samples_per_point
        self.current_calibration_samples: Dict[str, int] = {}
        
        self.frames_processed = 0
        self.frames_dropped = 0
        
        LOG.info(f"GazeTrackingService initialized (max {frame_rate_limit} FPS)")
    
    def decode_frame(self, base64_data: str) -> Optional[np.ndarray]:
        """
        Decode base64 image to numpy array.
        
        Args:
            base64_data: Base64 encoded image (optionally with data URL prefix)
            
        Returns:
            Decoded BGR frame or None on error
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',', 1)[1]
            
            # Decode base64 to bytes
            img_bytes = base64.b64decode(base64_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(img_bytes, np.uint8)
            
            # Decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                LOG.warning("Failed to decode frame")
                return None
            
            return frame
            
        except Exception as e:
            LOG.error(f"Error decoding frame: {e}")
            return None
    
    def should_process_frame(self) -> bool:
        """
        Check if enough time has passed to process next frame.
        
        Returns:
            True if frame should be processed
        """
        current_time = time.time()
        if current_time - self.last_frame_time >= self.min_frame_interval:
            self.last_frame_time = current_time
            return True
        
        self.frames_dropped += 1
        return False
    
    def add_calibration_sample(
        self, 
        frame: np.ndarray, 
        target_x: float, 
        target_y: float
    ) -> Dict[str, any]:
        """
        Add a calibration sample.
        
        Args:
            frame: Input BGR frame
            target_x: Target x coordinate (0.0-1.0)
            target_y: Target y coordinate (0.0-1.0)
            
        Returns:
            Dictionary with calibration status
        """
        # Create unique key for this target
        target_key = f"{target_x:.2f}_{target_y:.2f}"
        
        # Extract features
        features, _ = self.tracker.extract_features(frame)
        
        if features is None:
            return {
                "success": False,
                "error": "No face detected",
                "samples_collected": 0
            }
        
        # Add calibration point
        self.tracker.add_calibration_point(features, (target_x, target_y))
        
        # Track samples for this target
        if target_key not in self.current_calibration_samples:
            self.current_calibration_samples[target_key] = 0
        
        self.current_calibration_samples[target_key] += 1
        
        samples = self.current_calibration_samples[target_key]
        stats = self.tracker.get_statistics()
        
        return {
            "success": True,
            "samples_collected": samples,
            "total_calibration_points": stats["calibration_points"],
            "is_calibrated": stats["is_calibrated"],
            "target": {"x": target_x, "y": target_y}
        }
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, any]]:
        """
        Process frame and extract gaze data.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Dictionary with gaze data or None
        """
        # Rate limiting
        if not self.should_process_frame():
            return None
        
        self.frames_processed += 1
        
        # Get gaze
        result = self.tracker.get_gaze(frame)
        
        if result is None:
            return {
                "success": False,
                "error": "Face tracking lost",
                "x": None,
                "y": None
            }
        
        return {
            "success": True,
            "x": result.x,
            "y": result.y,
            "confidence": result.confidence,
            "quality": result.quality.value,
            "iris_centers": {
                "left": {"x": result.iris_centers['left'][0], "y": result.iris_centers['left'][1]},
                "right": {"x": result.iris_centers['right'][0], "y": result.iris_centers['right'][1]},
                "mid": {"x": result.iris_centers['mid'][0], "y": result.iris_centers['mid'][1]}
            }
        }
    
    def reset_calibration(self):
        """Reset calibration data."""
        self.tracker.reset_calibration()
        self.current_calibration_samples = {}
        LOG.info("Calibration reset")
    
    def get_stats(self) -> Dict[str, any]:
        """Get service statistics."""
        tracker_stats = self.tracker.get_statistics()
        
        return {
            **tracker_stats,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "effective_fps": self.frames_processed / max(1, time.time() - (self.last_frame_time - 60))
        }


async def gaze_tracking_websocket(websocket: WebSocket):
    """
    WebSocket endpoint handler for gaze tracking.
    
    Args:
        websocket: FastAPI WebSocket connection
    """
    await websocket.accept()
    
    # Create service instance for this connection
    service = GazeTrackingService(frame_rate_limit=30)
    
    LOG.info("Gaze tracking WebSocket connection established")
    
    try:
        while True:
            # Receive message from client
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                message = json.loads(data)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": time.time()
                })
                continue
            except Exception as e:
                LOG.error(f"Error receiving message: {e}")
                break
            
            action = message.get("action")
            
            if action == "calibrate":
                # Calibration mode
                frame_data = message.get("frame")
                target_x = message.get("target_x")
                target_y = message.get("target_y")
                
                if not frame_data or target_x is None or target_y is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Missing calibration data"
                    })
                    continue
                
                # Decode frame
                frame = service.decode_frame(frame_data)
                
                if frame is None:
                    await websocket.send_json({
                        "type": "calibration_response",
                        "success": False,
                        "error": "Failed to decode frame"
                    })
                    continue
                
                # Add calibration sample
                result = service.add_calibration_sample(frame, target_x, target_y)
                
                await websocket.send_json({
                    "type": "calibration_response",
                    **result
                })
            
            elif action == "track":
                # Normal gaze tracking
                frame_data = message.get("frame")
                
                if not frame_data:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Missing frame data"
                    })
                    continue
                
                # Decode frame
                frame = service.decode_frame(frame_data)
                
                if frame is None:
                    await websocket.send_json({
                        "type": "gaze_data",
                        "success": False,
                        "error": "Failed to decode frame"
                    })
                    continue
                
                # Process frame
                result = service.process_frame(frame)
                
                if result is not None:
                    await websocket.send_json({
                        "type": "gaze_data",
                        **result
                    })
            
            elif action == "reset_calibration":
                service.reset_calibration()
                
                await websocket.send_json({
                    "type": "calibration_reset",
                    "success": True
                })
            
            elif action == "get_stats":
                stats = service.get_stats()
                
                await websocket.send_json({
                    "type": "stats",
                    "data": stats
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}"
                })
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.001)
    
    except WebSocketDisconnect:
        LOG.info("Gaze tracking WebSocket disconnected normally")
    except Exception as e:
        LOG.error(f"Error in gaze tracking WebSocket: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except:
            pass
        LOG.info("Gaze tracking WebSocket connection closed")
