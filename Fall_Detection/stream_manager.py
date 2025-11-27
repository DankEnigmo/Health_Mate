"""
Video Stream Manager for Fall Detection Backend

This module manages video capture, frame generation for MJPEG streaming,
and integration with the fall detection system.
"""

import cv2
import time
import logging
import asyncio
import numpy as np
from typing import Union, AsyncGenerator, Dict, Any, Optional, Tuple, Callable
from collections import deque
from datetime import datetime

LOG = logging.getLogger(__name__)


class VideoStreamManager:
    """
    Manages video capture and frame generation for MJPEG streaming.
    Integrates with FallDetectionOrchestrator to provide real-time fall detection.
    """
    
    def __init__(
        self,
        source: Union[int, str],
        detector,
        processor,
        model,
        device,
        scale: float = 1.0,
        max_reconnect_attempts: int = 5,
        on_fall_detected: Optional[Callable[[Dict[str, Any]], None]] = None,
        patient_id: Optional[str] = None
    ):
        """
        Initialize the VideoStreamManager.
        
        Args:
            source: Video source (int for webcam, str for file/RTSP URL)
            detector: FallDetectionOrchestrator instance
            processor: OpenPifPaf processor for pose estimation
            model: OpenPifPaf model
            device: torch device (cpu or cuda)
            scale: Scale factor for input images
            max_reconnect_attempts: Maximum reconnection attempts for RTSP streams
            on_fall_detected: Optional callback function to call when a fall is detected
            patient_id: Optional patient ID for fall event tracking
        """
        self.source = source
        self.detector = detector
        self.processor = processor
        self.model = model
        self.device = device
        self.scale = scale
        self.max_reconnect_attempts = max_reconnect_attempts
        self.on_fall_detected = on_fall_detected
        self.patient_id = patient_id
        
        # Performance metrics
        self.fps = 0.0
        self.latency = 0.0
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # FPS calculation using moving average
        self.fps_history = deque(maxlen=30)
        
        # Video capture
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        
        # Track previous fall count to detect new falls
        self.previous_fall_count = 0
        
        # Initialize video capture
        self._initialize_capture()
    
    def _initialize_capture(self) -> bool:
        """
        Initialize video capture from the configured source.
        
        Returns:
            True if capture initialized successfully, False otherwise
        """
        try:
            if isinstance(self.source, int):
                # Webcam source
                self.capture = cv2.VideoCapture(self.source)
                LOG.info(f"Initialized webcam capture: {self.source}")
            elif isinstance(self.source, str) and self.source.startswith('rtsp'):
                # RTSP stream
                self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                LOG.info(f"Initialized RTSP stream: {self.source}")
            else:
                # File or other source
                self.capture = cv2.VideoCapture(self.source)
                LOG.info(f"Initialized video source: {self.source}")
            
            if self.capture.isOpened():
                self.is_connected = True
                LOG.info("Video capture initialized successfully")
                return True
            else:
                LOG.error(f"Failed to open video source: {self.source}")
                self.is_connected = False
                return False
                
        except Exception as e:
            LOG.error(f"Error initializing video capture: {e}")
            self.is_connected = False
            return False
    
    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to the video source (primarily for RTSP streams).
        
        Returns:
            True if reconnection successful, False otherwise
        """
        if self.capture:
            self.capture.release()
        
        LOG.warning(f"Attempting to reconnect to: {self.source}")
        time.sleep(2)  # Wait before reconnecting
        
        return self._initialize_capture()
    
    def _calculate_fps(self, frame_time: float):
        """
        Calculate FPS using moving average.
        
        Args:
            frame_time: Time taken to process current frame
        """
        if frame_time > 0:
            current_fps = 1.0 / frame_time
            self.fps_history.append(current_fps)
            self.fps = sum(self.fps_history) / len(self.fps_history)
    
    def _draw_overlays(
        self,
        frame: np.ndarray,
        annotations,
        fallen_people: Dict[int, Tuple[float, float, float, float]]
    ) -> np.ndarray:
        """
        Draw pose estimation and fall detection overlays on the frame.
        
        Args:
            frame: Input frame
            annotations: Pose annotations from OpenPifPaf
            fallen_people: Dictionary of fallen people with bounding boxes
            
        Returns:
            Frame with overlays drawn
        """
        # Draw pose skeletons
        for ann in annotations:
            # Get keypoints
            kps = ann.data
            
            # Draw skeleton connections (simplified version)
            # Define skeleton connections (COCO format)
            skeleton = [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ]
            
            # Draw connections
            for connection in skeleton:
                idx1, idx2 = connection[0] - 1, connection[1] - 1
                if idx1 < len(kps) and idx2 < len(kps):
                    x1, y1, c1 = kps[idx1]
                    x2, y2, c2 = kps[idx2]
                    
                    if c1 > 0.1 and c2 > 0.1:  # Confidence threshold
                        cv2.line(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),  # Green
                            2
                        )
            
            # Draw keypoints
            for i, (x, y, c) in enumerate(kps):
                if c > 0.1:  # Confidence threshold
                    cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)  # Blue
        
        # Draw red bounding boxes around fallen people
        for person_id, (x, y, w, h) in fallen_people.items():
            cv2.rectangle(
                frame,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                (0, 0, 255),  # Red
                3
            )
            # Add "FALL DETECTED" label
            cv2.putText(
                frame,
                f"FALL DETECTED (ID: {person_id})",
                (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        
        # Draw performance metrics
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            frame,
            f"Latency: {self.latency*1000:.0f}ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            frame,
            f"Fall Count: {self.detector.fall_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return frame

    
    async def generate_frames(self) -> AsyncGenerator[bytes, None]:
        """
        Generate MJPEG frames with fall detection overlays.
        
        Yields:
            JPEG-encoded frames in MJPEG format
        """
        import torch
        import PIL.Image
        from . import transforms
        
        if not self.is_connected:
            LOG.error("Video capture not initialized")
            return
        
        dropped_frames = 0
        reconnect_attempts = 0
        
        while True:
            try:
                # Read frame
                grabbed, frame = self.capture.read()
                
                # Handle RTSP stream reconnection
                if isinstance(self.source, str) and self.source.startswith('rtsp'):
                    if grabbed:
                        dropped_frames = 0
                        reconnect_attempts = 0
                    else:
                        dropped_frames += 1
                        
                        # Get input FPS
                        raw_input_fps = self.capture.get(cv2.CAP_PROP_FPS)
                        input_fps = 30.0 if not raw_input_fps or raw_input_fps <= 0 else raw_input_fps
                        
                        # Try to reconnect after 5 seconds of dropped frames
                        if dropped_frames > input_fps * 5:
                            if reconnect_attempts < self.max_reconnect_attempts:
                                reconnect_attempts += 1
                                LOG.warning(f"Reconnection attempt {reconnect_attempts}/{self.max_reconnect_attempts}")
                                
                                if self._reconnect():
                                    dropped_frames = 0
                                    continue
                                else:
                                    await asyncio.sleep(2)
                                    continue
                            else:
                                LOG.error("Max reconnection attempts reached")
                                break
                        
                        continue
                
                elif frame is None:
                    LOG.info("No more frames available")
                    break
                
                # Start timing for latency calculation
                frame_start_time = time.time()
                
                # Scale frame if needed
                if self.scale != 1.0:
                    frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale)
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Preprocess for OpenPifPaf
                image_pil = PIL.Image.fromarray(frame_rgb)
                processed_image, _, __ = transforms.EVAL_TRANSFORM(image_pil, [], None)
                
                # Run pose estimation (only if processor is available)
                annotations = []
                if self.processor is not None:
                    with torch.no_grad():
                        annotations = self.processor.batch(
                            self.model,
                            torch.unsqueeze(processed_image, 0),
                            device=self.device
                        )[0]
                else:
                    # No processor available - skip pose estimation
                    # This happens when using a freshly-built untrained model
                    LOG.debug("Skipping pose estimation (no processor available)")
                
                # Get input FPS for fall detection
                raw_input_fps = self.capture.get(cv2.CAP_PROP_FPS)
                input_fps = 30.0 if not raw_input_fps or raw_input_fps <= 0 else raw_input_fps
                
                # Run fall detection
                fall_count, fallen_people = self.detector.process_detections(annotations, input_fps)
                
                # Check if a new fall was detected and trigger callback
                if fall_count > self.previous_fall_count and self.on_fall_detected is not None:
                    # New fall detected!
                    for person_id, bbox in fallen_people.items():
                        # Only emit for newly fallen people
                        if person_id not in self.detector.prev_fallen or \
                           len(self.detector.prev_fallen) < len(fallen_people):
                            
                            # Create fall alert data
                            fall_alert_data = {
                                "patient_id": self.patient_id or "unknown",
                                "person_tracking_id": int(person_id),
                                "fall_count": fall_count,
                                "timestamp": datetime.utcnow().isoformat(),
                                "metadata": {
                                    "bounding_box": {
                                        "x": float(bbox[0]),
                                        "y": float(bbox[1]),
                                        "width": float(bbox[2]),
                                        "height": float(bbox[3])
                                    },
                                    "fps": float(self.fps),
                                    "frame_number": self.frame_count
                                }
                            }
                            
                            # Call the callback (this will broadcast via WebSocket)
                            try:
                                # If callback is async, await it
                                if asyncio.iscoroutinefunction(self.on_fall_detected):
                                    await self.on_fall_detected(fall_alert_data)
                                else:
                                    self.on_fall_detected(fall_alert_data)
                                
                                LOG.info(f"Fall alert emitted for person {person_id}, total falls: {fall_count}")
                            except Exception as e:
                                LOG.error(f"Error calling fall detection callback: {e}")
                    
                    # Update previous fall count
                    self.previous_fall_count = fall_count
                
                # Draw overlays
                frame_with_overlays = self._draw_overlays(frame_rgb, annotations, fallen_people)
                
                # Convert back to BGR for encoding
                frame_bgr = cv2.cvtColor(frame_with_overlays, cv2.COLOR_RGB2BGR)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if not ret:
                    LOG.error("Failed to encode frame")
                    continue
                
                # Calculate metrics
                frame_end_time = time.time()
                self.latency = frame_end_time - frame_start_time
                
                frame_time = frame_end_time - self.last_frame_time
                self._calculate_fps(frame_time)
                self.last_frame_time = frame_end_time
                
                self.frame_count += 1
                
                # Yield frame in MJPEG format
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.001)
                
            except Exception as e:
                LOG.error(f"Error generating frame: {e}", exc_info=True)
                await asyncio.sleep(0.1)
                continue
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary containing FPS, latency, fall count, and connection status
        """
        return {
            'fps': round(self.fps, 2),
            'latency': round(self.latency * 1000, 2),  # Convert to milliseconds
            'fall_count': self.detector.fall_count,
            'frame_count': self.frame_count,
            'is_connected': self.is_connected,
            'is_processing': self.capture is not None and self.capture.isOpened()
        }
    
    def release(self):
        """Release video capture resources."""
        if self.capture:
            self.capture.release()
            self.is_connected = False
            LOG.info("Video capture released")
