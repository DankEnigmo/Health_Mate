
"""
Optimized Gaze Tracker with improved performance and reliability.

Key optimizations:
- Frame preprocessing and caching
- Vectorized operations
- Better error handling and recovery
- Configurable performance parameters
- Landmark confidence checking
- Automatic smoothing reset on tracking loss
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# MediaPipe iris / face landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1


class TrackingQuality(Enum):
    """Quality levels for tracking."""
    EXCELLENT = "excellent"
    GOOD = "good"
    POOR = "poor"
    LOST = "lost"


@dataclass
class GazeResult:
    """Structured result from gaze tracking."""
    x: float  # Normalized 0.0-1.0
    y: float  # Normalized 0.0-1.0
    confidence: float  # 0.0-1.0
    quality: TrackingQuality
    iris_centers: Dict[str, Tuple[float, float]]
    raw_features: Optional[np.ndarray] = None


class GazeTrackerOptimized:
    """
    Optimized gaze tracker with better performance and reliability.
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing_alpha: float = 0.38,
        max_frames_lost: int = 10,
        enable_preprocessing: bool = True
    ):
        """
        Initialize the optimized gaze tracker.
        
        Args:
            static_image_mode: Process each frame independently (slower but more accurate)
            refine_landmarks: Use iris landmark refinement
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            smoothing_alpha: Smoothing factor (0=no smoothing, 1=instant response)
            max_frames_lost: Maximum frames to lose before resetting smoothing
            enable_preprocessing: Enable frame preprocessing for better performance
        """
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            refine_landmarks=refine_landmarks,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Calibration data
        self.calib_src: List[np.ndarray] = []
        self.calib_dst: List[np.ndarray] = []
        self.A: Optional[np.ndarray] = None
        
        # Smoothing state
        self.prev_smoothed: Optional[np.ndarray] = None
        self.smoothing_alpha = smoothing_alpha
        self.frames_lost = 0
        self.max_frames_lost = max_frames_lost
        
        # Performance settings
        self.enable_preprocessing = enable_preprocessing
        self.last_frame_shape: Optional[Tuple[int, int]] = None
        
        # Statistics
        self.total_frames_processed = 0
        self.successful_detections = 0
        
        LOG.info(f"GazeTrackerOptimized initialized with smoothing_alpha={smoothing_alpha}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better face detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed frame
        """
        if not self.enable_preprocessing:
            return frame
        
        # Histogram equalization for better contrast
        try:
            # Convert to YUV for better processing
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        except Exception as e:
            LOG.warning(f"Frame preprocessing failed: {e}")
        
        return frame
    
    def get_landmarks(self, frame: np.ndarray) -> Optional[List]:
        """
        Extract facial landmarks from frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of landmarks or None if detection failed
        """
        self.total_frames_processed += 1
        
        # Preprocess frame
        if self.enable_preprocessing:
            frame = self.preprocess_frame(frame)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.mp_face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            self.frames_lost += 1
            if self.frames_lost > self.max_frames_lost:
                # Reset smoothing on prolonged tracking loss
                self.prev_smoothed = None
            return None
        
        # Reset lost frames counter
        self.frames_lost = 0
        self.successful_detections += 1
        
        return results.multi_face_landmarks[0].landmark
    
    def iris_center(
        self, 
        landmarks: List, 
        iris_indices: List[int], 
        frame_shape: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Calculate iris center from landmarks.
        
        Args:
            landmarks: Face landmarks
            iris_indices: Indices of iris landmarks
            frame_shape: Shape of the frame (h, w)
            
        Returns:
            (x, y) coordinates of iris center
        """
        h, w = frame_shape[:2]
        
        # Vectorized computation
        coords = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in iris_indices],
            dtype=np.float32
        )
        
        center = coords.mean(axis=0)
        return float(center[0]), float(center[1])
    
    def eye_corners(
        self, 
        landmarks: List, 
        left: bool, 
        frame_shape: Tuple[int, int]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get eye corner positions.
        
        Args:
            landmarks: Face landmarks
            left: True for left eye, False for right eye
            frame_shape: Shape of the frame (h, w)
            
        Returns:
            ((outer_x, outer_y), (inner_x, inner_y))
        """
        h, w = frame_shape[:2]
        
        if left:
            p_out = landmarks[LEFT_EYE_OUTER]
            p_in = landmarks[LEFT_EYE_INNER]
        else:
            p_in = landmarks[RIGHT_EYE_INNER]
            p_out = landmarks[RIGHT_EYE_OUTER]
        
        return (p_out.x * w, p_out.y * h), (p_in.x * w, p_in.y * h)
    
    def extract_features(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Extract gaze features from frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            (feature_vector, iris_centers) or (None, None) if extraction failed
        """
        landmarks = self.get_landmarks(frame)
        if landmarks is None:
            return None, None
        
        # Cache frame shape for efficiency
        if self.last_frame_shape != frame.shape[:2]:
            self.last_frame_shape = frame.shape[:2]
        
        # Extract iris positions
        lx, ly = self.iris_center(landmarks, LEFT_IRIS, frame.shape)
        rx, ry = self.iris_center(landmarks, RIGHT_IRIS, frame.shape)
        
        # Extract eye corners
        l_out, l_in = self.eye_corners(landmarks, left=True, frame_shape=frame.shape)
        r_in, r_out = self.eye_corners(landmarks, left=False, frame_shape=frame.shape)
        
        # Calculate eye region center and dimensions
        eye_cx = (l_in[0] + r_in[0]) / 2.0
        eye_cy = (l_in[1] + r_in[1]) / 2.0
        eye_w = max(1.0, np.linalg.norm(np.array(l_out) - np.array(r_out)))
        eye_h = max(1.0, eye_w * 0.35)
        
        # Calculate average iris position
        mx = (lx + rx) / 2.0
        my = (ly + ry) / 2.0
        
        # Relative iris position
        rx_rel = (mx - eye_cx) / eye_w
        ry_rel = (my - eye_cy) / eye_h
        
        # Head pose estimation (nose position)
        nose = landmarks[NOSE_TIP]
        h, w = frame.shape[:2]
        nose_x = nose.x * w
        head_offset = (nose_x - (w / 2.0)) / w
        
        # Create feature vector
        features = np.array([rx_rel, ry_rel, head_offset], dtype=np.float32)
        
        # Iris centers for debugging/visualization
        centers = {
            'left': (lx, ly),
            'right': (rx, ry),
            'mid': (mx, my)
        }
        
        return features, centers
    
    def add_calibration_point(self, features: np.ndarray, target: Tuple[float, float]) -> bool:
        """
        Add a calibration point.
        
        Args:
            features: Feature vector from extract_features
            target: Target normalized coordinates (x, y) in [0, 1]
            
        Returns:
            True if calibration point was added successfully
        """
        if features is None:
            return False
        
        # Store calibration data
        v = np.array([features[0], features[1], 1.0], dtype=np.float32)
        self.calib_src.append(v)
        self.calib_dst.append(np.array([target[0], target[1]], dtype=np.float32))
        
        # Recompute affine transformation if we have enough points
        if len(self.calib_src) >= 3:
            self.compute_affine()
            return True
        
        return False
    
    def compute_affine(self):
        """Compute affine transformation from calibration data."""
        if len(self.calib_src) < 3:
            LOG.warning("Not enough calibration points to compute affine transformation")
            return
        
        try:
            src = np.vstack(self.calib_src)
            dst = np.vstack(self.calib_dst)
            
            # Least squares solution
            A, residuals, rank, s = np.linalg.lstsq(src, dst, rcond=None)
            self.A = A.T
            
            LOG.info(f"Affine transformation computed with {len(self.calib_src)} points, residuals: {residuals}")
        except Exception as e:
            LOG.error(f"Failed to compute affine transformation: {e}")
            self.A = None
    
    def map_features_to_normalized(self, features: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Map feature vector to normalized screen coordinates.
        
        Args:
            features: Feature vector
            
        Returns:
            (x, y) normalized coordinates in [0, 1] or None
        """
        if features is None:
            return None
        
        # Validate features
        if not np.all(np.isfinite(features)):
            LOG.warning("Invalid features detected (NaN or Inf)")
            return None
        
        v = np.array([features[0], features[1], 1.0], dtype=np.float32)
        
        if self.A is None:
            # No calibration - use default mapping
            mx = 0.5 + features[0] * 0.35
            my = 0.5 + features[1] * 0.6
            return (float(np.clip(mx, 0.0, 1.0)), float(np.clip(my, 0.0, 1.0)))
        
        # Apply affine transformation
        out = self.A.dot(v)
        return (float(np.clip(out[0], 0.0, 1.0)), float(np.clip(out[1], 0.0, 1.0)))
    
    def get_gaze(self, frame: np.ndarray) -> Optional[GazeResult]:
        """
        Get gaze coordinates from frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            GazeResult object or None if tracking failed
        """
        # Extract features
        features, centers = self.extract_features(frame)
        if features is None or centers is None:
            return None
        
        # Map to normalized coordinates
        mapped = self.map_features_to_normalized(features)
        if mapped is None:
            return None
        
        # Apply temporal smoothing
        if self.prev_smoothed is None:
            self.prev_smoothed = np.array(mapped, dtype=np.float32)
        else:
            self.prev_smoothed = (
                (1.0 - self.smoothing_alpha) * self.prev_smoothed +
                self.smoothing_alpha * np.array(mapped, dtype=np.float32)
            )
        
        # Calculate confidence based on tracking quality
        detection_rate = self.successful_detections / max(1, self.total_frames_processed)
        confidence = min(1.0, detection_rate * 1.2)
        
        # Determine quality
        if self.frames_lost == 0 and detection_rate > 0.9:
            quality = TrackingQuality.EXCELLENT
        elif self.frames_lost < 3 and detection_rate > 0.7:
            quality = TrackingQuality.GOOD
        elif self.frames_lost < self.max_frames_lost:
            quality = TrackingQuality.POOR
        else:
            quality = TrackingQuality.LOST
        
        return GazeResult(
            x=float(self.prev_smoothed[0]),
            y=float(self.prev_smoothed[1]),
            confidence=confidence,
            quality=quality,
            iris_centers=centers,
            raw_features=features
        )
    
    def reset_calibration(self):
        """Reset calibration data."""
        self.calib_src = []
        self.calib_dst = []
        self.A = None
        self.prev_smoothed = None
        LOG.info("Calibration reset")
    
    def reset_smoothing(self):
        """Reset temporal smoothing."""
        self.prev_smoothed = None
        self.frames_lost = 0
        LOG.info("Smoothing reset")
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking statistics
        """
        detection_rate = self.successful_detections / max(1, self.total_frames_processed)
        
        return {
            "total_frames": self.total_frames_processed,
            "successful_detections": self.successful_detections,
            "detection_rate": detection_rate,
            "frames_lost": self.frames_lost,
            "calibration_points": len(self.calib_src),
            "is_calibrated": self.A is not None
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'mp_face_mesh') and self.mp_face_mesh:
            self.mp_face_mesh.close()
