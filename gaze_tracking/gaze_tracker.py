import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, Dict

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe iris / face landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1

class GazeTracker:
    def __init__(self, static_image_mode=False, refine_landmarks=True, min_detection_confidence=0.5):
        self.mp = mp_face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                        refine_landmarks=refine_landmarks,
                                        max_num_faces=1,
                                        min_detection_confidence=min_detection_confidence,
                                        min_tracking_confidence=0.5)
        self.calib_src = []
        self.calib_dst = []
        self.A = None
        self.prev_smoothed = None
        self.smoothing_alpha = 0.38

    def get_landmarks(self, frame) -> Optional[np.ndarray]:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp.process(img_rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        return lm

    def iris_center(self, landmarks, iris_indices, frame_shape) -> Tuple[float,float]:
        h,w = frame_shape[:2]
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in iris_indices])
        cx, cy = pts.mean(axis=0)
        return float(cx), float(cy)

    def eye_corners(self, landmarks, left=True, frame_shape=None):
        h,w = frame_shape[:2] if frame_shape is not None else (1,1)
        if left:
            p_out = landmarks[LEFT_EYE_OUTER]
            p_in = landmarks[LEFT_EYE_INNER]
        else:
            p_in = landmarks[RIGHT_EYE_INNER]
            p_out = landmarks[RIGHT_EYE_OUTER]
        return (p_out.x * w, p_out.y * h), (p_in.x * w, p_in.y * h)

    def feature_vector(self, frame):
        lm = self.get_landmarks(frame)
        if lm is None:
            return None, None
        lx, ly = self.iris_center(lm, LEFT_IRIS, frame.shape)
        rx, ry = self.iris_center(lm, RIGHT_IRIS, frame.shape)
        l_out, l_in = self.eye_corners(lm, left=True, frame_shape=frame.shape)
        r_in, r_out = self.eye_corners(lm, left=False, frame_shape=frame.shape)
        eye_cx = (l_in[0] + r_in[0]) / 2.0
        eye_cy = (l_in[1] + r_in[1]) / 2.0
        eye_w = max(1.0, np.linalg.norm(np.array(l_out) - np.array(r_out)))
        eye_h = max(1.0, eye_w * 0.35)
        mx = (lx + rx) / 2.0
        my = (ly + ry) / 2.0
        rx_rel = (mx - eye_cx) / eye_w
        ry_rel = (my - eye_cy) / eye_h
        nose = lm[NOSE_TIP]
        h,w = frame.shape[:2]
        nose_x = nose.x * w
        head_offset = (nose_x - (w/2.0)) / w
        feat = np.array([rx_rel, ry_rel, head_offset], dtype=np.float32)
        centers = {'left': (lx,ly), 'right': (rx,ry), 'mid': (mx,my)}
        return feat, centers

    def observe_calib(self, frame):
        feat_centers = self.feature_vector(frame)
        if feat_centers is None:
            return None
        feat, centers = feat_centers
        return feat

    def add_calibration_pair(self, observed_feat, target_norm):
        v = np.array([observed_feat[0], observed_feat[1], 1.0], dtype=np.float32)
        self.calib_src.append(v)
        self.calib_dst.append(np.array([target_norm[0], target_norm[1]], dtype=np.float32))
        if len(self.calib_src) >= 3:
            self.compute_affine()

    def compute_affine(self):
        src = np.vstack(self.calib_src)
        dst = np.vstack(self.calib_dst)
        A,_,_,_ = np.linalg.lstsq(src, dst, rcond=None)
        self.A = A.T

    def map_feature_to_norm(self, feat):
        if feat is None:
            return None
        v = np.array([feat[0], feat[1], 1.0], dtype=np.float32)
        if self.A is None:
            mx = 0.5 + feat[0] * 0.35
            my = 0.5 + feat[1] * 0.6
            return (float(mx), float(my))
        out = self.A.dot(v)
        return (float(np.clip(out[0], 0.0, 1.0)), float(np.clip(out[1], 0.0, 1.0)))

    def get_gaze(self, frame):
        feat_centers = self.feature_vector(frame)
        if feat_centers is None:
            return None
        feat, centers = feat_centers
        mapped = self.map_feature_to_norm(feat)
        if self.prev_smoothed is None:
            self.prev_smoothed = np.array(mapped, dtype=np.float32)
        else:
            self.prev_smoothed = (1.0 - self.smoothing_alpha) * self.prev_smoothed + self.smoothing_alpha * np.array(mapped, dtype=np.float32)
        return (float(self.prev_smoothed[0]), float(self.prev_smoothed[1])), centers
