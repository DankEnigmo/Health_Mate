import cv2
import numpy as np
import time
from typing import Tuple, Optional

class OnScreenKeyboard:
    """
    On-screen keyboard with dwell selection.
    Larger keys for easier selection (centered layout).
    """
    def __init__(self, rows=None, key_w=120, key_h=110, dwell_time=0.6):
        if rows is None:
            self.rows = [
                list("QWERTYUIOP"),
                list("ASDFGHJKL"),
                list("ZXCVBNM") + [" ", ".", "<"]
            ]
        else:
            self.rows = rows
        self.key_w = key_w
        self.key_h = key_h
        self.dwell_time = dwell_time
        self.text = ""
        self._last_key = None
        self._dwell_start = None
        cols = max(len(r) for r in self.rows)
        self.width = cols * key_w + 60
        self.height = len(self.rows) * key_h + 160

    def draw(self, gaze_mapped_norm: Optional[Tuple[float,float]], offset=(0.0,0.0)):
        ox, oy = offset
        img = np.ones((self.height, self.width,3), dtype=np.uint8) * 240
        y0 = 20
        for ri, row in enumerate(self.rows):
            x0 = 20 + ((max(len(r) for r in self.rows) - len(row)) * self.key_w) // 2
            for ci, key in enumerate(row):
                x = x0 + ci * self.key_w
                y = y0 + ri * self.key_h
                cv2.rectangle(img, (x,y), (x+self.key_w-10, y+self.key_h-10), (200,200,200), -1)
                cv2.rectangle(img, (x,y), (x+self.key_w-10, y+self.key_h-10), (80,80,80), 3)
                cv2.putText(img, key, (x+18, y+65), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,0,0), 3, cv2.LINE_AA)
        if gaze_mapped_norm is not None:
            gx = int((gaze_mapped_norm[0] - ox) * self.width)
            gy = int((gaze_mapped_norm[1] - oy) * self.height)
            gx = np.clip(gx, 0, self.width-1)
            gy = np.clip(gy, 0, self.height-1)
            cv2.circle(img, (gx, gy), 14, (0,0,255), -1)
        cv2.rectangle(img, (10, self.height-120), (self.width-10, self.height-40), (255,255,255), -1)
        cv2.rectangle(img, (10, self.height-120), (self.width-10, self.height-40), (60,60,60),2)
        cv2.putText(img, self.text, (20, self.height-70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0),2, cv2.LINE_AA)
        return img

    def key_at(self, norm_point: Tuple[float,float], offset=(0.0,0.0)) -> Optional[str]:
        if norm_point is None:
            return None
        x_norm, y_norm = norm_point
        ox, oy = offset
        x = int((x_norm - ox) * self.width)
        y = int((y_norm - oy) * self.height)
        y0 = 20
        for ri, row in enumerate(self.rows):
            x0 = 20 + ((max(len(r) for r in self.rows) - len(row)) * self.key_w) // 2
            for ci, key in enumerate(row):
                rx = x0 + ci * self.key_w
                ry = y0 + ri * self.key_h
                if rx <= x <= rx + self.key_w -10 and ry <= y <= ry + self.key_h -10:
                    return key
        return None

    def update_with_gaze(self, norm_point: Optional[Tuple[float,float]], offset=(0.0,0.0)):
        key = self.key_at(norm_point, offset) if norm_point is not None else None
        now = time.time()
        if key is None:
            self._last_key = None
            self._dwell_start = None
            return None
        if key != self._last_key:
            self._last_key = key
            self._dwell_start = now
            return None
        if self._dwell_start and (now - self._dwell_start) >= self.dwell_time:
            self._dwell_start = now + 0.4
            if key == " ":
                self.text += " "
            elif key == "<":
                self.text = self.text[:-1]
            else:
                self.text += key
            return key
        return None
    
    def update_with_gaze_scaled(self, norm_point: Optional[Tuple[float,float]], 
                               offset=(0.0,0.0), render_scale=1.0):
        """
        Update with gaze considering render scaling.
        render_scale: the factor by which the keyboard display was scaled
        """
        if norm_point is None:
            self._last_key = None
            self._dwell_start = None
            return None
        
        # The render_scale doesn't affect normalized coordinates
        # So we just pass through to the regular method
        return self.update_with_gaze(norm_point, offset)
