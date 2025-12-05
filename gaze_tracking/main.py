import cv2
import time
import numpy as np
from gaze_tracker import GazeTracker
from keyboard_ui import OnScreenKeyboard
import tts
import math


def draw_keyboard_calib(kb_img, targets, idx):
    h, w = kb_img.shape[:2]
    for i, (nx, ny) in enumerate(targets):
        x = int(nx * w)
        y = int(ny * h)
        color = (0,255,0) if i == idx else (200,200,200)
        cv2.circle(kb_img, (x,y), 16, color, -1)
    return kb_img


def run():
    cap = cv2.VideoCapture(0)
    gt = GazeTracker()

    # Bigger keys for easier typing
    kb = OnScreenKeyboard(key_w=180, key_h=150, dwell_time=0.6)

    # Calibration: 3x3 grid in keyboard normalized coords
    margin = 0.06
    xs = [margin, 0.5, 1.0 - margin]
    ys = [margin, 0.5, 1.0 - margin]
    targets = [(x, y) for y in ys for x in xs]

    calib_idx = 0
    obs_per_target = 10
    collected = 0

    last_tts_time = 0
    keyboard_offset = (0.0, 0.0)

    instructions = "Look at green dot ON THE KEYBOARD (right). Press 'c' to collect sample. 'b' to go back."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cam_h, cam_w = frame.shape[:2]

        # ---- GET GAZE ----
        gaze_res = gt.get_gaze(frame)
        mapped = None
        centers = {}

        if gaze_res:
            mapped, centers = gaze_res

        # ---- GUARD AGAINST NaN ----
        if mapped is not None:
            if not (math.isfinite(mapped[0]) and math.isfinite(mapped[1])):
                mapped = None

        # ---- CREATE NATIVE KEYBOARD IMAGE ----
        kb_img_native = kb.draw(mapped, offset=keyboard_offset)
        kb_h, kb_w = kb_img_native.shape[:2]

        # ---- CALIBRATION DOTS ----
        if calib_idx < len(targets):
            kb_img_native = draw_keyboard_calib(kb_img_native, targets, calib_idx)
            kb_h, kb_w = kb_img_native.shape[:2]

        # ---- SCALE KEYBOARD IF TOO WIDE ----
        max_total_width = max(1000, cam_w + 500)
        available_for_kb = max_total_width - cam_w
        render_scale = 1.0

        if kb_w > available_for_kb and available_for_kb > 200:
            render_scale = available_for_kb / kb_w
            new_kbw = int(kb_w * render_scale)
            new_kbh = int(kb_h * render_scale)
            kb_img_display = cv2.resize(kb_img_native, (new_kbw, new_kbh), interpolation=cv2.INTER_AREA)
        else:
            kb_img_display = kb_img_native.copy()

        disp_kb_h, disp_kb_w = kb_img_display.shape[:2]

        # ---- SCALE CAMERA IF WHOLE WINDOW TOO WIDE ----
        total_w = cam_w + disp_kb_w
        max_window_w = 1500

        if total_w > max_window_w:
            available_cam_w = max_window_w - disp_kb_w
            cam_scale = max(0.35, available_cam_w / cam_w)
            cam_disp_w = int(cam_w * cam_scale)
            cam_disp_h = int(cam_h * cam_scale)
            cam_disp = cv2.resize(frame, (cam_disp_w, cam_disp_h), interpolation=cv2.INTER_AREA)
        else:
            cam_disp = frame.copy()

        cam_disp_h, cam_disp_w = cam_disp.shape[:2]

        # ---- COMPOSE SIDE BY SIDE ----
        out_h = max(cam_disp_h, disp_kb_h)
        out_w = cam_disp.shape[1] + disp_kb_w
        out = 255 * np.ones((out_h, out_w, 3), dtype=np.uint8)

        # camera (left)
        out[0:cam_disp_h, 0:cam_disp_w] = cam_disp

        # keyboard (right), vertically centered
        kb_y = (out_h - disp_kb_h) // 2
        out[kb_y:kb_y + disp_kb_h, cam_disp_w:cam_disp_w + disp_kb_w] = kb_img_display

        cv2.putText(out, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Gaze Keyboard", out)
        k = cv2.waitKey(1) & 0xFF

        # Quit
        if k == ord('q') or k == 27:
            break

        # ---- CALIBRATION ----
        if calib_idx < len(targets):
            if k == ord('c') or k == 32:
                obs = gt.observe_calib(frame)
                if obs is not None:
                    gt.add_calibration_pair(obs, targets[calib_idx])
                    collected += 1
                    print(f"Collected {collected}/{obs_per_target} for target {calib_idx + 1}/9")
                else:
                    print("No landmarks detected — adjust lighting")

            if k == ord('b'):
                if calib_idx > 0:
                    calib_idx -= 1
                    collected = 0

            if collected >= obs_per_target:
                collected = 0
                calib_idx += 1

            continue  # do not type until calibration is done

        # ---- AFTER CALIBRATION: HIT DETECTION (scale-aware) ----
        sel = kb.update_with_gaze_scaled(
            mapped,
            offset=keyboard_offset,
            render_scale=render_scale
        )

        # ---- TTS ON SELECTION ----
        if sel:
            print("Selected:", sel)
            # OPTIONAL: speak only after pause (not per letter)
            # I keep it minimal here; you can enable delayed-speak logic if you want.
            if hasattr(tts, "speak"):
                tts.speak(kb.text)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
