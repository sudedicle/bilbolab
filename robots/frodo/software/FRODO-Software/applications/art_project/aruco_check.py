# =====================================================================================
#  ARUCO CHECK  -  live camera + a box drawn around every ArUco marker it detects.
#
#  For focusing the lens by hand: carry the camera around, watch which markers get
#  boxed and at what pixel size. Uses THIS robot's own camera config from
#  ~/robot/settings.json and the SAME detector art_project_frodo.py uses
#  (aruco_utils.create_aruco_detector / detect_markers), so what you see here is
#  what navigation sees.
#
#  Box colour:
#     GREEN  - would be ACCEPTED by nav (id on the grid map, size >= MIN_SIZE_PX,
#              inside the accept window)
#     YELLOW - detected but too small / outside the window -> nav says "TOO FAR"
#     GREY   - detected but the id is not a grid node (0-53) -> nav says "NOT ON MAP"
#
#  Top-left FOCUS number = sharpness (variance of Laplacian). Turn the focus ring
#  until it peaks, then lock the screw.
#
#  Run (on the robot):
#     cd /home/admin/robot/software/applications/art_project
#     PYTHONPATH=../.. python3 aruco_check.py
#  Open:  http://<robot-ip>:5001/preview      Ctrl+C to stop.
# =====================================================================================
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np

from robot.common import FRODO_Common
from robot.sensing.camera.pycamera import PyCamera
from robot.utilities.video_streamer.video_streamer import VideoStreamer
from core.utils.network import getInterfaceIP
from aruco_utils import create_aruco_detector, detect_markers, marker_bbox

STREAM_PORT = 5001

# --- nav accept filter (keep in sync with ROBOT_TUNING for the robot you're on) ---
MIN_SIZE_PX = 140        # frodo1/frodo4 = 140  (see ROBOT_TUNING in art_project_frodo.py)
Y_MIN = 0.15             # marker centre must be BELOW this fraction of the frame
X_MIN, X_MAX = 0.10, 0.90  # ...and within this horizontal band
GRID_IDS = set(range(54))  # CITY_MAP nodes 0-53

SCAN_WHOLE_FRAME = True   # True = detect anywhere (good for a hand-held focus check);
                          # False = only the bottom region, like nav


def main():
    cfg = FRODO_Common.getDefinitions().camera
    print("=" * 70)
    print("  ARUCO CHECK  -  this robot's camera config:")
    print(f"    version {cfg.camera}  resolution {tuple(cfg.resolution)}  "
          f"fov {round(float(np.degrees(cfg.fov)), 1)} deg")
    print(f"    image_format {cfg.image_format}  exposure {cfg.exposure_time}us  gain {cfg.gain}")
    print(f"    accept: size >= {MIN_SIZE_PX}px, y > {Y_MIN}, x in [{X_MIN}, {X_MAX}], id in 0-53")
    print("=" * 70)

    camera = PyCamera(
        version=cfg.camera,
        resolution=tuple(cfg.resolution),
        exposure_time=cfg.exposure_time,
        gain=cfg.gain,
        frame_rate=cfg.frame_rate,
        image_format=cfg.image_format,
        auto_focus=getattr(cfg, "autofocus", False),
    )
    camera.start()

    detector = create_aruco_detector()
    scan_y_ratio = 0.0 if SCAN_WHOLE_FRAME else 0.10
    focus_ema = None
    last_log = 0.0

    def get_frame():
        nonlocal focus_ema, last_log
        raw = camera.takeFrame()
        if raw is None:
            return None

        if raw.ndim == 2:
            gray = raw
            disp = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        else:
            disp = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]

        # focus score over the centre 50%
        y0, y1 = int(h * 0.25), int(h * 0.75)
        x0, x1 = int(w * 0.25), int(w * 0.75)
        f = float(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var())
        focus_ema = f if focus_ema is None else (0.7 * focus_ema + 0.3 * f)

        detected, _ = detect_markers(detector, gray, scan_y_ratio)

        accepted, seen = [], []
        details = []
        for mid, corners in detected:
            bx, by, bw, bh, cx, cy, size = marker_bbox(corners)
            on_map = mid in GRID_IDS
            in_win = (cy > Y_MIN * h) and (X_MIN * w < cx < X_MAX * w)
            big = size >= MIN_SIZE_PX
            ok = on_map and in_win and big
            why = "OK" if ok else (
                "small" if not big else ("off-window" if not in_win else "not-on-map"))
            details.append(
                f"id={mid} size={size}px xy=({cx},{cy})/{w}x{h} "
                f"x%={cx / w:.2f} y%={cy / h:.2f} -> {why}")
            color = (0, 220, 0) if ok else ((150, 150, 150) if not on_map else (0, 200, 255))
            cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), color, 3)
            label = f"{mid}  {size}px"
            if not big:
                label += " small"
            if on_map and not in_win:
                label += " off-window"
            if not on_map:
                label += " not-on-map"
            cv2.putText(disp, label, (bx, max(by - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            (accepted if ok else seen).append(mid)

        # accept window guides
        cv2.line(disp, (0, int(Y_MIN * h)), (w, int(Y_MIN * h)), (0, 200, 255), 1)
        cv2.line(disp, (int(X_MIN * w), 0), (int(X_MIN * w), h), (0, 200, 255), 1)
        cv2.line(disp, (int(X_MAX * w), 0), (int(X_MAX * w), h), (0, 200, 255), 1)

        cv2.putText(disp, f"FOCUS {focus_ema:7.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(disp, f"accept: {sorted(accepted)}   seen: {sorted(seen)}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        now = time.time()
        if now - last_log > 0.5 and detected:
            last_log = now
            print(f"[{now:.1f}] focus={focus_ema:.0f}  accept={sorted(accepted)}  seen={sorted(seen)}")
            for d in details:
                print(f"           {d}")

        ok_enc, buf = cv2.imencode(".jpg", disp)
        return buf.tobytes() if ok_enc else None

    streamer = VideoStreamer(image_fetcher=get_frame, port=STREAM_PORT)
    streamer.start()

    ip = getInterfaceIP("wlan0")
    print(f"\n---> LIVE:  http://{ip}:{STREAM_PORT}/preview   <---")
    print("Green box = nav would ACCEPT it. Turn the focus ring until FOCUS peaks. Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping.")
        try:
            camera.picam.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
