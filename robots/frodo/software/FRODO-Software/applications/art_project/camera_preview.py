# =====================================================================================
#  CAMERA PREVIEW  -  just open THIS robot's camera and watch the live feed.
#
#  Uses the robot's OWN config from ~/robot/settings.json (lens/version, resolution,
#  exposure, gain, rgb-vs-gray) - so what you see is exactly what art_project_frodo.py
#  sees. No motors, no servo, no navigation.
#
#  Also prints a FOCUS SCORE on the image (variance of the Laplacian over the centre
#  of the frame): higher = sharper. Turn the lens focus ring until the number peaks,
#  then lock the screw. frodo4's soft focus has been the root cause of its missed
#  markers - fix it here.
#
#  Run (on the robot, one at a time):
#     cd /home/admin/robot/software/applications/art_project
#     PYTHONPATH=../.. python3 camera_preview.py
#  Then open:  http://<robot-ip>:5001/preview
#  Ctrl+C to stop.
# =====================================================================================
import os
import sys
import time

# --- repo root + own dir on sys.path (so `python3 camera_preview.py` just works) ---
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

STREAM_PORT = 5001
FOCUS_ROI = 0.5          # measure focus over the centre 50% of the frame
SHOW_CROSSHAIR = True


def main():
    cfg = FRODO_Common.getDefinitions().camera
    print("=" * 70)
    print("  CAMERA PREVIEW  -  this robot's own camera config:")
    print(f"    version        : {cfg.camera}")
    print(f"    resolution     : {tuple(cfg.resolution)}")
    print(f"    fov            : {round(float(np.degrees(cfg.fov)), 1)} deg")
    print(f"    image_format   : {cfg.image_format}")
    print(f"    exposure_time  : {cfg.exposure_time} us")
    print(f"    gain           : {cfg.gain}")
    print(f"    frame_rate     : {cfg.frame_rate}")
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

    focus_ema = None

    def get_frame():
        nonlocal focus_ema
        raw = camera.takeFrame()
        if raw is None:
            return None

        # -> BGR for drawing / JPEG
        if raw.ndim == 2:
            gray = raw
            disp = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        else:
            disp = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        m = FOCUS_ROI
        y0, y1 = int(h * (1 - m) / 2), int(h * (1 + m) / 2)
        x0, x1 = int(w * (1 - m) / 2), int(w * (1 + m) / 2)
        focus = float(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var())
        focus_ema = focus if focus_ema is None else (0.7 * focus_ema + 0.3 * focus)

        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 1)
        cv2.putText(disp, f"FOCUS {focus_ema:7.0f}  (maximise me)", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(disp, f"{w}x{h}  {cfg.image_format}  exp={cfg.exposure_time}us gain={cfg.gain}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        if SHOW_CROSSHAIR:
            cv2.line(disp, (w // 2, 0), (w // 2, h), (0, 200, 255), 1)
            cv2.line(disp, (0, h // 2), (w, h // 2), (0, 200, 255), 1)

        ok, buf = cv2.imencode(".jpg", disp)
        return buf.tobytes() if ok else None

    streamer = VideoStreamer(image_fetcher=get_frame, port=STREAM_PORT)
    streamer.start()

    ip = getInterfaceIP("wlan0")
    print(f"\n---> LIVE:  http://{ip}:{STREAM_PORT}/preview   <---")
    print("Turn the focus ring until FOCUS peaks, then lock it. Ctrl+C to stop.\n")

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
