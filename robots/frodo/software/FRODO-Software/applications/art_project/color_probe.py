# =====================================================================================
#  COLOR PROBE  -  find the HSV range of the floor tape UNDER THE CURRENT LIGHT.
#
#  Aim the red sample box (centre-bottom of the frame, where line_following.py
#  actually looks) at a stretch of tape. Slowly sweep it along the WHOLE length
#  of that colour's tape - through the sunny patches AND the shaded ones - so the
#  accumulated range covers every lighting condition on the grid.
#
#  Console prints once a second:
#     box     : median H,S,V of the pixels in the box RIGHT NOW
#     session : the P2..P98 envelope over everything sampled so far
#
#  On Ctrl+C it prints ready-to-paste lower_bound / upper_bound lines for
#  line_following.get_color_mask().
#
#  Same camera config + the SAME raw->BGR->HSV pipeline as line_following.py, so
#  the numbers match what the robot actually sees.
#
#  Run (on the robot, one colour at a time - edit SAMPLE below):
#     cd /home/admin/robot/software/applications/art_project
#     PYTHONPATH=../.. python3 color_probe.py
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
from line_following import get_color_mask

STREAM_PORT = 5001

# Which colour you're sampling. Only changes the printed label + which mask is
# painted yellow on the preview; the HSV accumulation is whatever is in the box.
SAMPLE = "green"                 # "pink" or "green"

# Sample box, as fractions of the frame. Centre-bottom by default - that is where
# the line sits for the follower (ROI_TOP = 0.55 in line_following.py).
BOX_X0, BOX_X1 = 0.40, 0.60
BOX_Y0, BOX_Y1 = 0.62, 0.80

PRINT_PERIOD = 1.0
P_LOW, P_HIGH = 0.02, 0.98      # percentile envelope (2%..98% ignores stray pixels)

# The box is wider than the tape, so it also sees bare floor. Only pixels with
# saturation >= SAT_GATE are counted as "tape" - the grey carpet sits near S~5,
# coloured tape is much higher. Drop this if a tape reads very low-saturation.
SAT_GATE = 30

# Coarse hue window per colour: pixels whose H is outside it are ignored, so a
# crossing line of a different colour / a marker edge / sunlit floor in the box
# doesn't pollute the range. Keep these WIDE - the real range is discovered
# inside them. Pink wraps 0/179 so it has two segments.
HUE_GATE = {
    "pink": [(150, 180), (0, 10)],
    "green": [(28, 95)],
    "blue": [(96, 135)],
}


def _pct(hist, q):
    """Value at cumulative fraction q (0..1) of a 1-D histogram."""
    c = np.cumsum(hist)
    if c[-1] == 0:
        return 0
    return int(np.searchsorted(c, q * c[-1]))


def main():
    cfg = FRODO_Common.getDefinitions().camera
    print("=" * 70)
    print(f"  COLOR PROBE  -  sampling '{SAMPLE}'")
    print(f"    camera {cfg.camera}  {tuple(cfg.resolution)}  {cfg.image_format}  "
          f"exp={cfg.exposure_time}us gain={cfg.gain}")
    print("    Sweep the box along the whole tape - sunny bits AND shaded bits.")
    print("    Ctrl+C -> prints lower_bound / upper_bound for line_following.py")
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

    hist_h = np.zeros(180, dtype=np.int64)
    hist_s = np.zeros(256, dtype=np.int64)
    hist_v = np.zeros(256, dtype=np.int64)
    last_print = 0.0

    def get_frame():
        nonlocal last_print, hist_h, hist_s, hist_v
        raw = camera.takeFrame()
        if raw is None:
            return None

        # SAME pipeline as line_following.get_color_mask()
        if raw.ndim == 2:
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        disp = bgr.copy()
        h, w = hsv.shape[:2]

        x0, x1 = int(BOX_X0 * w), int(BOX_X1 * w)
        y0, y1 = int(BOX_Y0 * h), int(BOX_Y1 * h)
        box = hsv[y0:y1, x0:x1].reshape(-1, 3)
        n_box = len(box)

        sel = box[:, 1] >= SAT_GATE            # drop bare floor (low saturation)
        hue_ok = np.zeros(n_box, dtype=bool)
        for h_lo, h_hi in HUE_GATE.get(SAMPLE, [(0, 180)]):
            hue_ok |= (box[:, 0] >= h_lo) & (box[:, 0] <= h_hi)
        sel &= hue_ok
        tape = box[sel]
        if len(tape):
            hist_h += np.bincount(tape[:, 0], minlength=180)[:180]
            hist_s += np.bincount(tape[:, 1], minlength=256)[:256]
            hist_v += np.bincount(tape[:, 2], minlength=256)[:256]
            med = np.median(tape, axis=0).astype(int)
        else:
            med = np.zeros(3, dtype=int)

        lo = (_pct(hist_h, P_LOW), _pct(hist_s, P_LOW), _pct(hist_v, P_LOW))
        hi = (_pct(hist_h, P_HIGH), _pct(hist_s, P_HIGH), _pct(hist_v, P_HIGH))

        # overlay the CURRENT line_following mask for this colour (reference)
        mask = get_color_mask(raw, SAMPLE)
        disp[mask > 0] = (0, 255, 255)

        on_tape = int(sel.sum())
        frac = (100 * on_tape // n_box) if n_box else 0

        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(disp, f"tape med HSV {tuple(med)}   ({frac}% of box on tape)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(disp, f"session  lower {lo}  upper {hi}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp, f"yellow = current '{SAMPLE}' mask   (S gate {SAT_GATE})", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        now = time.time()
        if now - last_print > PRINT_PERIOD:
            last_print = now
            print(f"tape med H,S,V = {tuple(med)}  ({frac}% on tape)   session  "
                  f"lower~[{lo[0]}, {lo[1]}, {lo[2]}]  upper~[{hi[0]}, {hi[1]}, {hi[2]}]")

        ok, buf = cv2.imencode(".jpg", disp)
        return buf.tobytes() if ok else None

    streamer = VideoStreamer(image_fetcher=get_frame, port=STREAM_PORT)
    streamer.start()
    ip = getInterfaceIP("wlan0")
    print(f"\n---> LIVE:  http://{ip}:{STREAM_PORT}/preview   <---")
    print("Sweep the red box along the tape. Ctrl+C when done.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

    total = int(hist_h.sum())
    print("\n" + "=" * 70)
    if total == 0:
        print("  no samples collected - was the box ever over the tape?")
        print("=" * 70)
        return

    lh, ls, lv = _pct(hist_h, P_LOW), _pct(hist_s, P_LOW), _pct(hist_v, P_LOW)
    uh, us, uv = _pct(hist_h, P_HIGH), _pct(hist_s, P_HIGH), _pct(hist_v, P_HIGH)

    # H wraparound check: pink lives near 179; if the hue mass is split between
    # both ends of the circle, plain min/max is meaningless.
    ends = int(hist_h[:10].sum() + hist_h[170:].sum())
    middle = int(hist_h[15:165].sum())
    wrap = ends > 0.15 * total and middle > 0.15 * total

    print(f"  RECOMMENDED for line_following.get_color_mask()  ('{SAMPLE}',"
          f" {total} px sampled)")
    print("=" * 70)
    if wrap:
        print("  !! hue wraps around 0/179 - needs TWO ranges OR'd together,")
        print("     don't paste a single min..max. Hue histogram peaks:")
        top = np.argsort(hist_h)[-5:][::-1]
        print(f"     {sorted(int(t) for t in top)}")
    print(f"    lower_bound = np.array([{max(lh - 3, 0)}, {max(ls - 15, 0)}, {max(lv - 20, 0)}])")
    print(f"    upper_bound = np.array([{min(uh + 3, 179)}, 255, 255])")
    print()
    print("  S/V lower padded down (~15/20) for shade headroom. If it still")
    print("  drops the line in shadow, lower them more; if it grabs the bare")
    print("  floor, raise them. Keep the H band tight.")
    print("=" * 70)

    try:
        camera.picam.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
