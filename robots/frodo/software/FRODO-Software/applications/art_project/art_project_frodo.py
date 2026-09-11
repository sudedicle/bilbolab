# =====================================================================================
#  FRODO - Line Following + ArUco Grid Navigation
#
#  WiFi command surface matches hoca's host-side protocol (see
#  software/robots/frodo/applications/artproject/application_artproject.py):
#  go_to_position, stop, get_pose, get_status - names/payloads kept 1:1 compatible so
#  ArtProject_Application/ArtProject_FRODO on the host can drive this robot unchanged.
#  trigger_servo is an extra, FRODO-specific command (no host-side equivalent). The
#  metronome servo otherwise cycles automatically on arrival at the target node
#  (SERVO_ON_ARRIVAL) - the metronome device sits ON that grid node.
#
#  Navigation itself is still grid/line-based (not free x,y driving), so
#  go_to_position() snaps the requested world (x, y) to the nearest known grid node
#  and reuses the existing turn-by-turn navigation - see world_to_grid_node() below.
# =====================================================================================
import time
import threading
import cv2
import numpy as np

from core.communication.wifi.data_link import CommandArgument
from robot.frodo import FRODO
from robot.control.frodo_control import FRODO_ControlMode
from robot.utilities.video_streamer.video_streamer import VideoStreamer
from core.utils.network import getInterfaceIP
from hardware.hardware.servo import HardwareServo, NullServo
from pose_estimator import PoseEstimator, wrap_pi, MARKER_WORLD_MAP, GRID_CELL_M
from line_following import (
    get_color_mask, find_line, calculate_deviation,
    proportional_controller, calculate_wheel_speeds,
)
from aruco_utils import (
    ARUCO_DICT_TYPE, ARUCO_SCAN_Y_MIN,
    create_aruco_detector, detect_markers, marker_bbox,
)
from servo_trigger import ArucoServoTrigger
from grid_nav import grid_nodes, next_heading, direction_between, DIRECTIONS
from path_planner import plan_cooperative_paths
from peer_sync import PeerSync

# --- SERVO ---
# The metronome device is placed at a plain grid NODE. The robot navigates there
# with the normal floor markers (CITY_MAP, 0-53); the moment the TARGET marker is
# ACCEPTED it opens the servo (starts the metronome), keeps line-following slowly
# forward for SERVO_OPEN_DRIVE_TIME, then closes the servo, stops, and reports
# MISSION COMPLETE (see the METRONOME state in run()). There is NO special floor
# marker for the servo any more - the old 995-999 "metronome trigger" markers are
# now robot BODY markers (see ROBOT_BODY_MARKERS below).
# First verify the pin/servo with servo_test.py.
SERVO_PIN = 19                 # BCM GPIO19 - verified with the servo_pin_scan.py sweep
SERVO_ANGLE_HOME = 0
SERVO_ANGLE_TRIGGER = 90
SERVO_ON_ARRIVAL = True         # open the servo (metronome) automatically on arrival at the target node
SERVO_OPEN_DRIVE_TIME = 1.5     # after the target marker is ACCEPTED: keep line-following slowly
                                # forward with the servo OPEN this long (metronome runs), then
                                # close the servo, stop, and report MISSION COMPLETE.

# =====================================================================================
#  PER-ROBOT TUNING  -  the ONE place for every value that differs between robots.
#
#  TUNING_DEFAULTS = the shared baseline. ROBOT_TUNING[<id>] overrides individual
#  keys for that robot; anything it doesn't list falls back to TUNING_DEFAULTS.
#  run() reads the merged dict once as `tune` (= TUNING_DEFAULTS + ROBOT_TUNING[id]).
#
#  To bring a new robot online: add a ROBOT_TUNING entry with ONLY the keys that
#  need to differ (usually aruco_min_size_px for its lens FOV, turn_kp for its
#  motor strength, sometimes advance_time). Watch the console on the first run
#  ("Per-robot tuning for ...", "size=NNNpx -> TOO FAR / ACCEPTED", turn logs) and
#  adjust. Constants NOT in this dict (TURN_OMEGA_MAX, TURN_TOLERANCE, ADVANCE
#  speeds, MARKER_TIMEOUT_S, ...) are the same for every robot.
# =====================================================================================
TUNING_DEFAULTS = {
    # --- ArUco grid-marker acceptance ---
    # Markers smaller than this (= further away) are NOT accepted for a
    # position/heading fix or a turn decision. 160 was measured on frodo4's 60deg
    # lens (robot/definitions.py); a wider lens puts the same marker on fewer
    # pixels, so those robots need a lower floor (roughly 160 * fov_deg / 60).
    # Too high -> mostly "TOO FAR" at real intersections, robot navigates blind.
    # Too low  -> accepts noise / a marker in the next lane over.
    "aruco_min_size_px": 160,

    # ArUco ACCEPT window (fractions of the frame). A grid marker the robot should
    # read is directly ahead => LOW in the frame (big center_y) and near the
    # horizontal centre. A marker one row/column over sits HIGHER and/or off to the
    # side. On a wide (120deg) lens those adjacent markers are still well inside a
    # loose window and get mistaken for the current node -> wrong position / wrong
    # heading calibration (frodo1 read (7,2) instead of (6,3)). Tighten per robot:
    #   aruco_y_min  - reject anything ABOVE this height (further away / next row)
    #   aruco_x_min/max - reject anything outside this horizontal band (next lane)
    "aruco_y_min": 0.15,
    "aruco_x_min": 0.10,
    "aruco_x_max": 0.90,

    # --- closed-loop turning (EKF psi + PI) ---
    # turn_angle_deg = the RELATIVE angle commanded per grid turn. >90 on purpose:
    # the drive undershoots a true right angle. Re-measure with a protractor after
    # changing and adjust.
    "turn_angle_deg": 96.0,
    "turn_kp": 1.6,            # rad/s per rad of error
    "turn_ki": 0.4,            # integral gain (overcomes friction / static error)

    # --- open-loop advance to the intersection centre BEFORE a pivot ---
    # After the turn marker is ACCEPTED the robot keeps driving straight (open-loop,
    # not looking at the image) for advance_time, THEN pivots - so it ends up on the
    # node instead of turning early (the accept threshold fires while the marker is
    # still ~a node away). advance_time_boundary is used instead when the cell
    # straight ahead is off the grid (grid edge) so it doesn't drive off the end.
    # 1.8 was tuned on frodo1 (80px far-accept); a robot that accepts the marker
    # closer needs less - override per robot.
    "advance_time": 1.8,
    "advance_time_boundary": 0.9,

    # --- "lost / overshot" guard ---
    # Give up (FAILED: NO MARKERS) after driving this far since the last ACCEPTED
    # marker while navigating to a target. Must exceed the longest gap between two
    # readable markers on a route - a narrow-lens / soft-focus robot that skips
    # intermediate node markers needs more (the final leg to the target can be 3
    # cells = ~0.86 m).
    "marker_lost_dist_m": 0.9,

    # --- line following / drive ---
    "line_kp": 0.008,         # steering P-gain (px error -> omega)
    "line_kd": 0.0015,        # steering D-gain (px/s -> omega) - damps the left-right
                              # weave after a turn. Raise if it still weaves, lower if
                              # steering feels twitchy / jerky. 0 = pure P (old behaviour).
    "base_speed": 0.08,       # forward speed while following / advancing (m/s)
    "track_width": 0.150,     # wheel separation (m) - physically per-robot
}

ROBOT_TUNING = {
    "frodo1": {
        # Deliberately IDENTICAL to frodo4 (user's call, for an apples-to-apples run).
        # NOTE: frodo1 physically has a 120deg lens vs frodo4's 60deg, so it may
        # still read the next row/column's marker as "this node" - if it does, put
        # aruco_y_min ~0.45 / aruco_x_min-max 0.30-0.70 back here.
        "aruco_min_size_px": 140,
        "turn_kp": 2.0,
        "turn_angle_deg": 108.0,
        "advance_time": 1.6,
        "advance_time_boundary": 0.4,
        "marker_lost_dist_m": 1.3,
    },
    "frodo2": {
        "aruco_min_size_px": 80,     # 120deg lens
    },
    "frodo3": {
        "aruco_min_size_px": 107,    # 90deg lens
    },
    "frodo4": {
        "aruco_min_size_px": 140,
        "turn_kp": 1.8,
        "turn_angle_deg": 13.0,    # 96 under-rotated - frodo4 ended the pivot short of the new
                                    # line and lost it. Raise until it lands ON the line (watch the
        "advance_time": 1.6,
        "advance_time_boundary": 0.4,
        "marker_lost_dist_m": 1.3,
    },
    # "frodo5": { ... add its entry here ... },
}


def tuning_for(robot_id):
    """TUNING_DEFAULTS with this robot's ROBOT_TUNING overrides merged on top.
    Unknown robot id -> plain defaults."""
    merged = dict(TUNING_DEFAULTS)
    merged.update(ROBOT_TUNING.get(robot_id, {}))
    return merged


# (The ArUco ACCEPT window - aruco_y_min / aruco_x_min / aruco_x_max - and the
#  size floor aruco_min_size_px are all per-robot: see the PER-ROBOT TUNING block.)

# Navigating with a target but NO accepted marker after driving marker_lost_dist_m
# (per-robot, see TUNING_DEFAULTS) since the last one => drifted off the grid /
# overshot with nothing to stop it. Distance-based, not time-based, so it does NOT
# false-fire on a robot that is merely slow to reach the next marker. A narrow-lens
# robot that can't read every intermediate marker on a straight leg needs a bigger
# budget (the final target leg can be 3 cells = ~0.86 m).
MARKER_TIMEOUT_S = 30.0   # backstop for a robot stuck not moving and seeing nothing (shared)

# In FOLLOWING (a plain pass-through node, not the final target), a hard stop
# the instant the line is lost can permanently strand the robot: an ArUco card
# sitting on/near the line increasingly covers it as the robot gets close,
# which can lose the line just short of ARUCO_MIN_SIZE_PX (observed in the
# field: marker size froze at ~. 145px, just under the 160px accept threshold,
# because the robot fully stopped instead of creeping the last bit forward).
# Creep straight open-loop for a bit first - either the line reappears past
# the marker, or the marker grows enough to be ACCEPTED - before giving up
# and actually stopping.
LINE_LOST_CREEP_TIME = 1.0

# detectMarkers() is slow when scanning the whole frame; that delay makes the
# robot overshoot an intersection and miss the turn. Only scan the bottom region
# (markers are never expected in the top part anyway).
ARUCO_LOG_PERIOD = 0.5     # log a marker at most twice per second

# --- CLOSED-LOOP TURNING (EKF psi feedback), values shared by every robot ---
# Instead of a blind timed turn, we turn a relative +-turn_angle_deg (per-robot,
# see TUNING_DEFAULTS) from whatever psi actually reads when the turn starts, with
# a PI controller (see the turn-start block below for why this is relative rather
# than an absolute EAST=0/NORTH=90/.. target - psi is odometry-only and drifts
# over a multi-leg run). The turn PI gains (turn_kp / turn_ki) and the pre-pivot
# advance (advance_time / advance_time_boundary) are ALSO per-robot - see the
# PER-ROBOT TUNING block above. Only the limits below are the same for all robots.
TURN_I_LIMIT = 0.5           # clamp on the integral term's contribution (rad/s)
TURN_OMEGA_MAX = 1.4         # max angular speed allowed while turning (rad/s)
TURN_TOLERANCE = np.radians(4.0)   # tolerance for "reached the target"
TURN_SETTLE_TIME = 0.15      # stay within tolerance this long before calling the turn done (noise rejection)

# Marker ID -> grid coordinate. PLACEHOLDER: the real ID/coordinate mapping
# will be assigned once the markers are placed in the field.
CITY_MAP = {
    0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0), 5: (5, 0), 6: (6, 0), 7: (7, 0), 8: (8, 0),
    9: (0, 1), 10: (1, 1), 11: (2, 1), 12: (3, 1), 13: (4, 1), 14: (5, 1), 15: (6, 1), 16: (7, 1), 17: (8, 1),
    18: (0, 2), 19: (1, 2), 20: (2, 2), 21: (3, 2), 22: (4, 2), 23: (5, 2), 24: (6, 2), 25: (7, 2), 26: (8, 2),
    27: (0, 3), 28: (1, 3), 29: (2, 3), 30: (3, 3), 31: (4, 3), 32: (5, 3), 33: (6, 3), 34: (7, 3), 35: (8, 3),
    36: (0, 4), 37: (1, 4), 38: (2, 4), 39: (3, 4), 40: (4, 4), 41: (5, 4), 42: (6, 4), 43: (7, 4), 44: (8, 4),
    45: (0, 5), 46: (1, 5), 47: (2, 5), 48: (3, 5), 49: (4, 5), 50: (5, 5), 51: (6, 5), 52: (7, 5), 53: (8, 5),
}

# Closed-loop turning (via EKF psi) no longer depends on a time estimate;
# this is only a safety net in case the PI never converges at all.
TURN_TIMEOUT = 10.0


# =====================================================================================
#  MULTI-ROBOT COLLISION AVOIDANCE
#    Approach A (below, unchanged): reactive, camera-only, no central host - a robot
#    only "sees" whoever is physically ahead of it right now (ArUco body markers) and
#    reroutes/emergency-stops off that. Always active, still the final safety net.
#
#    Approach B (hoca's "both start AND target known up front" scenario - see
#    path_planner.py / peer_sync.py): each robot broadcasts its OWN (current_node,
#    target_node) to the others over a dedicated UDP channel (peer_sync.py - no
#    change to the host protocol at all), then runs Cooperative A*
#    (path_planner.plan_cooperative_paths) locally using every robot's latest
#    broadcast to plan a full conflict-free route, not just "avoid what the camera
#    sees right now". Used in ArtProject._cooperative_next_step() below, layered ON
#    TOP of Approach A (never replaces the camera-based emergency stop) - if no peer
#    has broadcast recently (single-robot test, WiFi peer link down), it returns None
#    and driving falls back to plain Approach A, unchanged.
# =====================================================================================
# Floor grid size - MUST match CITY_MAP's layout (id -> (id % COLS, id // COLS)).
GRID_COLS = 9
GRID_ROWS = 6
GRID_NODES = grid_nodes(GRID_COLS, GRID_ROWS)

_OPPOSITE_HEADING = {"EAST": "WEST", "WEST": "EAST", "NORTH": "SOUTH", "SOUTH": "NORTH"}


def grid_node_to_world(node: tuple[int, int]) -> tuple[float, float]:
    """Grid (col, row) -> world (x, y) meters. Same formula as MARKER_WORLD_MAP /
    CITY_MAP (id -> (id % GRID_COLS, id // GRID_COLS)) - GRID_CELL_M is the real,
    field-measured spacing between grid nodes (see pose_estimator.py)."""
    col, row = node
    return col * GRID_CELL_M, row * GRID_CELL_M


def world_to_grid_node(x: float, y: float) -> tuple[int, int]:
    """World (x, y) meters -> nearest grid node, clamped to the CITY_MAP grid.
    Used by go_to_position() - hoca's host protocol hands us a world position,
    but navigation here only knows how to drive to a grid node."""
    col = int(round(x / GRID_CELL_M))
    row = int(round(y / GRID_CELL_M))
    col = max(0, min(GRID_COLS - 1, col))
    row = max(0, min(GRID_ROWS - 1, row))
    return col, row


# frodo1 wears 995 (front) + 996 (back), frodo4 wears 997 (front) + 998 (back).
# The FRONT face is used separately (see higher_priority_front_markers below):
# a HIGHER-priority robot facing us means oncoming/crossing traffic and gets a
# full priority-gated stop-and-wait (_sighting_hold_active). Either face, from
# ANY robot, also feeds the priority-independent proximity brake
# (_proximity_speed_cap) so nobody - priority-privileged or not - drives
# straight into whatever's physically ahead of them.
ROBOT_BODY_MARKERS = {
    "frodo1": {"front": 995, "back": 996},
    "frodo4": {"front": 997, "back": 998},
}
# Right-of-way: earlier = higher priority. It only governs who does a full
# stop-and-wait when two robots are actively contesting the same crossing
# (_sighting_hold_active - a higher-priority robot never yields this way to a
# lower-priority one). Everything else is priority-blind: ANY robot occupying
# the cell straight ahead gets routed around (_forward_cell_blocked - shortest
# path with that cell removed, regardless of whose marker it is), and the
# EMERGENCY stop / proximity brake below apply to every robot the same way -
# a robot sitting in your path is an obstacle to avoid no matter its rank.
ROBOT_PRIORITY = ["frodo1", "frodo2", "frodo3", "frodo4"]

# Field-testing target node per robot, used when no go_to_position() has arrived
# over WiFi (SSH-run standalone). The metronome device sits ON this grid node -
# the robot drives here with the floor markers, parks, and cycles the servo.
# (col, row); CITY_MAP id = row * GRID_COLS + col.  Edit freely per test.
# 2026-09-08: metronome at marker id 9 = (0,1) for frodo1, id 28 = (1,3) for frodo4.
TEST_TARGET_BY_ROBOT = {
    "frodo1": (7, 0),   # CITY_MAP id
    "frodo4": (3, 0),   # CITY_MAP id 38
}

# Another robot seen closer than *_BLOCK_DISTANCE_M and within +-*_AHEAD_BEARING
# of straight-ahead makes the cell ahead "occupied". *_EMERGENCY_DISTANCE_M is a
# hard stop for any robot (prevents actual contact while the other one clears).
OTHER_ROBOT_BLOCK_DISTANCE_M = 0.55
OTHER_ROBOT_EMERGENCY_DISTANCE_M = 0.28
OTHER_ROBOT_AHEAD_BEARING = np.radians(45)

# A body marker can drop out of view for reasons that have nothing to do with
# the other robot actually clearing out of the way - most notably it pivoting
# in place (TURNING_LEFT/RIGHT), which turns its front/back marker face away
# from our camera while it's still sitting on the same spot (body markers only
# cover the front+back faces - see ROBOT_BODY_MARKERS). frodo_sensors.py
# replaces aruco_measurements wholesale every detection cycle with no
# persistence, so a marker gone from this frame's list looks identical to
# "the robot left" - it isn't. Keep the last real sighting "live" for this
# long before treating the path as actually clear (observed in the field:
# frodo4 lost frodo1's marker the instant frodo1 started turning and drove
# into it). 3s ~= one grid-turn's worth of time (TURN_TIMEOUT=10s is the
# worst case, but a normal turn completes well within this).
OTHER_ROBOT_LAST_SEEN_GRACE_S = 3.0

# The distance-triggered emergency stop above (OTHER_ROBOT_EMERGENCY_DISTANCE_M)
# still fires too late in the field: by the time the other robot is MEASURED
# within that distance, camera/processing lag plus the robot's own momentum has
# already carried it into contact (observed: frodo4 kept driving for a bit after
# first reading frodo1's marker and hit it). So, separately, cap how long we're
# allowed to keep driving after FIRST seeing the other robot's marker at all
# (any distance, still gated to the forward bearing cone) - then force a stop
# and hold it, regardless of what the marker does meanwhile.
OTHER_ROBOT_SIGHT_STOP_DELAY_S = 1.0   # keep driving at most this long after first sighting
OTHER_ROBOT_SIGHT_HOLD_S = 4.0         # then hold stopped this long before re-evaluating
# NOTE: the sighting-hold above is now keyed to a HIGHER-priority robot's
# FRONT marker only (an oncoming/crossing robot) - see _sighting_hold_active()
# and higher_priority_front_markers below.

# --- PROXIMITY GOVERNOR (any other robot's marker ahead, ANY face, ANY priority) ---
# The priority-gated sighting-hold above only protects a LOWER-priority robot
# (it stops for a higher-priority one) - a higher-priority robot deliberately
# does NOT hold/yield (see the comment on higher_priority_front_markers below),
# so on its own it would drive at full base_speed right up to the abrupt
# OTHER_ROBOT_EMERGENCY_DISTANCE_M stop - too late given camera/processing lag
# + momentum (same problem the sighting-hold was built to fix, just now
# unprotected again for the robot that never yields). Observed in the field:
# frodo1 (priority 0, never yields) drove straight into a stationary frodo4
# that was correctly waiting on frodo1's own path ahead of frodo1's turn.
# Fix: a smooth, continuously-recomputed (no timer, so it can't fall out of
# sync and burst back to full speed like a hold-then-resume would) proportional
# brake against the CLOSEST other robot marker of any face/priority. Linear
# ramp: at/inside the stop gap -> 0 speed, at/beyond the full-speed gap ->
# uncapped, linear in between. See _proximity_speed_cap().
PROXIMITY_STOP_GAP_M = 0.10        # desired minimum gap to any other robot's marker
PROXIMITY_FULL_SPEED_GAP_M = 0.35  # at/beyond this gap, no speed cap at all


# =====================================================================================
class ArtProject:
    """

    Line-following + ArUco grid-navigation + servo-trigger agent.

    WiFi command surface matches hoca's host protocol (application_artproject.py /
    ArtProject_FRODO), so ArtProject_Application.plan()+move_robots() can drive this
    robot as-is:
      - go_to_position(x, y, psi=None, speed=None, tolerance=None): drive to a world
        position (meters) - async, snaps to the nearest grid node (see
        world_to_grid_node() above) and reuses the grid navigation below. Completion/
        failure is reported via the 'art_project' event ('position_reached' / 'error' /
        'aborted'), matching the host's ArtProject_FRODO._on_robot_event demux.
      - stop(): abort the current move, halt in place, fires 'aborted' if a move was active
      - get_pose(): synchronous {x, y, psi, time} - matches ArtProject_Pose.from_dict
      - get_status(): synchronous {state, target, pose} (+ FRODO-specific extras)

    Extra, FRODO-specific command (no host-side equivalent):
      - trigger_servo(): manually cycle the metronome servo, independent of position
        (it otherwise cycles automatically on arrival at the target node)

    All hardware access (motors, servo, camera) happens exclusively on the run()
    loop thread. The WiFi-invoked methods above only set/read plain attributes
    under self._lock (a mailbox pattern) so there is never concurrent hardware access.
    """

    # === INIT =========================================================================================================
    def __init__(self, frodo: FRODO):
        self.frodo = frodo
        self._lock = threading.Lock()

        # --- shared state (written by run(), read by get_status()) ---
        self.state = "FOLLOWING"
        self.current_node = None          # last grid node confirmed by a marker read

        # --- shared state (written by WiFi commands, read by run()) ---
        # Field-testing target (see TEST_TARGET_BY_ROBOT) - the robot heads here
        # immediately on start. A robot with no entry there stays put (target None)
        # until the host calls go_to_position().
        self.target_node = TEST_TARGET_BY_ROBOT.get(frodo.common.id)
        if self.target_node is not None and tuple(self.target_node) not in CITY_MAP.values():
            frodo.logger.error(
                f"TEST_TARGET_BY_ROBOT[{frodo.common.id!r}] = {self.target_node} is not a grid node "
                f"(cols 0-{GRID_COLS - 1}, rows 0-{GRID_ROWS - 1}) - ignoring, robot will wait for go_to_position()")
            self.target_node = None
        self.stopped = False              # manual halt-in-place, set by stop()
        self._manual_servo_request = False

        # --- streaming ---
        self._stream_frame_lock = threading.Lock()
        self.frame_out = None
        self.streamer = VideoStreamer(image_fetcher=self._get_stream_frame, port=5001)

        # --- ArUco detector ---
        self.aruco_detector = create_aruco_detector(ARUCO_DICT_TYPE)

        # --- servo (cycled in place on arrival at the target node) ---
        # Fall back to a no-op servo if the hardware isn't wired up / provisioned
        # yet (missing rpi_hardware_pwm or the config.txt PWM overlay) - the grid
        # navigation and ArUco logic still run, the servo just doesn't move.
        try:
            # self.servo = NullServo()
            self.servo = HardwareServo(pin=SERVO_PIN)
        except (ImportError, ModuleNotFoundError, FileNotFoundError, OSError) as e:
            frodo.logger.warning(f"HardwareServo unavailable ({e}) - using NullServo (servo will not move)")
            self.servo = NullServo()
        # No auto-trigger any more (no floor marker for the servo) - trigger_ids is
        # empty, the object is kept only for its rotate_to_trigger()/rotate_to_home()
        # mechanics, used on target arrival and by the manual trigger_servo() command.
        self.servo_trigger = ArucoServoTrigger(
            self.servo, frodo.control.setTrackSpeed,
            trigger_ids=set(),
            angle_home=SERVO_ANGLE_HOME, angle_trigger=SERVO_ANGLE_TRIGGER,
        )

        # --- pose estimation (EKF: prediction + ArUco correction) ---
        # PoseEstimator in pose_estimator.py: predicts at 100 Hz from wheel odometry,
        # applies an EKF correction using measurements from frodo.sensors.aruco_detector
        # (which it converts itself from the camera frame into the robot frame). For
        # now this is ONLY shown on screen - it does NOT drive the FSM/turn decisions,
        # that logic still relies on the pixel-based ArUco detection.
        self.pose_est = PoseEstimator(frodo, verbose=False)

        # --- multi-robot collision avoidance ---
        my_id = frodo.common.id
        try:
            self.my_priority = ROBOT_PRIORITY.index(my_id)
        except ValueError:
            self.my_priority = len(ROBOT_PRIORITY)      # unknown host -> lowest priority
        # marker IDs of robots that outrank me -> I route around / yield to these.
        # FRONT face only, split out separately: it means that robot is facing
        # me (oncoming/crossing) -> I do a full priority-gated stop-and-wait
        # (_sighting_hold_active). A HIGHER-priority robot never yields this way
        # to a LOWER-priority one - "a higher-priority robot ignores the others
        # and drives its own shortest path" (see the ROBOT_PRIORITY comment
        # above) - otherwise, if both robots are the same kind of blocked-by-
        # the-other, BOTH stop together, BOTH resume together after the same
        # hold, and are still on a collision course with nothing having
        # actually yielded (observed in the field: frodo1 and frodo4 driving
        # straight at each other both paused 4s and then drove into each other
        # anyway - priority is what's supposed to break that symmetry, so only
        # the LOWER-priority robot may honor this). The higher-priority robot's
        # own protection against actually hitting something is the priority-
        # independent _proximity_speed_cap() below instead.
        self.higher_priority_front_markers = set()
        for other_id in ROBOT_PRIORITY[:self.my_priority]:
            faces = ROBOT_BODY_MARKERS.get(other_id, {})
            if "front" in faces:
                self.higher_priority_front_markers.add(faces["front"])
        # every other robot's markers (both faces, any priority) -> the
        # symmetric distance emergency stop AND the proximity speed governor -
        # neither is priority-gated, both are last-resort "just don't hit it"
        # protection against actual contact (see the EMERGENCY STOP comment
        # and _proximity_speed_cap() below).
        self.other_robot_markers = set()
        for other_id, faces in ROBOT_BODY_MARKERS.items():
            if other_id != my_id:
                self.other_robot_markers |= set(faces.values())
        # latch_key -> (last_seen_time, dist_m, bearing_rad) - see
        # OTHER_ROBOT_LAST_SEEN_GRACE_S / _nearest_robot_ahead()
        self._last_robot_seen = {}
        # time-based sighting hold - see _sighting_hold_active() /
        # OTHER_ROBOT_SIGHT_STOP_DELAY_S / OTHER_ROBOT_SIGHT_HOLD_S
        self._other_seen_since = None
        self._sight_hold_until = None
        frodo.logger.info(
            f"Collision avoidance: id={my_id!r} priority={self.my_priority} "
            f"stop-for(front)={sorted(self.higher_priority_front_markers)} "
            f"route-around/brake-for(any)={sorted(self.other_robot_markers)}")

        # Approach B (see the big comment above CITY_MAP) - opened in run() once the
        # robot's WiFi IP is known; stays None (and _cooperative_next_step() then
        # always returns None, i.e. "no opinion, use Approach A") if that fails.
        self.peer_sync: PeerSync | None = None

        self._register_wifi_commands()

    # === WIFI COMMANDS (called remotely from the host) ================================================================
    def _register_wifi_commands(self):
        wifi = self.frodo.communication.wifi

        wifi.newCommand(
            identifier='go_to_position',
            function=self.go_to_position,
            description='Drive to a world-frame position (meters). Async - reports completion via '
                        'the art_project event (position_reached/error/aborted).',
            arguments=[
                CommandArgument(name='x', type=float, description='Target X [m]'),
                CommandArgument(name='y', type=float, description='Target Y [m]'),
                CommandArgument(name='psi', type=float, description='Final heading [rad] (accepted, not used yet)',
                                optional=True, default=None),
                CommandArgument(name='speed', type=float, description='Speed [m/s] (accepted, not used yet)',
                                optional=True, default=None),
                CommandArgument(name='tolerance', type=float, description='Arrival tolerance [m] (accepted, not used yet)',
                                optional=True, default=None),
            ]
        )

        wifi.newCommand(
            identifier='stop',
            function=self.stop,
            description='Abort the current move (if any), halt in place, and clear any pending target.',
            arguments=[]
        )

        wifi.newCommand(
            identifier='get_pose',
            function=self.get_pose,
            description='Return the latest pose estimate as a dict (x, y, psi, time).',
            arguments=[],
            execute_in_thread=False,
        )

        wifi.newCommand(
            identifier='get_status',
            function=self.get_status,
            description='Return current state, target, and pose.',
            arguments=[],
            execute_in_thread=False,
        )

        wifi.newCommand(
            identifier='trigger_servo',
            function=self.trigger_servo,
            description='Manually cycle the metronome servo, independent of position (for testing). '
                        'FRODO-specific - no host-side equivalent.',
            arguments=[]
        )

    # ------------------------------------------------------------------------------------------------------------------
    def go_to_position(self, x: float, y: float, psi: float | None = None,
                        speed: float | None = None, tolerance: float | None = None) -> dict:
        """Host-compatible entry point (matches ArtProject.go_to_position on hoca's template).
        Navigation here is grid/line-based, not free-space - so the requested world (x, y) is
        snapped to the nearest known grid node and handed to the existing turn-by-turn logic.
        psi/speed/tolerance are accepted for protocol compatibility but not used yet."""
        node = world_to_grid_node(x, y)
        if node not in CITY_MAP.values():
            self.frodo.logger.warning(f"go_to_position({x:.2f},{y:.2f}) -> nearest node {node} not in CITY_MAP")
        with self._lock:
            self.target_node = node
            self.stopped = False
        self.frodo.communication.send_event('art_project', {
            'type': 'move_started',
            'data': {'target': {'x': x, 'y': y, 'psi': psi, 'speed': speed, 'tolerance': tolerance}},
        })
        return {'accepted': True}

    # ------------------------------------------------------------------------------------------------------------------
    def stop(self):
        with self._lock:
            was_moving = self.target_node is not None
            self.stopped = True
            self.target_node = None
        if was_moving:
            pose_x, pose_y, pose_psi = self.pose_est.get()
            self.frodo.communication.send_event('art_project', {
                'type': 'aborted',
                'data': {'pose': {'x': float(pose_x), 'y': float(pose_y),
                                   'psi': float(pose_psi), 'time': time.time()}},
            })

    # ------------------------------------------------------------------------------------------------------------------
    def trigger_servo(self):
        with self._lock:
            self._manual_servo_request = True

    # ------------------------------------------------------------------------------------------------------------------
    def get_pose(self) -> dict:
        pose_x, pose_y, pose_psi = self.pose_est.get()
        return {'x': float(pose_x), 'y': float(pose_y), 'psi': float(pose_psi), 'time': time.time()}

    # ------------------------------------------------------------------------------------------------------------------
    def get_status(self) -> dict:
        with self._lock:
            state = self.state
            target_node = self.target_node
            current_node = self.current_node
        pose_x, pose_y, pose_psi = self.pose_est.get()
        target = None
        if target_node is not None:
            tx, ty = grid_node_to_world(target_node)
            target = {'x': tx, 'y': ty, 'psi': None, 'speed': None, 'tolerance': None}
        return {
            "state": state,
            "target": target,
            "pose": {"x": float(pose_x), "y": float(pose_y), "psi": float(pose_psi), "time": time.time()},
            # FRODO-specific extras (no host-side equivalent, kept for debugging)
            "target_node": list(target_node) if target_node is not None else None,
            "current_node": list(current_node) if current_node is not None else None,
        }

    # === STREAMING ====================================================================================================
    def _get_stream_frame(self):
        with self._stream_frame_lock:
            if self.frame_out is not None:
                return self.frodo.sensors.camera.getImageBufferBytes(self.frame_out)
            return None

    # === MULTI-ROBOT SENSING ==========================================================================================
    def _nearest_robot_ahead(self, marker_ids, latch_key):
        """(distance_m, bearing_rad) of the closest body marker in `marker_ids`,
        or None. bearing: + = left, - = right, 0 = straight ahead. Only markers
        in front (fwd > 0) count.

        Latched (see OTHER_ROBOT_LAST_SEEN_GRACE_S): if no marker is seen THIS
        frame, the last real sighting under `latch_key` is still returned for a
        grace period instead of immediately reporting "nothing there" - a
        vanished marker usually means the other robot turned its marker face
        away (e.g. pivoting in place), not that it actually left."""
        if not marker_ids:
            return None
        best = None
        try:
            sample = self.frodo.sensors.getSample()
        except Exception:
            sample = None
        if sample is not None:
            for m in sample.aruco_measurements:
                if m.measured_aruco_id not in marker_ids:
                    continue
                fwd, left = float(m.position[0]), float(m.position[1])
                if fwd <= 0.0:
                    continue
                dist = float(np.hypot(fwd, left))
                if best is None or dist < best[0]:
                    best = (dist, float(np.arctan2(left, fwd)))

        now = time.time()
        if best is not None:
            self._last_robot_seen[latch_key] = (now, best[0], best[1])
            return best
        seen = self._last_robot_seen.get(latch_key)
        if seen is not None and (now - seen[0]) <= OTHER_ROBOT_LAST_SEEN_GRACE_S:
            return (seen[1], seen[2])
        return None

    def _robot_emergency_ahead(self) -> bool:
        """Another robot (any priority) close and straight ahead -> hard stop."""
        hit = self._nearest_robot_ahead(self.other_robot_markers, "emergency")
        return (hit is not None
                and hit[0] <= OTHER_ROBOT_EMERGENCY_DISTANCE_M
                and abs(hit[1]) <= OTHER_ROBOT_AHEAD_BEARING)

    def _sighting_hold_active(self) -> bool:
        """Time-based safety stop (see OTHER_ROBOT_SIGHT_STOP_DELAY_S above):
        independent of distance - the instant a HIGHER-priority robot's FRONT
        marker (it's facing us: oncoming / crossing our path) is seen anywhere
        in the forward bearing cone, driving is allowed for at most
        OTHER_ROBOT_SIGHT_STOP_DELAY_S more seconds, then this returns True and
        keeps returning True for OTHER_ROBOT_SIGHT_HOLD_S regardless of what the
        marker does meanwhile (latched sighting via _nearest_robot_ahead, so a
        brief flicker doesn't reset the 1s clock either). Priority-gated (only
        higher_priority_front_markers, never a same/lower-priority robot's) so
        two robots facing each other don't BOTH pause and BOTH resume in lock-
        step with neither having yielded - the higher-priority one just keeps
        going (protected instead by the priority-independent
        _proximity_speed_cap()). A BACK marker never triggers this either way."""
        now = time.time()
        if self._sight_hold_until is not None:
            if now < self._sight_hold_until:
                return True
            self._sight_hold_until = None
            self._other_seen_since = None

        hit = self._nearest_robot_ahead(self.higher_priority_front_markers, "sighting")
        seen_now = hit is not None and abs(hit[1]) <= OTHER_ROBOT_AHEAD_BEARING
        if not seen_now:
            self._other_seen_since = None
            return False

        if self._other_seen_since is None:
            self._other_seen_since = now
            return False

        if now - self._other_seen_since >= OTHER_ROBOT_SIGHT_STOP_DELAY_S:
            self._sight_hold_until = now + OTHER_ROBOT_SIGHT_HOLD_S
            return True
        return False

    def _forward_cell_blocked(self) -> bool:
        """Some other robot (ANY priority) occupies the cell I'd enter by going
        straight. NOT priority-gated: priority decides who yields/stops when
        two robots are actively contesting the same cell/intersection (see
        _sighting_hold_active), but a robot sitting still in my literal path is
        an obstacle regardless of rank - even the top-priority robot has to
        route around it or it just sits there forever creeping to a stop via
        _proximity_speed_cap and never reaching its target (observed in the
        field: frodo1, priority 0, driving straight through frodo4 parked
        directly on frodo1's leg - frodo1 needs to detour, not just slow down
        and wait)."""
        hit = self._nearest_robot_ahead(self.other_robot_markers, "forward")
        return (hit is not None
                and hit[0] <= OTHER_ROBOT_BLOCK_DISTANCE_M
                and abs(hit[1]) <= OTHER_ROBOT_AHEAD_BEARING)

    def _proximity_speed_cap(self, desired_speed: float):
        """Smooth, priority-INDEPENDENT braking against the closest other
        robot's marker (any face, any priority) - see the PROXIMITY GOVERNOR
        comment above PROXIMITY_STOP_GAP_M. This is what protects a HIGHER-
        priority robot (which never engages _sighting_hold_active/stops for a
        lower-priority one) from driving straight into one sitting in its path
        - e.g. frodo4 correctly waiting somewhere on frodo1's straight leg,
        before frodo1's own turn. Unlike the hold, there's no timer here: the
        cap is recomputed fresh every frame from the live distance, so it
        can't fall out of sync and burst back to full speed the way a fixed-
        duration hold could. Linear ramp: at/inside PROXIMITY_STOP_GAP_M -> 0,
        at/beyond PROXIMITY_FULL_SPEED_GAP_M -> uncapped. Returns
        `desired_speed` unmodified if no other robot marker is (or was
        recently, via the _nearest_robot_ahead latch) seen ahead."""
        hit = self._nearest_robot_ahead(self.other_robot_markers, "proximity")
        if hit is None or abs(hit[1]) > OTHER_ROBOT_AHEAD_BEARING:
            return desired_speed
        dist = hit[0]
        if dist <= PROXIMITY_STOP_GAP_M:
            return 0.0
        if dist >= PROXIMITY_FULL_SPEED_GAP_M:
            return desired_speed
        frac = (dist - PROXIMITY_STOP_GAP_M) / (PROXIMITY_FULL_SPEED_GAP_M - PROXIMITY_STOP_GAP_M)
        return desired_speed * frac

    # ------------------------------------------------------------------------------------------------------------------
    def _cooperative_next_step(self, current_coord, target_node):
        """Approach B (see the big comment above CITY_MAP): Cooperative A* over
        every robot's latest broadcast (current_node, target_node) from peer_sync.py.

        Returns a cardinal direction ("EAST"/...) for the first hop of MY OWN plan,
        "WAIT" if the plan has me yield a step, or None if there is nothing to say
        (no peer_sync, no fresh peers, or planning failed) - the caller then falls
        back to the plain reactive next_heading()/blocked_cells logic unchanged."""
        if self.peer_sync is None:
            return None
        peers = self.peer_sync.get_fresh_peers()
        if not peers:
            return None

        my_id = self.frodo.common.id
        agents = {my_id: (current_coord, target_node), **peers}
        # Same right-of-way convention as ROBOT_PRIORITY: earlier = planned first =
        # higher priority. Any id not in ROBOT_PRIORITY (shouldn't happen) goes last.
        order = [rid for rid in ROBOT_PRIORITY if rid in agents]
        order += [rid for rid in agents if rid not in order]

        plans = plan_cooperative_paths(agents, GRID_NODES, priority_order=order)
        my_path = plans.get(my_id)
        if not my_path or len(my_path) < 2:
            return None   # no conflict-free path within the search horizon - Approach A guards

        next_node, _ = my_path[1]
        if next_node == current_coord:
            return "WAIT"
        return direction_between(current_coord, next_node)

    # === MAIN LOOP ====================================================================================================
    def run(self):
        frodo = self.frodo

        self.streamer.start()
        ip = getInterfaceIP("wlan0")
        print(f"\n---> LIVE: http://{ip}:5001/preview <--- \n")

        # Approach B peer link (see _cooperative_next_step) - broadcasts our own
        # (current_node, target_node) and listens for the other robots' the same way.
        # Not safety-critical (Approach A's camera check still guards regardless), so
        # a failure here (e.g. no WiFi IP yet) just disables Approach B, not the robot.
        try:
            self.peer_sync = PeerSync(
                robot_id=frodo.common.id,
                state_fn=lambda: (self.current_node, self.target_node),
                address=ip,
            )
            self.peer_sync.start()
        except OSError as e:
            frodo.logger.warning(f"PeerSync unavailable ({e}) - Approach B disabled, Approach A still active")
            self.peer_sync = None

        # ---------------- PARAMETERS ----------------
        # Every per-robot value comes from the one PER-ROBOT TUNING block at the top
        # (TUNING_DEFAULTS + ROBOT_TUNING[id]). Nothing per-robot is hard-coded here.
        tune = tuning_for(frodo.common.id)
        _overrides = ROBOT_TUNING.get(frodo.common.id, {})
        if _overrides:
            print(f"Per-robot tuning for {frodo.common.id!r}: "
                  + ", ".join(f"{k}={tune[k]}" for k in sorted(_overrides)))
        else:
            print(f"Per-robot tuning for {frodo.common.id!r}: (none - using TUNING_DEFAULTS)")

        CURRENT_TARGET_COLOR = "pink"
        kp = tune["line_kp"]
        kd = tune["line_kd"]
        base_speed = tune["base_speed"]
        track_width = tune["track_width"]
        turn_kp = tune["turn_kp"]
        turn_ki = tune["turn_ki"]
        turn_angle = np.radians(tune["turn_angle_deg"])
        aruco_min_size_px = tune["aruco_min_size_px"]
        aruco_y_min = tune["aruco_y_min"]
        aruco_x_min = tune["aruco_x_min"]
        aruco_x_max = tune["aruco_x_max"]
        advance_time = tune["advance_time"]
        advance_time_boundary = tune["advance_time_boundary"]
        marker_lost_dist_m = tune["marker_lost_dist_m"]

        # ROBOT_HEADING is NOT guessed/assumed - it's derived from the robot's own
        # first two marker fixes (see HEADING CALIBRATION below), so it's correct
        # regardless of which physical direction the robot happens to be facing
        # when placed on the line (a blind "assume I'm already driving toward the
        # target" guess was tried before and drove the robot the wrong way whenever
        # it was placed facing away from the target).
        ROBOT_HEADING = None
        LAST_SEEN_ID = None
        first_fix_coord = None     # grid coord of the very first marker seen
        heading_known = False      # True once ROBOT_HEADING has been derived from 2 real fixes
        # ROBOT_HEADING is set once at calibration and then ONLY at an explicit turn
        # (closed-loop psi PI, lands within ~2 deg every time). There is NO continuous
        # node-to-node "re-sync" any more: it used to fire on diagonal / skipped-node
        # reads ((3,2)->(2,3)) and "correct" the heading to a bogus value, which then
        # triggered phantom 180 reversals and sent the robot staircasing off the grid.
        line_lost_since = None     # FOLLOWING only - see LINE_LOST_CREEP_TIME above
        line_lost_reported = False # so the one-shot LINE_LOST error event below doesn't spam every frame
        prev_line_error = None     # for the steering D term - reset to None whenever the line is
        prev_line_error_t = 0.0    # not seen so a stale delta can't kick the wheels on re-acquire

        # TURNING pivots the robot roughly in place (no forward speed) - it does NOT
        # drive the robot onto the new-color lane by itself. Since the axis color is
        # fixed by design (X=pink, Y=green) and switches the instant the turn starts
        # (see the color switch at ADVANCING_TO_TURN below), the robot needs a short
        # advance AFTER the turn too - same idea as ADVANCE_TIME before the turn - to
        # physically get onto the new lane before FOLLOWING starts looking for it.
        # This advance is NOT purely blind: the pivot is never perfectly centered on
        # the intersection, so a fixed blind time either ends before the new lane is
        # even in frame (instant LINE LOST the moment FOLLOWING takes over) or drives
        # past it. Instead we check the camera every frame here too and drop into
        # FOLLOWING the moment the new-color line is actually visible; MIN_TIME is
        # just a brief blind floor (avoid reacting to a stray glimpse before the robot
        # has moved off the pivot point at all), MAX_TIME is a ceiling so we don't
        # drive forever if the lane is genuinely not reachable this way.
        # ADVANCING_FROM_TURN: a pivot turn is never perfectly centred on the
        # intersection (and TURN_ANGLE deliberately over-commands), so afterwards
        # the robot is offset from / angled off the new-colour line. Driving
        # straight open-loop CANNOT correct a lateral offset - the robot then
        # drives parallel to the line, never re-centres, and misses every marker
        # from there on (seen in the field: turned NORTH at (8,0), then drove the
        # whole column reading nothing, ending at (8,5)). So: as soon as the new
        # line is visible ANYWHERE in the ROI, actively steer onto it; only hand
        # over to FOLLOWING once it is roughly centred. If it never comes into
        # view by driving straight, sweep left/right in place to find it.
        TURN_EXIT_BLIND_CREEP_TIME = 0.25  # brief blind creep off the pivot point first
        TURN_EXIT_STRAIGHT_TIME = 1.2      # then creep straight this long, hoping the line appears
        TURN_EXIT_SWEEP_TIME = 2.0         # then sweep +-this long each side to find it
        TURN_EXIT_CENTER_PX = 25           # |error| under this = "centred enough" to start FOLLOWING
                                           # (60 -> 40 -> 25: handing over further off the new line
                                           #  meant FOLLOWING started from an angled pose and wove)
        TURN_EXIT_CENTER_HOLD = 0.3        # ...for this long (0.2 -> 0.3: let the pose settle so
                                           #  FOLLOWING starts roughly parallel, not mid-correction)
        turn_exit_start = None
        turn_exit_centered_since = None
        turn_exit_sweep_start = None

        # METRONOME state: the target marker was ACCEPTED, the servo is OPEN and the
        # robot line-follows slowly forward until this deadline, then closes the
        # servo, stops, and goes DONE (see the METRONOME branch in LINE FOLLOWING).
        metronome_close_time = 0.0
        arriving_node = None          # grid node currently being arrived at (METRONOME)
        last_accepted_marker_time = None   # for the drifted/blind guard below
        path_len_at_last_marker = 0.0      # pose_est.path_length when we last got a fix
        arrived_node = None          # the grid node we finished a mission at (see DONE)

        # ADVANCING_TO_TURN: a turn decision was made, the robot keeps driving straight
        # for advance_time (per-robot), then the turn actually starts (see the note above).
        pre_turn_next_state = None   # "TURNING_LEFT" | "TURNING_RIGHT"
        pre_turn_start = None
        advance_time_this_turn = advance_time   # per-turn (advance_time_boundary at grid edges)

        turn_start = None
        turn_last_time = None
        turn_i_acc = 0.0
        turn_settle_start = None
        turn_target_psi = None
        TURN_TARGET_HEADING = None   # set once a turn decision is made (see the decision block)
        STOP_REASON = None      # "TURN_TIMEOUT" - distinguishes the STOPPED message
        last_turn_log = 0.0     # throttles the diagnostic log while turning

        self.pose_est.start()

        print("Remote-controllable Grid Navigation Started (ArUco Marker Mode).")
        print("Waiting for go_to_position(x, y) from the host. Press Ctrl+C to stop.\n")

        shape_printed = False
        last_aruco_log = 0.0
        last_robot_hold_log = 0.0
        last_robot_sensor_log = 0.0

        try:
            while True:
                frame = frodo.sensors.camera.events.frame.get_data()
                if frame is None:
                    continue

                display_frame = frame.copy()
                h_img, w_img = display_frame.shape[:2]

                if not shape_printed:
                    print(f">>> Frame size: {w_img} x {h_img}")
                    shape_printed = True

                # ---------------- EKF POSE OVERLAY (visual sanity check only) ----------------
                pose_x, pose_y, pose_psi = self.pose_est.get()
                cv2.putText(display_frame,
                            f"EKF x={pose_x:+.2f}m y={pose_y:+.2f}m psi={np.degrees(pose_psi):+.0f}deg",
                            (10, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                # ---------------- MANUAL STOP (from the host) ----------------
                # Aborts whatever it was doing (turn/approach/park included) and holds
                # in place. go_to_position() clears the flag and resumes FOLLOWING.
                with self._lock:
                    manual_stopped = self.stopped
                if manual_stopped:
                    frodo.control.setTrackSpeed(0.0, 0.0)
                    self.state = "IDLE"
                    cv2.putText(display_frame, "STOPPED (manual)", (60, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                    with self._stream_frame_lock:
                        self.frame_out = display_frame
                    time.sleep(0.05)
                    continue
                elif self.state == "IDLE":
                    self.state = "FOLLOWING"

                # ---------------- DONE (arrived at the assigned target + servo fired) ----------------
                # Hold in place. A fresh go_to_position() (sets a new target_node and
                # clears self.stopped) resumes navigation.
                if self.state == "DONE":
                    frodo.control.setTrackSpeed(0.0, 0.0)
                    with self._lock:
                        new_target = self.target_node
                    if new_target is not None and new_target != arrived_node:
                        print(f"New target {new_target} received - resuming navigation")
                        self.state = "FOLLOWING"
                    else:
                        cv2.putText(display_frame, "MISSION COMPLETE", (40, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 0), 3)
                        with self._stream_frame_lock:
                            self.frame_out = display_frame
                        time.sleep(0.05)
                        continue

                # ---------------- STOPPED (unrecoverable failure) ----------------
                if self.state == "STOPPED":
                    frodo.control.setTrackSpeed(0.0, 0.0)
                    cv2.putText(display_frame, f"FAILED: {STOP_REASON}", (60, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    print(f"\n!!! MISSION FAILED ({STOP_REASON}). Motors locked. !!!")
                    with self._stream_frame_lock:
                        self.frame_out = display_frame
                    break

                # ---------------- DEBUG: raw frodo.sensors body-marker readings ----------------
                # All the collision-avoidance checks below (_robot_emergency_ahead,
                # _sighting_hold_active, _forward_cell_blocked, _proximity_speed_cap)
                # read frodo.sensors.getSample().aruco_measurements, NOT the
                # detected_markers/console prints from the local self.aruco_detector
                # further down - those are two independent detection pipelines on two
                # different cv2.aruco.ArucoDetector instances. A robot can be clearly
                # visible in the "ArUco NNN ... TOO FAR/NOT ON MAP" floor-marker prints
                # while frodo.sensors reports nothing for it at all (allowlist gap,
                # is_mostly_z_axis() rejecting the pose, marker_size mismatch skewing
                # the computed distance, ...) - print what THIS pipeline actually sees
                # for other robots' body markers so a collision can be diagnosed
                # instead of guessed at.
                _dbg_now = time.time()
                if _dbg_now - last_robot_sensor_log > 0.5:
                    last_robot_sensor_log = _dbg_now
                    try:
                        _dbg_sample = self.frodo.sensors.getSample()
                    except Exception as _dbg_e:
                        print(f"[{_dbg_now:.1f}] sensors.getSample() FAILED: {_dbg_e}")
                        _dbg_sample = None
                    if _dbg_sample is not None:
                        _dbg_hits = [m for m in _dbg_sample.aruco_measurements
                                     if m.measured_aruco_id in self.other_robot_markers]
                        if _dbg_hits:
                            for m in _dbg_hits:
                                fwd, left = float(m.position[0]), float(m.position[1])
                                print(f"[{_dbg_now:.1f}] SENSOR body-marker {m.measured_aruco_id}: "
                                      f"fwd={fwd:+.2f}m left={left:+.2f}m "
                                      f"dist={float(np.hypot(fwd, left)):.2f}m")
                        else:
                            print(f"[{_dbg_now:.1f}] SENSOR body-marker: none in "
                                  f"aruco_measurements (watching for {sorted(self.other_robot_markers)})")

                # ---------------- EMERGENCY STOP (another robot dead ahead) ----------------
                # Symmetric (ignores priority) - just don't hit each other. Only
                # while driving forward in a corridor; the short in-place TURNING
                # states are left alone so the psi PI controller isn't disturbed.
                # Two independent triggers, either one stops us:
                #   - distance: within OTHER_ROBOT_EMERGENCY_DISTANCE_M right now
                #   - sighting: seen at ALL for OTHER_ROBOT_SIGHT_STOP_DELAY_S, then
                #     held for OTHER_ROBOT_SIGHT_HOLD_S (see _sighting_hold_active) -
                #     the distance check alone fires too late (camera/processing lag
                #     + momentum already closes the gap by the time it trips).
                if self.state in ("FOLLOWING", "METRONOME", "ADVANCING_TO_TURN",
                                  "ADVANCING_FROM_TURN"):
                    _emergency_hit = self._robot_emergency_ahead()
                    _sighting_hit = self._sighting_hold_active()
                    if _emergency_hit or _sighting_hit:
                        frodo.control.setTrackSpeed(0.0, 0.0)
                        cv2.putText(display_frame, "ROBOT AHEAD - HOLDING", (30, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        if time.time() - last_robot_hold_log > 1.0:
                            last_robot_hold_log = time.time()
                            _reason = "distance" if _emergency_hit else "sighting-hold"
                            print(f"[{last_robot_hold_log:.1f}] EMERGENCY HOLD ({_reason})")
                        with self._stream_frame_lock:
                            self.frame_out = display_frame
                        time.sleep(0.05)
                        continue

                # ================= ARUCO DETECTION =================
                # The camera can be configured for a gray output (image_format="gray"
                # in robot/definitions.py) - in that case the frame is already
                # single-channel. Same guard as aruco_detector.py / line_following.py.
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                detected_markers, _aruco_scan_y0 = detect_markers(self.aruco_detector, gray, ARUCO_SCAN_Y_MIN)

                now = time.time()
                log_aruco = (now - last_aruco_log) > ARUCO_LOG_PERIOD
                if log_aruco:
                    last_aruco_log = now
                    if not detected_markers:
                        print(f"[{now:.1f}] ArUco: none visible")

                # ---------------- MANUAL SERVO TRIGGER (from the host) ----------------
                # The servo otherwise only fires on arrival at the target node
                # (PARKING/DONE block) - there is no floor marker for it any more.
                if self.state == "FOLLOWING":
                    with self._lock:
                        manual_request = self._manual_servo_request
                        if manual_request:
                            self._manual_servo_request = False
                    if manual_request and self.state == "FOLLOWING":
                        print(f"[{now:.1f}] Manual trigger_servo() requested - cycling servo in place")
                        frodo.control.setTrackSpeed(0.0, 0.0)
                        self.servo_trigger.rotate_to_trigger()
                        self.servo_trigger.rotate_to_home()
                        self.frodo.communication.send_event('art_project_servo_triggered', {'node': None})
                        print(">>> MANUAL SERVO CYCLE COMPLETE.")

                for detected_id, marker_corners in detected_markers:
                    # marker_corners: shape (1, 4, 2) - pixel coordinates relative to the full frame
                    x, y, w_r, h_r, center_x, center_y, aruco_size = marker_bbox(marker_corners)

                    cv2.rectangle(display_frame, (x, y), (x + w_r, y + h_r), (255, 0, 255), 3)
                    cv2.putText(display_frame, f"{detected_id} ({aruco_size}px)", (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                    # --- filter decision --- (window is per-robot: aruco_y_min / _x_min / _x_max)
                    if center_y < aruco_y_min * h_img or aruco_size < aruco_min_size_px:
                        status = "TOO FAR"
                    elif center_x < aruco_x_min * w_img or center_x > aruco_x_max * w_img:
                        status = "WRONG LANE"
                    elif detected_id not in CITY_MAP:
                        status = "NOT ON MAP"
                    elif detected_id == LAST_SEEN_ID:
                        status = "already processed"
                    else:
                        # ACCEPTED regardless of FSM state - POSITION TRACKING below
                        # must run for a marker the robot physically drives over even
                        # while ADVANCING_FROM_TURN / TURNING (it used to be dropped
                        # with "state=..." and the robot never recorded that node ->
                        # missed markers right after a turn). The turn/target DECISION
                        # is still gated on FOLLOWING further down.
                        status = "ACCEPTED"

                    if log_aruco:
                        print(f"[{now:.1f}] ArUco {detected_id!r:6} "
                              f"center=({center_x:4d},{center_y:4d}) size={aruco_size:3d}px "
                              f"-> {status}")

                    if status == "TOO FAR":
                        cv2.putText(display_frame, f"ArUco {detected_id}: TOO FAR", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        continue

                    if status == "WRONG LANE":
                        cv2.putText(display_frame, f"ArUco {detected_id}: WRONG LANE", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                        continue

                    cv2.putText(display_frame, f"ArUco ID: {detected_id} DETECTED", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if status != "ACCEPTED":
                        continue

                    # ---------------- POSITION TRACKING (always active, independent of target) ----------------
                    # Must update even with no target set - otherwise LAST_SEEN_ID never
                    # leaves None, which makes the "probe pink vs green" fallback in LINE
                    # FOLLOWING below (gated on LAST_SEEN_ID is None) re-run every single
                    # frame instead of just once. If both colors are ever visible at the
                    # same time (e.g. near a grid intersection), that flip-flops
                    # CURRENT_TARGET_COLOR frame to frame and the robot never drives
                    # straight (observed in the field: spins in place near markers).
                    current_coord = CITY_MAP[detected_id]
                    last_accepted_marker_time = now
                    path_len_at_last_marker = self.pose_est.path_length
                    if detected_id != LAST_SEEN_ID:
                        LAST_SEEN_ID = detected_id
                        self.current_node = current_coord
                        print(f"\n[{now:.1f}] NODE {detected_id} -> {current_coord}")

                        # ---------------- HEADING CALIBRATION (first two real fixes) ----------------
                        # Do NOT assume "whatever direction I'm currently driving must be
                        # the direction I want" - that depends on which way the robot
                        # happened to be physically facing when placed on the line, which
                        # is not guaranteed to already point toward the goal (observed in
                        # the field: robot placed facing away from the target just kept
                        # "confirming" that wrong direction forever). Instead figure out
                        # the REAL travel direction from two actual marker fixes.
                        if not heading_known:
                            if first_fix_coord is None:
                                first_fix_coord = current_coord
                                print(f"First fix: position={current_coord} - waiting for a second "
                                      f"marker to determine the real heading")
                            else:
                                dxr = current_coord[0] - first_fix_coord[0]
                                dyr = current_coord[1] - first_fix_coord[1]
                                if dxr == 0 and dyr == 0:
                                    print(f"!!! Heading calibration: second fix {current_coord} is the same "
                                          f"as the first - retrying with the next marker")
                                    first_fix_coord = current_coord
                                else:
                                    if abs(dxr) >= abs(dyr):
                                        ROBOT_HEADING = "EAST" if dxr > 0 else "WEST"
                                    else:
                                        ROBOT_HEADING = "NORTH" if dyr > 0 else "SOUTH"
                                    CURRENT_TARGET_COLOR = "pink" if ROBOT_HEADING in ("EAST", "WEST") else "green"
                                    heading_known = True
                                    print(f"Heading calibrated from real movement {first_fix_coord} -> "
                                          f"{current_coord}: {ROBOT_HEADING} (color={CURRENT_TARGET_COLOR})")
                        # NOTE: no continuous node-to-node heading re-sync - heading is
                        # only ever changed here (calibration) and at an explicit turn.

                        # If the turn-exit drive reached the NEXT grid node before it
                        # finished centering, we're clearly back on the new line - snap
                        # to FOLLOWING now so the decision for THIS node runs this frame
                        # (otherwise a turn needed right after another turn is missed).
                        if self.state == "ADVANCING_FROM_TURN":
                            print(f"  [turn exit] reached node {current_coord} -> FOLLOWING")
                            self.state = "FOLLOWING"
                            turn_exit_start = None
                            turn_exit_centered_since = None
                            turn_exit_sweep_start = None

                    # ---------------- DECISION MAKING ----------------
                    # Only ACT on a decision while actually FOLLOWING (position/heading
                    # tracking above already ran regardless of state).
                    if self.state != "FOLLOWING":
                        break

                    # No target set yet -> nothing to decide, just keep following the
                    # line (no turns). Waits for the host to call go_to_position().
                    with self._lock:
                        target_node = self.target_node
                    if target_node is None:
                        break

                    if current_coord == target_node:
                        # Target marker ACCEPTED (already close - the accept filter
                        # requires a real pixel size / lane / y-min). Open the servo
                        # NOW (metronome starts), keep line-following slowly forward
                        # for SERVO_OPEN_DRIVE_TIME, then close it + stop + DONE.
                        arriving_node = target_node
                        print(f"*** TARGET {target_node} marker {detected_id} ACCEPTED ({aruco_size}px) "
                              f"- opening servo, running the metronome for {SERVO_OPEN_DRIVE_TIME:.1f}s ***")
                        frodo.control.setTrackSpeed(0.0, 0.0)
                        if SERVO_ON_ARRIVAL:
                            self.servo_trigger.rotate_to_trigger()   # blocks ~settle_time, robot stopped
                            self.frodo.communication.send_event('art_project_servo_triggered',
                                                                {'node': list(target_node)})
                        metronome_close_time = time.time() + SERVO_OPEN_DRIVE_TIME
                        self.state = "METRONOME"
                        break

                    if not heading_known:
                        # Real heading isn't known yet (only one marker seen so far) -
                        # can't safely decide CONTINUE vs TURN. Wait for the second fix.
                        break

                    # ---------------- NEXT STEP (fewest-turns shortest path) ----------------
                    # BFS with prefer=ROBOT_HEADING => drive straight along one axis
                    # until in line with the target, then turn ONCE onto the other
                    # axis (grid_nav.next_heading). No cooperative A* / peer planning
                    # any more - the path is deterministic. The only reroute is the
                    # reactive camera one: if ANY other robot's body marker (any
                    # priority) is sitting in the cell straight ahead, drop that cell
                    # so BFS finds a detour (the symmetric EMERGENCY stop and
                    # _proximity_speed_cap still guard against actual contact). Not
                    # priority-gated - a robot blocking my literal path is an obstacle
                    # to route around regardless of rank, even for the top-priority
                    # robot (which otherwise never yields to anyone).
                    blocked_cells = set()
                    if self._forward_cell_blocked():
                        step = DIRECTIONS.get(ROBOT_HEADING, (0, 0))
                        ahead_cell = (current_coord[0] + step[0], current_coord[1] + step[1])
                        blocked_cells.add(ahead_cell)
                        print(f"  [avoidance] robot ahead - routing around {ahead_cell}")

                    desired_heading = next_heading(current_coord, target_node, GRID_NODES,
                                                   blocked=blocked_cells, prefer=ROBOT_HEADING)
                    if desired_heading is None and blocked_cells:
                        # No detour exists (rare on an open grid). Fall back to the
                        # unblocked shortest path - the EMERGENCY stop still keeps
                        # the robots from actually touching.
                        print("  [avoidance] no detour - holding to shortest path, emergency-stop will guard")
                        desired_heading = next_heading(current_coord, target_node, GRID_NODES,
                                                       prefer=ROBOT_HEADING)
                    if desired_heading is None:
                        print(f"!!! No path from {current_coord} to {target_node} - staying put")
                        break

                    print(f"Heading: {ROBOT_HEADING} -> Desired: {desired_heading}")

                    # ---------------- 180 degree ("wrong way on this axis") handling ----------------
                    # The robot can't pivot 180 in place on the line (and a blind
                    # 180 loses the line entirely). If BFS wants a full reversal -
                    # e.g. dropped facing WEST but the target is EAST - turn 90
                    # toward whichever perpendicular axis moves us CLOSER to the
                    # target; the remaining 90 is taken at a later node once the
                    # robot is heading along the perpendicular. If the target is
                    # dead behind on this exact axis (no perpendicular reduces the
                    # distance), still turn 90 toward any in-grid perpendicular
                    # neighbor so the robot works its way around instead of driving
                    # off the end of the row/column.
                    if desired_heading == _OPPOSITE_HEADING.get(ROBOT_HEADING):
                        cx, cy = current_coord
                        tx, ty = target_node
                        if ROBOT_HEADING in ("EAST", "WEST"):
                            if ty > cy:
                                options = ["NORTH", "SOUTH"]
                            elif ty < cy:
                                options = ["SOUTH", "NORTH"]
                            else:
                                options = ["NORTH", "SOUTH"]
                        else:
                            if tx > cx:
                                options = ["EAST", "WEST"]
                            elif tx < cx:
                                options = ["WEST", "EAST"]
                            else:
                                options = ["EAST", "WEST"]
                        detour_heading = None
                        for cand in options:
                            cdx, cdy = DIRECTIONS[cand]
                            if (cx + cdx, cy + cdy) in GRID_NODES:
                                detour_heading = cand
                                break
                        if detour_heading is None:
                            print(f"!!! 180 required at {current_coord} and no in-grid "
                                  f"perpendicular neighbor - staying put")
                            break
                        print(f"  [180] reversal needed ({ROBOT_HEADING} -> {desired_heading}); "
                              f"turning {detour_heading} now, remaining turn at the next node")
                        desired_heading = detour_heading

                    if ROBOT_HEADING == desired_heading:
                        print("Action: CONTINUE")
                    else:
                        # The turn does NOT start immediately: since ARUCO_MIN_SIZE_PX
                        # already makes the decision fire close to the intersection,
                        # the robot first drives straight for ADVANCE_TIME OPEN-LOOP
                        # (without looking at the image) to reach the exact center of
                        # the intersection, then the turn actually begins.
                        TURN_TARGET_HEADING = desired_heading
                        if (ROBOT_HEADING, desired_heading) in [
                            ("EAST", "SOUTH"), ("SOUTH", "WEST"),
                            ("WEST", "NORTH"), ("NORTH", "EAST"),
                        ]:
                            pre_turn_next_state = "TURNING_RIGHT"
                            print("Action: TURN RIGHT (after short advance)")
                        elif (ROBOT_HEADING, desired_heading) in [
                            ("EAST", "NORTH"), ("NORTH", "WEST"),
                            ("WEST", "SOUTH"), ("SOUTH", "EAST"),
                        ]:
                            pre_turn_next_state = "TURNING_LEFT"
                            print("Action: TURN LEFT (after short advance)")
                        else:
                            # Should be unreachable: a true reversal is rewritten to a
                            # 90 degree detour above, and every remaining (heading,
                            # desired) pair is a left or right quarter turn.
                            print(f"!!! Unexpected turn {ROBOT_HEADING} -> {desired_heading} "
                                  f"- not supported, staying put")
                            pre_turn_next_state = None

                        if pre_turn_next_state is not None:
                            # Pre-turn open-loop advance to the intersection centre.
                            # At a BOUNDARY node the cell straight ahead doesn't
                            # exist - a full advance_time there drives the robot off
                            # the end of the grid, so the pivot happens away from the
                            # new line and it never re-acquires it (seen in the field
                            # at (8,0)). Use advance_time_boundary in that case.
                            _odx, _ody = DIRECTIONS.get(ROBOT_HEADING, (0, 0))
                            _ahead = (current_coord[0] + _odx, current_coord[1] + _ody)
                            advance_time_this_turn = advance_time if _ahead in GRID_NODES else advance_time_boundary
                            # NOTE: CURRENT_TARGET_COLOR is NOT changed here - the robot
                            # is still coming from the old position during
                            # ADVANCING_TO_TURN, the color only switches once the turn
                            # actually starts.
                            # Only update ROBOT_HEADING once a turn is actually going to
                            # be executed - in the 180 degree ("not supported") case no
                            # turn happens at all, so the robot keeps physically moving
                            # in the OLD direction. Updating ROBOT_HEADING anyway (as
                            # before) desynced it from the robot's real orientation and
                            # made later decisions silently report CONTINUE/matching
                            # instead of repeatedly flagging the unsupported 180 turn.
                            ROBOT_HEADING = desired_heading
                            pre_turn_start = now
                            last_accepted_marker_time = now  # restart the "lost" clock for the post-turn leg
                            path_len_at_last_marker = self.pose_est.path_length
                            self.state = "ADVANCING_TO_TURN"

                    break

                # ---------------- DRIFTED / OVERSHOT GUARD ----------------
                # Navigating with a target, heading known, but no accepted marker
                # after driving marker_lost_dist_m since the last one (or stuck and
                # blind for MARKER_TIMEOUT_S) -> we've left the grid / overshot with
                # nothing to stop us, OR the camera simply can't read the markers.
                if (self.state == "FOLLOWING" and heading_known
                        and last_accepted_marker_time is not None):
                    with self._lock:
                        _tgt = self.target_node
                    _dist_since = self.pose_est.path_length - path_len_at_last_marker
                    _blind_time = now - last_accepted_marker_time
                    if _tgt is not None and (_dist_since > marker_lost_dist_m
                                             or _blind_time > MARKER_TIMEOUT_S):
                        print(f"!!! No marker for {_dist_since:.2f} m / {_blind_time:.1f}s while navigating "
                              f"to {_tgt} - lost, or the camera can't read the markers.")
                        frodo.control.setTrackSpeed(0.0, 0.0)
                        gx, gy, gpsi = self.pose_est.get()
                        self.frodo.communication.send_event('art_project', {
                            'type': 'error',
                            'data': {'type': 'LOST',
                                     'message': f'no marker for {_dist_since:.2f} m - lost / overshot / bad detection',
                                     'pose': {'x': float(gx), 'y': float(gy), 'psi': float(gpsi), 'time': time.time()}},
                        })
                        STOP_REASON = "NO MARKERS"
                        self.state = "STOPPED"

                # ================= LINE FOLLOWING =================
                # CURRENT_TARGET_COLOR is a PURE FUNCTION of ROBOT_HEADING now
                # (E/W = pink X-axis, N/S = green Y-axis). It is set at calibration
                # and at every turn - never switched here. The old "current colour
                # lost but a clear other-colour line is here -> switch" recovery is
                # GONE: at every grid intersection the crossing line legitimately
                # has a big contour, so it fired constantly on the green legs,
                # flipped the robot onto pink, and sent it staircasing off the grid.
                if self.state in ("FOLLOWING", "METRONOME"):
                    if LAST_SEEN_ID is None and self.state == "FOLLOWING" and not heading_known:
                        # No marker read yet -> we don't know which colour line we're
                        # on. Follow whichever is visible until the first fix; heading
                        # calibration then locks the colour to the real axis.
                        best_color, best_found = None, None
                        for probe_color in ("pink", "green"):
                            found = find_line(get_color_mask(frame, probe_color))
                            if found is not None and (best_found is None or found[1] > best_found[1]):
                                best_color, best_found = probe_color, found
                        if best_color is not None:
                            CURRENT_TARGET_COLOR = best_color

                    mask = get_color_mask(frame, CURRENT_TARGET_COLOR)          # RAW frame!
                    error, line_detected, _area = calculate_deviation(mask, display_frame)

                    # METRONOME: servo is OPEN, keep line-following slowly forward so
                    # the metronome runs while moving (SERVO_OPEN_DRIVE_TIME), then
                    # close the servo + stop + DONE.
                    follow_speed = (base_speed * 0.5) if self.state == "METRONOME" else base_speed
                    # Smooth, priority-independent brake against whatever robot
                    # marker is closest ahead - catches what the priority-gated
                    # sighting-hold above deliberately leaves unguarded (a
                    # higher-priority robot approaching a lower-priority one).
                    follow_speed = self._proximity_speed_cap(follow_speed)

                    if line_detected:
                        line_lost_since = None
                        line_lost_reported = False
                        # D term: rate of change of the pixel error. Reset (prev=None)
                        # whenever the line was not seen, so re-acquiring it after a
                        # gap doesn't produce a huge spurious derivative.
                        d_err = 0.0
                        if prev_line_error is not None and (now - prev_line_error_t) > 1e-3:
                            d_err = (error - prev_line_error) / (now - prev_line_error_t)
                        prev_line_error, prev_line_error_t = error, now
                        forward_speed, angular_speed = proportional_controller(
                            error, kp, follow_speed, d_error=d_err, kd=kd)
                        v_left, v_right = calculate_wheel_speeds(forward_speed, angular_speed, track_width)
                        frodo.control.setTrackSpeed(v_left, v_right)
                    elif self.state == "METRONOME":
                        # line lost right on top of the target marker is normal - just
                        # creep straight, the close timer ends it in a moment anyway
                        prev_line_error = None
                        frodo.control.setTrackSpeed(follow_speed, follow_speed)
                    elif self.state == "FOLLOWING" and \
                            (now - (line_lost_since or now)) < LINE_LOST_CREEP_TIME:
                        if line_lost_since is None:
                            line_lost_since = now
                        prev_line_error = None
                        frodo.control.setTrackSpeed(base_speed, base_speed)
                        cv2.putText(display_frame, "LINE LOST - CREEPING", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    else:
                        prev_line_error = None
                        frodo.control.setTrackSpeed(0.0, 0.0)
                        # FOLLOWING, past the creep window, line still not found: no automatic
                        # recovery beyond this point (needs a human nudge or the line reappearing
                        # on its own) - report it to the host ONCE (not every frame at ~20 Hz) so
                        # it's at least visible that the robot stalled, even though we can't fix
                        # it here.
                        if self.state == "FOLLOWING" and line_lost_since is not None and not line_lost_reported:
                            line_lost_reported = True
                            stuck_pose_x, stuck_pose_y, stuck_pose_psi = self.pose_est.get()
                            self.frodo.communication.send_event('art_project', {
                                'type': 'error',
                                'data': {
                                    'type': 'LINE_LOST',
                                    'message': f"Line lost for >{LINE_LOST_CREEP_TIME:.1f}s while "
                                               f"FOLLOWING - robot stopped, needs manual recovery",
                                    'pose': {'x': float(stuck_pose_x), 'y': float(stuck_pose_y),
                                             'psi': float(stuck_pose_psi), 'time': time.time()},
                                },
                            })

                    # ---------------- METRONOME: close the servo, then MISSION COMPLETE ----------------
                    if self.state == "METRONOME":
                        cv2.putText(display_frame, "METRONOME (servo open)", (10, 190),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
                        if time.time() >= metronome_close_time:
                            frodo.control.setTrackSpeed(0.0, 0.0)
                            if SERVO_ON_ARRIVAL:
                                self.servo_trigger.rotate_to_home()   # close the servo
                            arrived_node = arriving_node
                            arriving_node = None
                            with self._lock:
                                if self.target_node == arrived_node:
                                    self.target_node = None
                            self.state = "DONE"
                            print(f"\n*** MISSION COMPLETE - arrived at {arrived_node}, metronome done. "
                                  f"Holding until a new go_to_position(). ***")
                            ap_x, ap_y, ap_psi = self.pose_est.get()
                            ap_target = None
                            if arrived_node is not None:
                                _ax, _ay = grid_node_to_world(arrived_node)
                                ap_target = {'x': _ax, 'y': _ay, 'psi': None, 'speed': None, 'tolerance': None}
                            self.frodo.communication.send_event('art_project', {
                                'type': 'position_reached',
                                'data': {
                                    'pose': {'x': float(ap_x), 'y': float(ap_y), 'psi': float(ap_psi),
                                             'time': time.time()},
                                    'target': ap_target,
                                    'node': list(arrived_node) if arrived_node is not None else None,
                                },
                            })

                # ================= SHORT STRAIGHT ADVANCE TO INTERSECTION =================
                # We do NOT look at the image at all - just drive forward at a fixed
                # speed (see the ADVANCE_TIME note above). We leave the actual turning
                # ENTIRELY to the psi PI controller.
                elif self.state == "ADVANCING_TO_TURN":
                    advance_elapsed = now - pre_turn_start
                    frodo.control.setTrackSpeed(base_speed, base_speed)
                    cv2.putText(display_frame, f"ADVANCING {advance_elapsed:.1f}s -> {TURN_TARGET_HEADING}",
                                (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

                    if advance_elapsed >= advance_time_this_turn:
                        print(f"Reached intersection ({advance_elapsed:.2f}s) -> turn starting")
                        # The color switches now: the turn is actually starting, the
                        # robot will look for the new direction's color instead of
                        # the old path's.
                        CURRENT_TARGET_COLOR = "pink" if TURN_TARGET_HEADING in ("EAST", "WEST") else "green"
                        frodo.control.setTrackSpeed(0.0, 0.0)
                        self.state = pre_turn_next_state
                        turn_start = None
                        pre_turn_next_state = None

                # ================= SHORT STRAIGHT ADVANCE OUT OF A TURN =================
                # The turn (above) only pivots the robot - it doesn't drive it forward
                # onto the new lane. Advance straight OPEN-LOOP (without looking at the
                # image, same as ADVANCING_TO_TURN) for a moment so the robot is
                # physically on top of the new-color line before FOLLOWING starts
                # looking for it - otherwise a pivot that isn't perfectly centered on
                # the intersection leaves the new lane just out of frame and FOLLOWING
                # reports LINE LOST even though the color/logic are both correct.
                elif self.state == "ADVANCING_FROM_TURN":
                    exit_elapsed = now - turn_exit_start
                    exit_mask = get_color_mask(frame, CURRENT_TARGET_COLOR)
                    # Wider search than plain FOLLOWING: the new line may sit high in
                    # the frame (further ahead) or look small/angled right after a
                    # not-perfectly-centred pivot.
                    _ef = find_line(exit_mask, roi_top=0.25, min_area=300)
                    if _ef is not None:
                        exit_err, exit_line_seen = _ef[0], True
                    else:
                        exit_err, exit_line_seen = 0.0, False
                    calculate_deviation(exit_mask, display_frame)   # overlay only
                    cv2.putText(display_frame, f"EXITING TURN {exit_elapsed:.1f}s",
                                (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

                    if exit_line_seen and exit_elapsed >= TURN_EXIT_BLIND_CREEP_TIME:
                        # line is in view somewhere - steer onto it (not blind straight)
                        turn_exit_sweep_start = None
                        fwd_e, ang_e = proportional_controller(exit_err, kp, base_speed)
                        vL_e, vR_e = calculate_wheel_speeds(fwd_e, ang_e, track_width)
                        frodo.control.setTrackSpeed(vL_e, vR_e)
                        if abs(exit_err) < TURN_EXIT_CENTER_PX:
                            if turn_exit_centered_since is None:
                                turn_exit_centered_since = now
                            elif now - turn_exit_centered_since > TURN_EXIT_CENTER_HOLD:
                                print(f"  [turn exit] line re-acquired (err={exit_err:+.0f}px) -> FOLLOWING")
                                self.state = "FOLLOWING"
                                turn_exit_start = None
                                turn_exit_centered_since = None
                        else:
                            turn_exit_centered_since = None
                    elif exit_elapsed < TURN_EXIT_BLIND_CREEP_TIME + TURN_EXIT_STRAIGHT_TIME:
                        # no line yet - creep straight, it may still come into frame
                        frodo.control.setTrackSpeed(base_speed, base_speed)
                    elif exit_elapsed < TURN_EXIT_BLIND_CREEP_TIME + TURN_EXIT_STRAIGHT_TIME + 2 * TURN_EXIT_SWEEP_TIME:
                        # still nothing - sweep in place: one way, then back the other way
                        if turn_exit_sweep_start is None:
                            turn_exit_sweep_start = now
                        swept = now - turn_exit_sweep_start
                        sweep_omega = 0.45 if swept < TURN_EXIT_SWEEP_TIME else -0.45
                        vL_e, vR_e = calculate_wheel_speeds(0.0, sweep_omega, track_width)
                        frodo.control.setTrackSpeed(vL_e, vR_e)
                        cv2.putText(display_frame, "TURN EXIT - SEARCHING LINE", (10, 220),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        # give up - let FOLLOWING report LINE LOST to the host
                        print("  [turn exit] line not found by straight+sweep - handing to FOLLOWING (will report lost)")
                        frodo.control.setTrackSpeed(0.0, 0.0)
                        self.state = "FOLLOWING"
                        turn_exit_start = None
                        turn_exit_centered_since = None
                        turn_exit_sweep_start = None

                # (Arrival is handled entirely by the METRONOME branch inside LINE
                # FOLLOWING above: open servo -> drive slowly for SERVO_OPEN_DRIVE_TIME
                # -> close servo -> DONE. There is no separate APPROACHING/PARKING
                # state any more.)

                # ================= TURNING (closed loop, EKF psi + PI) =================
                elif self.state in ("TURNING_LEFT", "TURNING_RIGHT"):
                    if turn_start is None:
                        turn_start = now
                        turn_last_time = now
                        turn_i_acc = 0.0
                        turn_settle_start = None
                        # Target is a RELATIVE +-90deg from wherever psi actually is
                        # right now - NOT the absolute HEADING_TO_PSI value. psi is
                        # odometry-only (the EKF correction only touches x,y - see
                        # pose_estimator.py) and drifts over a multi-leg run, so by
                        # the time we reach an intersection psi can be well off the
                        # "EAST=0/NORTH=90/.." values HEADING_TO_PSI assumes. Using
                        # those absolute values as the target made a correctly-
                        # decided LEFT turn spin the wrong way (drift put the target
                        # >90deg away, so the shortest path was actually CW), and
                        # forcing the sign without fixing the target just turned that
                        # into an almost-360deg turn the "right" way instead. Turning
                        # a relative 90deg from the CURRENT psi is correct regardless
                        # of how much psi has drifted since the last turn.
                        psi_at_turn_start = self.pose_est.get()[2]
                        turn_target_psi = wrap_pi(
                            psi_at_turn_start + (turn_angle if self.state == "TURNING_LEFT" else -turn_angle)
                        )

                    psi_now = self.pose_est.get()[2]
                    e_psi = wrap_pi(turn_target_psi - psi_now)

                    dt_turn = max(now - turn_last_time, 1e-3)
                    turn_last_time = now

                    # Anti-windup: don't accumulate the integral when the output is
                    # already saturated AND the error would push it further into
                    # saturation. Otherwise (in the first few seconds, omega saturated
                    # at +1.0 while the error is still positive) the integral silently
                    # winds up to the ceiling, and once the robot overshoots the
                    # target (the error flips sign) the P term's correction gets
                    # smothered for seconds - that's the "very slow recovery after
                    # overshoot" observed in the field.
                    w_pretest = turn_kp * e_psi + turn_i_acc
                    is_saturated = abs(w_pretest) >= TURN_OMEGA_MAX
                    same_direction = (e_psi * w_pretest) > 0
                    if not (is_saturated and same_direction):
                        turn_i_acc = float(np.clip(turn_i_acc + e_psi * dt_turn * turn_ki,
                                                    -TURN_I_LIMIT, TURN_I_LIMIT))
                    omega_cmd = float(np.clip(turn_kp * e_psi + turn_i_acc,
                                               -TURN_OMEGA_MAX, TURN_OMEGA_MAX))

                    v_left, v_right = calculate_wheel_speeds(0.0, omega_cmd, track_width)
                    frodo.control.setTrackSpeed(v_left, v_right)

                    elapsed = now - turn_start
                    cv2.putText(display_frame,
                                f"TURNING {elapsed:.1f}s -> {TURN_TARGET_HEADING} "
                                f"err={np.degrees(e_psi):+.0f}deg",
                                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                    if (now - last_turn_log) > 0.3:
                        last_turn_log = now
                        print(f"  [TURN {elapsed:.2f}s] target={TURN_TARGET_HEADING} "
                              f"psi={np.degrees(psi_now):+.1f} err={np.degrees(e_psi):+.1f} "
                              f"omega={omega_cmd:+.2f}")

                    if abs(e_psi) < TURN_TOLERANCE:
                        if turn_settle_start is None:
                            turn_settle_start = now
                        elif now - turn_settle_start > TURN_SETTLE_TIME:
                            print(f"Heading {TURN_TARGET_HEADING} reached after {elapsed:.2f}s "
                                  f"(err={np.degrees(e_psi):+.1f}deg)")
                            frodo.control.setTrackSpeed(0.0, 0.0)
                            self.state = "ADVANCING_FROM_TURN"
                            turn_exit_start = now
                            turn_exit_centered_since = None
                            turn_exit_sweep_start = None
                            turn_start = None
                            turn_settle_start = None
                    else:
                        turn_settle_start = None

                    if turn_start is not None and elapsed > TURN_TIMEOUT:
                        print("!!! TURN TIMEOUT - failed to reach the target angle")
                        STOP_REASON = "TURN_TIMEOUT"
                        err_pose_x, err_pose_y, err_pose_psi = self.pose_est.get()
                        self.frodo.communication.send_event('art_project', {
                            'type': 'error',
                            'data': {
                                'type': STOP_REASON,
                                'message': f"{STOP_REASON}: failed to reach the target heading",
                                'pose': {'x': float(err_pose_x), 'y': float(err_pose_y),
                                         'psi': float(err_pose_psi), 'time': time.time()},
                            },
                        })
                        self.state = "STOPPED"
                        turn_start = None

                with self._stream_frame_lock:
                    self.frame_out = display_frame

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        except Exception as e:
            print(f"\n!!! ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                frodo.control.setTrackSpeed(0.0, 0.0)
                print("Motors stopped.")
            except Exception:
                pass
            try:
                self.pose_est.stop()
            except Exception:
                pass
            if self.peer_sync is not None:
                try:
                    self.peer_sync.stop()
                except Exception:
                    pass
            self.servo_trigger.shutdown()


# =====================================================================================
def main():
    frodo = FRODO()
    frodo.init()
    frodo.start()
    frodo.control.setMode(FRODO_ControlMode.EXTERNAL)

    agent = ArtProject(frodo)
    agent.run()


if __name__ == "__main__":
    main()
