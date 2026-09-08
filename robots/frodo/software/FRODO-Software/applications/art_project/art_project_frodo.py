# =====================================================================================
#  FRODO - Line Following + ArUco Grid Navigation
#
#  WiFi command surface matches hoca's host-side protocol (see
#  software/robots/frodo/applications/artproject/application_artproject.py):
#  go_to_position, stop, get_pose, get_status - names/payloads kept 1:1 compatible so
#  ArtProject_Application/ArtProject_FRODO on the host can drive this robot unchanged.
#  trigger_servo is an extra, FRODO-specific command (no host-side equivalent - the
#  metronome servo also triggers automatically off ArUco IDs, see SERVO_TRIGGER_IDS).
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

# --- SERVO TRIGGER (on specific ArUco IDs) ---
# First verify the pin/servo with servo_test.py.
SERVO_PIN = 19                 # BCM GPIO19 - verified with the servo_pin_scan.py sweep
SERVO_ANGLE_HOME = 0
SERVO_ANGLE_TRIGGER = 90
SERVO_TRIGGER_IDS = {999, 998, 997, 996, 995}  # seeing one of these IDs triggers the servo action
SERVO_FORWARD_DURATION = 1.5   # seconds - drive straight this long after the servo turns, then rotate_to_home()
SERVO_RETRIGGER_COOLDOWN = 5.0 # so passing the same marker doesn't retrigger it over and over
# When a marker is first seen, the robot is not yet lined up with it (the camera
# sees it ahead of time) - close this distance open-loop (without looking at the
# image) before triggering the servo. Measure in the field and adjust as needed.
SERVO_TRIGGER_APPROACH_DISTANCE_M = 0.2
# 2026-09-07 field test: the raw ArUco detector occasionally misreads noise as a
# fake marker ID (observed: 656, 470, 944, 246, 190, 382, 549, 394 - none of these
# are real markers anywhere in this app). Unlike CITY_MAP node acceptance (which
# requires ARUCO_MIN_SIZE_PX=160px), the servo trigger had NO size check at all - a
# noise ID that happened to land on 995-999 would have fired the servo with zero
# validation. Observed noise topped out at 50px, real trigger-distance detections
# were 90-180px - this sits with margin in between.
SERVO_TRIGGER_MIN_SIZE_PX = 70

# hoca's ACTUAL target points (2026-09-07): each metronome marker sits ON a path
# LINE - i.e. at the midpoint of an EDGE between two adjacent CITY_MAP nodes, not
# on a node itself (row is a whole grid step, col is X.5) - field-measured, in grid
# units (already divided by GRID_CELL_M, same units as CITY_MAP). See
# METRONOME_TARGET_NODES below (defined after world_to_grid_node()) for the node
# each one snaps to - same rounding go_to_position() would apply if the host sends
# these as real-world (x, y) instead.
METRONOME_MARKER_GRID_POS = {
    995: (2.5, 0.0),
    996: (5.5, 2.0),
    997: (7.5, 3.0),
    998: (5.5, 4.0),
    999: (3.5, 5.0),
}

# --- ARUCO FILTERS ---
# The DECISION (direction/target) is made as soon as a marker is ACCEPTED - but
# it isn't APPLIED right away, the robot first drives straight for ADVANCE_TIME
# (see ADVANCING_TO_TURN/PARKING below). These two solve different problems:
# decisions used to be made way too early, because ARUCO_Y_MIN=0.15 was loose
# enough that a marker got ACCEPTED while still in the middle of the path
# (observed in the field: decided at size~80-100px, even though a marker at an
# intersection could grow to 150-170px) - fixed by adding a hard floor on pixel
# SIZE (ARUCO_MIN_SIZE_PX). The decision now fires at the right time, but the
# robot may not physically be at the exact center of the intersection/marker yet
# - that last step is finished by ADVANCE_TIME (driving straight open-loop,
# WITHOUT looking at the image).
ARUCO_Y_MIN = 0.15         # markers below this height are "too far away"
ARUCO_MIN_SIZE_PX = 160    # markers smaller than this (i.e. far away) are NOT accepted
ARUCO_X_MIN = 0.10         # left edge of the lane window
ARUCO_X_MAX = 0.90         # right edge of the lane window

# Per-robot override for ARUCO_MIN_SIZE_PX. The 160px default was measured on
# frodo4, which has a 60deg lens (robot/definitions.py). frodo1/frodo2 have a
# 120deg lens and frodo3 a 90deg lens, so the SAME marker at the SAME distance
# lands on far fewer pixels for them - with the 160px floor a wide-lens robot
# almost never ACCEPTS a grid marker, so it never gets a position/heading fix and
# navigates blind. Scale the floor by (robot_fov / 60deg): ~80px for the 120deg
# robots, ~107px for the 90deg one. FIRST GUESS - watch the "size=NNNpx -> TOO FAR"
# vs "-> ACCEPTED" console lines and adjust: still mostly TOO FAR at real
# intersections -> lower further; accepting noise/markers from the next lane over
# -> raise. The real fix is a matching narrow lens + re-calibration; this just
# makes the wide lens usable.
ARUCO_MIN_SIZE_PX_OVERRIDES = {
    "frodo1": 80,
    "frodo2": 80,
    "frodo3": 107,
    # 2026-09-08 field logs: frodo4's real grid markers, even centred, only reach
    # ~133-176px (noise tops out ~110px) - the 160 default rejected most of them,
    # so on straight runs it read every node but after a turn, once slightly off
    # the line, it stopped accepting markers entirely and drifted. 140 catches the
    # real ones with margin over noise.
    "frodo4": 140,
}

ARRIVE_DISTANCE_M = 0.06   # "arrived" once the EKF (x,y) is this close (m) to the target marker's world position
APPROACH_TIMEOUT = 6.0     # safety net: stop anyway if we haven't arrived within this time
APPROACH_STALL_GRACE = 0.5 # if the line has been lost this long (robot already stopped), call it "arrived" right away

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

# --- CLOSED-LOOP TURNING (EKF psi feedback) ---
# Instead of a blind timed turn, we turn a relative +-TURN_ANGLE from whatever
# psi actually reads when the turn starts, with a PI controller (see the
# turn-start block below for why this is relative rather than an absolute
# EAST=0/NORTH=90/.. target - psi is odometry-only and drifts over a
# multi-leg run).
# TURN_ANGLE is NOT 90deg on purpose: field-measured actual rotation for a
# commanded 90deg came out ~85deg (RADIUS calibration still undershoots a
# bit), so we command more than a true right angle to compensate. Re-measure
# with a protractor/compass after changing this and adjust again if needed.
TURN_ANGLE = np.radians(105.0)   # commanded relative turn per grid turn
TURN_KP = 1.6               # rad/s per rad of error - bumped up, turn was too gentle and lost the line
TURN_KI = 0.4                # integral gain - overcomes friction/static error
TURN_I_LIMIT = 0.5           # clamp on the integral term's contribution (rad/s)
TURN_OMEGA_MAX = 1.4         # max angular speed allowed while turning (rad/s)
TURN_TOLERANCE = np.radians(4.0)   # tolerance for "reached the target"
TURN_SETTLE_TIME = 0.15      # stay within tolerance this long before calling the turn done (noise rejection)

# Per-robot override for the turn PI gains above. TURN_KP/TURN_KI were tuned once in
# the field and applied globally to every robot, but real motor/wheel/friction
# differences between physical units mean one shared gain doesn't fit all - 2026-09-07
# field test: frodo4 turns corners cleanly on the defaults, frodo1 visibly turns
# weaker/undershoots with the SAME gains. Override per robot ID here instead of
# bumping the global default (that would also change frodo4, which already works).
# These are a first guess (not field-measured) - watch frodo1 turn and re-tune: still
# weak/slow to converge -> raise "kp" further; oscillates/overshoots past the target
# -> back "kp" off and/or raise "ki" instead.
TURN_GAIN_OVERRIDES = {
    "frodo1": {"kp": 2.2, "ki": 0.6},   # was kp=1.6 ki=0.4 (the global default)
}

# After a turn/park decision is made, the robot keeps driving STRAIGHT (WITHOUT
# looking at the image) for this long, open-loop - so it ends up at the exact
# center of the intersection/marker. Now that ARUCO_MIN_SIZE_PX already makes the
# decision fire close to the intersection (fixed the old "turning too early"
# issue), this short advance is safe: since it never looks at the image at all,
# there's no risk of intersection/marker contours steering line-following the
# wrong way (the old design used line-following here and that caused a "two-step"
# wobble).
ADVANCE_TIME = 1.0   # seconds
ADVANCE_TIME_BOUNDARY = 0.7   # shorter pre-turn advance when the cell ahead is off the grid
                              # (a full ADVANCE_TIME would drive past the grid edge) - but not
                              # SO short that the robot pivots before reaching the intersection
                              # and can't reach the new line (2026-09-08: 0.3s left it short at (8,1))

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


# marker_id -> nearest CITY_MAP node, from METRONOME_MARKER_GRID_POS above - same
# round()+clamp world_to_grid_node() would apply if the host sent these as real
# (x, y) meters instead. NOTE: each marker sits on an EDGE (col is X.5), so this
# is one of its TWO adjacent nodes, picked by Python's round-half-to-even (e.g.
# 2.5 -> 2, 3.5 -> 4) - if a marker should instead be approached from the OTHER
# side, override that entry by hand.
METRONOME_TARGET_NODES = {
    marker_id: (
        max(0, min(GRID_COLS - 1, int(round(gx)))),
        max(0, min(GRID_ROWS - 1, int(round(gy)))),
    )
    for marker_id, (gx, gy) in METRONOME_MARKER_GRID_POS.items()
}

# Body markers per robot (robot/definitions.py -> ARUCO_SETTINGS_*). These are
# the VERTICAL markers on the robot bodies; they are detected by frodo.sensors
# (the floor grid has its own separate detector, see aruco_utils.py). Kept in
# the 900s so they never collide with the floor grid (0-53) or the metronome
# servo triggers (995-999).
ROBOT_BODY_MARKERS = {
    "frodo1": {900, 901},
    "frodo2": {902, 903},
    "frodo3": {904, 905},
    "frodo4": {906, 907},
}
# Right-of-way: earlier = higher priority. A higher-priority robot ignores the
# others and drives its own shortest path. A lower-priority robot, when it sees
# a higher-priority robot occupying the cell straight ahead at an intersection,
# re-routes around it (shortest path with that cell removed). The symmetric
# EMERGENCY stop below still applies to every robot regardless of priority.
ROBOT_PRIORITY = ["frodo1", "frodo2", "frodo3", "frodo4"]

# TEMPORARY for field testing without a host/hub connection (see self.target_node
# below) - which metronome marker's node each robot heads for when no go_to_position()
# has been called yet.
# 2026-09-07 field test: frodo1 placed at (2,0), frodo4 placed at (0,0) - NOT
# frodo1/frodo2 (the file's own placeholder default before today). Targets picked
# so neither equals its own start (frodo1 starts ON 995's node - giving it that as
# a target would mean "don't move") and the two paths cross rather than one just
# chasing the other. Change freely for a different pairing/robots.
TEST_TARGET_BY_ROBOT = {
    "frodo1": METRONOME_TARGET_NODES[999],   # (2, 0) -> (4, 5)
    "frodo4": METRONOME_TARGET_NODES[997],   # (0, 0) -> (8, 3)
}

# Another robot seen closer than *_BLOCK_DISTANCE_M and within +-*_AHEAD_BEARING
# of straight-ahead makes the cell ahead "occupied". *_EMERGENCY_DISTANCE_M is a
# hard stop for any robot (prevents actual contact while the other one clears).
OTHER_ROBOT_BLOCK_DISTANCE_M = 0.55
OTHER_ROBOT_EMERGENCY_DISTANCE_M = 0.28
OTHER_ROBOT_AHEAD_BEARING = np.radians(45)


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
        (the servo also triggers automatically off ArUco IDs - see SERVO_TRIGGER_IDS)

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
        # TEMPORARY for field testing without a host/hub connection - remove once
        # go_to_position() is actually being called over WiFi, this bypasses that
        # entirely and starts the robot heading here immediately. Now points at a
        # real target (a metronome marker's node, see TEST_TARGET_BY_ROBOT above)
        # instead of the old arbitrary (5, 3).
        self.target_node = TEST_TARGET_BY_ROBOT.get(frodo.common.id, METRONOME_TARGET_NODES[995])
        self.stopped = False              # manual halt-in-place, set by stop()
        self._manual_servo_request = False

        # --- streaming ---
        self._stream_frame_lock = threading.Lock()
        self.frame_out = None
        self.streamer = VideoStreamer(image_fetcher=self._get_stream_frame, port=5001)

        # --- ArUco detector ---
        self.aruco_detector = create_aruco_detector(ARUCO_DICT_TYPE)

        # --- servo trigger (SERVO_TRIGGER_IDS -> rotate 90 degrees -> forward -> rotate back) ---
        # Fall back to a no-op servo if the hardware isn't wired up / provisioned
        # yet (missing rpi_hardware_pwm or the config.txt PWM overlay) - the grid
        # navigation and ArUco logic still run, the servo just doesn't move.
        try:
            # self.servo = NullServo()
            # self.servo = HardwareServo(pin=SERVO_PIN, min_pulse_ms=0.15, max_pulse_ms=2.5)

            self.servo = HardwareServo(pin=SERVO_PIN)

        except (ImportError, ModuleNotFoundError, FileNotFoundError, OSError) as e:
            frodo.logger.warning(f"HardwareServo unavailable ({e}) - using NullServo (servo will not move)")
            self.servo = NullServo()
        # Auto servo-trigger fires ONLY on THIS robot's own assigned metronome
        # marker (the one whose node is our current target), not on any 995-999
        # marker seen along the way - the mission is "reach YOUR marker, then run
        # the servo". If the target isn't one of the metronome nodes (host sent an
        # arbitrary go_to_position), there's no auto trigger - arrival at the
        # target node still fires the servo (see the PARKING/DONE block).
        self._assigned_metronome_id = next(
            (mid for mid, node in METRONOME_TARGET_NODES.items() if node == self.target_node), None)
        _auto_trigger_ids = {self._assigned_metronome_id} if self._assigned_metronome_id is not None else set()
        frodo.logger.info(f"Assigned metronome marker: {self._assigned_metronome_id} "
                          f"(target node {self.target_node})")
        self.servo_trigger = ArucoServoTrigger(
            self.servo, frodo.control.setTrackSpeed,
            trigger_ids=_auto_trigger_ids,
            angle_home=SERVO_ANGLE_HOME, angle_trigger=SERVO_ANGLE_TRIGGER,
            forward_speed=0.08, forward_duration=SERVO_FORWARD_DURATION,
            approach_distance_m=SERVO_TRIGGER_APPROACH_DISTANCE_M,
            cooldown=SERVO_RETRIGGER_COOLDOWN,
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
        # marker IDs of robots that outrank me -> I route around / yield to these
        self.higher_priority_markers = set()
        for other_id in ROBOT_PRIORITY[:self.my_priority]:
            self.higher_priority_markers |= ROBOT_BODY_MARKERS.get(other_id, set())
        # every other robot's markers -> used by the symmetric emergency stop
        self.other_robot_markers = set()
        for other_id, markers in ROBOT_BODY_MARKERS.items():
            if other_id != my_id:
                self.other_robot_markers |= markers
        frodo.logger.info(
            f"Collision avoidance: id={my_id!r} priority={self.my_priority} "
            f"route-around={sorted(self.higher_priority_markers)}")

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
    def _nearest_robot_ahead(self, marker_ids):
        """(distance_m, bearing_rad) of the closest body marker in `marker_ids`
        currently reported by frodo.sensors, or None. bearing: + = left,
        - = right, 0 = straight ahead. Only markers in front (fwd > 0) count."""
        if not marker_ids:
            return None
        try:
            sample = self.frodo.sensors.getSample()
        except Exception:
            return None
        best = None
        for m in sample.aruco_measurements:
            if m.measured_aruco_id not in marker_ids:
                continue
            fwd, left = float(m.position[0]), float(m.position[1])
            if fwd <= 0.0:
                continue
            dist = float(np.hypot(fwd, left))
            if best is None or dist < best[0]:
                best = (dist, float(np.arctan2(left, fwd)))
        return best

    def _robot_emergency_ahead(self) -> bool:
        """Another robot (any priority) close and straight ahead -> hard stop."""
        hit = self._nearest_robot_ahead(self.other_robot_markers)
        return (hit is not None
                and hit[0] <= OTHER_ROBOT_EMERGENCY_DISTANCE_M
                and abs(hit[1]) <= OTHER_ROBOT_AHEAD_BEARING)

    def _forward_cell_blocked(self) -> bool:
        """A HIGHER-priority robot occupies the cell I'd enter by going straight."""
        if self.my_priority == 0:
            return False
        hit = self._nearest_robot_ahead(self.higher_priority_markers)
        return (hit is not None
                and hit[0] <= OTHER_ROBOT_BLOCK_DISTANCE_M
                and abs(hit[1]) <= OTHER_ROBOT_AHEAD_BEARING)

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
        CURRENT_TARGET_COLOR = "pink"
        kp = 0.008
        base_speed = 0.08           # kept low for the straight-drive test
        track_width = 0.150

        # This robot's turn PI gains - see TURN_GAIN_OVERRIDES above (per-robot,
        # defaults to the shared TURN_KP/TURN_KI if this robot has no override).
        _turn_gain = TURN_GAIN_OVERRIDES.get(frodo.common.id, {})
        turn_kp = _turn_gain.get("kp", TURN_KP)
        turn_ki = _turn_gain.get("ki", TURN_KI)
        if _turn_gain:
            print(f"Turn gains (override for {frodo.common.id!r}): kp={turn_kp} ki={turn_ki}")

        # This robot's ArUco accept threshold - see ARUCO_MIN_SIZE_PX_OVERRIDES
        # above (per-robot, defaults to ARUCO_MIN_SIZE_PX if no override).
        aruco_min_size_px = ARUCO_MIN_SIZE_PX_OVERRIDES.get(frodo.common.id, ARUCO_MIN_SIZE_PX)
        if aruco_min_size_px != ARUCO_MIN_SIZE_PX:
            print(f"ArUco accept threshold (override for {frodo.common.id!r}): {aruco_min_size_px}px "
                  f"(default {ARUCO_MIN_SIZE_PX}px)")

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
        # Continuous heading re-sync (see HEADING CALIBRATION below): ROBOT_HEADING
        # is dead-reckoned (only updated at turns) and psi is odometry-only, so a
        # slipped wheel, an imperfect turn, or the robot being physically picked up
        # and moved silently desyncs it - after which the robot drives the wrong
        # way, still "confirming" its stale heading. While FOLLOWING the robot can
        # ONLY move straight along a line, so any two consecutive node fixes with no
        # turn between them give the TRUE travel direction - use that to correct.
        prev_node_for_heading = None
        turned_since_prev_node = False
        line_lost_since = None     # FOLLOWING only - see LINE_LOST_CREEP_TIME above
        line_lost_reported = False # so the one-shot LINE_LOST error event below doesn't spam every frame

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
        TURN_EXIT_CENTER_PX = 60           # |error| under this = "centred enough" to start FOLLOWING
        TURN_EXIT_CENTER_HOLD = 0.2        # ...for this long
        turn_exit_start = None
        turn_exit_centered_since = None
        turn_exit_sweep_start = None

        # Once we decide "arrived" at the target, the robot keeps driving straight
        # OPEN-LOOP (without looking at the image) for PARKING_TIME before stopping -
        # so it ends up at the exact center of the marker (see the ADVANCE_TIME note above).
        PARKING_TIME = ADVANCE_TIME + 0.2
        stop_timer = 0

        # APPROACHING state: a direction decision was made but not yet APPLIED. Line
        # following keeps running until the EKF (x,y) gets close to the marker's world
        # position (pending_node_xy), then pending_action is applied ONCE (see the note above).
        pending_action = None        # "TARGET" | None
        pending_node_xy = None
        approach_start = None
        last_approach_log = 0.0
        approach_line_lost_since = None
        arriving_node = None         # the grid node we're APPROACHING/PARKING toward
        arrived_node = None          # the grid node we finished a mission at (see DONE)

        # ADVANCING_TO_TURN: a turn decision was made, the robot keeps driving straight
        # for ADVANCE_TIME, then the turn actually starts (see the note above).
        pre_turn_next_state = None   # "TURNING_LEFT" | "TURNING_RIGHT"
        pre_turn_start = None
        advance_time_this_turn = ADVANCE_TIME   # per-turn (shorter at grid boundaries)

        # SERVO_APPROACHING/SERVO_ADVANCING: starts once the servo trigger ID is seen
        # (see the SERVO_* constants above). Same idea as ADVANCING_TO_TURN - open-loop
        # driving, but ArUco scanning/grid decisions KEEP RUNNING during this too (see
        # the widened status check below) - in the field, the earlier design (which
        # blocked the whole action in one go) made the robot miss the next grid marker
        # during its ~4s of blind driving and navigate to the wrong place.
        servo_phase_start = None

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
                    cv2.putText(display_frame, "FAILED: TURN TIMEOUT", (60, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    print(f"\n!!! MISSION FAILED ({STOP_REASON}). Motors locked. !!!")
                    with self._stream_frame_lock:
                        self.frame_out = display_frame
                    break

                # ---------------- EMERGENCY STOP (another robot dead ahead) ----------------
                # Symmetric (ignores priority) - just don't hit each other. Only
                # while driving forward in a corridor; the short in-place TURNING
                # states are left alone so the psi PI controller isn't disturbed.
                if self.state in ("FOLLOWING", "APPROACHING", "ADVANCING_TO_TURN",
                                  "ADVANCING_FROM_TURN", "PARKING",
                                  "SERVO_APPROACHING", "SERVO_ADVANCING") \
                        and self._robot_emergency_ahead():
                    frodo.control.setTrackSpeed(0.0, 0.0)
                    cv2.putText(display_frame, "ROBOT AHEAD - HOLDING", (30, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if time.time() - last_robot_hold_log > 1.0:
                        last_robot_hold_log = time.time()
                        print(f"[{last_robot_hold_log:.1f}] EMERGENCY HOLD - robot within "
                              f"{OTHER_ROBOT_EMERGENCY_DISTANCE_M} m ahead")
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

                # ---------------- SERVO TRIGGER (SERVO_TRIGGER_IDS, automatic) ----------------
                # Independent of grid navigation: these IDs aren't in CITY_MAP so they
                # never enter the decision loop below, which is why it's checked
                # separately and first. Only triggers while driving straight
                # (FOLLOWING) - so it doesn't clash with the robot's position/speed
                # during a turn/park/approach.
                if self.state == "FOLLOWING":
                    detected_ids_now = [
                        detected_id for detected_id, marker_corners in detected_markers
                        if marker_bbox(marker_corners)[-1] >= SERVO_TRIGGER_MIN_SIZE_PX
                    ]
                    servo_matched_id = self.servo_trigger.matching_id(detected_ids_now, now)
                    if servo_matched_id is not None:
                        print(f"[{now:.1f}] Trigger ArUco ID seen: {servo_matched_id} - approaching marker")
                        self.servo_trigger.mark_triggered(now)
                        servo_phase_start = now
                        self.state = "SERVO_APPROACHING"

                    # ---------------- MANUAL SERVO TRIGGER (from the host) ----------------
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

                    # --- filter decision ---
                    if center_y < ARUCO_Y_MIN * h_img or aruco_size < aruco_min_size_px:
                        status = "TOO FAR"
                    elif center_x < ARUCO_X_MIN * w_img or center_x > ARUCO_X_MAX * w_img:
                        status = "WRONG LANE"
                    elif detected_id not in CITY_MAP:
                        status = "NOT ON MAP"
                    elif detected_id == LAST_SEEN_ID:
                        status = "already processed"
                    elif self.state not in ("FOLLOWING", "SERVO_APPROACHING", "SERVO_ADVANCING"):
                        # SERVO_APPROACHING/SERVO_ADVANCING included: don't miss a grid
                        # decision even while driving open-loop for the servo (see the
                        # servo_phase_start note above).
                        status = f"state={self.state}"
                    else:
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
                                    prev_node_for_heading = current_coord
                                    turned_since_prev_node = False
                                    print(f"Heading calibrated from real movement {first_fix_coord} -> "
                                          f"{current_coord}: {ROBOT_HEADING} (color={CURRENT_TARGET_COLOR})")
                        else:
                            # ---------------- CONTINUOUS HEADING RE-SYNC ----------------
                            # Heading already known. If we reached this node on a
                            # STRAIGHT run (no turn since the previous node fix), the
                            # node-to-node delta is the real heading - snap to it if it
                            # disagrees with the dead-reckoned ROBOT_HEADING.
                            if prev_node_for_heading is not None and not turned_since_prev_node:
                                hdx = current_coord[0] - prev_node_for_heading[0]
                                hdy = current_coord[1] - prev_node_for_heading[1]
                                if hdx != 0 or hdy != 0:
                                    if abs(hdx) >= abs(hdy):
                                        observed = "EAST" if hdx > 0 else "WEST"
                                    else:
                                        observed = "NORTH" if hdy > 0 else "SOUTH"
                                    if observed != ROBOT_HEADING:
                                        print(f"  [heading re-sync] {prev_node_for_heading} -> "
                                              f"{current_coord} = {observed}, was {ROBOT_HEADING} - correcting")
                                        ROBOT_HEADING = observed
                                        CURRENT_TARGET_COLOR = ("pink" if observed in ("EAST", "WEST")
                                                                else "green")
                            prev_node_for_heading = current_coord
                            turned_since_prev_node = False

                    # ---------------- DECISION MAKING ----------------
                    # Only ACT on a decision while actually FOLLOWING. Position/heading
                    # tracking above still runs during SERVO_APPROACHING/SERVO_ADVANCING
                    # (so we don't miss it), but enacting a decision here would overwrite
                    # self.state (to APPROACHING or ADVANCING_TO_TURN) mid-servo-sequence,
                    # permanently hijacking it away before it ever reaches rotate_to_home()
                    # - observed in the field: servo triggered and stayed open because a
                    # different grid marker was seen while driving the open-loop
                    # SERVO_ADVANCING leg. Defer any real decision until back in FOLLOWING.
                    if self.state != "FOLLOWING":
                        break

                    # No target set yet -> nothing to decide, just keep following the
                    # line (no turns). Waits for the host to call go_to_position().
                    with self._lock:
                        target_node = self.target_node
                    if target_node is None:
                        break

                    if current_coord == target_node:
                        print(f"*** TARGET NODE SEEN - approaching marker center before parking ***")
                        pending_action = "TARGET"
                        pending_node_xy = MARKER_WORLD_MAP.get(detected_id)
                        approach_start = now
                        arriving_node = target_node
                        self.state = "APPROACHING"
                        break

                    if not heading_known:
                        # Real heading isn't known yet (only one marker seen so far) -
                        # can't safely decide CONTINUE vs TURN. Wait for the second fix.
                        break

                    # ---------------- NEXT STEP (shortest path, route around a blocking robot) ----------------
                    # BFS over the grid gives a minimum-length path; if a
                    # higher-priority robot is sitting in the cell I'd drive into
                    # by going straight, drop that cell so BFS finds a detour.
                    blocked_cells = set()
                    reactive_ahead_cell = None
                    if self._forward_cell_blocked():
                        step = DIRECTIONS.get(ROBOT_HEADING, (0, 0))
                        reactive_ahead_cell = (current_coord[0] + step[0], current_coord[1] + step[1])
                        blocked_cells.add(reactive_ahead_cell)
                        print(f"  [avoidance] higher-priority robot ahead - routing around {reactive_ahead_cell}")

                    # Approach B: Cooperative A* over what every robot last broadcast
                    # (peer_sync.py) - plans a full conflict-free route instead of only
                    # reacting to whoever the camera sees right now. The camera check
                    # above still wins if the two disagree (ground truth beats a
                    # <=0.5s-old broadcast) - see the `!= reactive_ahead_cell` guard.
                    coop_step = self._cooperative_next_step(current_coord, target_node)
                    coop_cell = None
                    if coop_step and coop_step != "WAIT":
                        dx, dy = DIRECTIONS[coop_step]
                        coop_cell = (current_coord[0] + dx, current_coord[1] + dy)

                    if coop_cell is not None and coop_cell != reactive_ahead_cell:
                        desired_heading = coop_step
                        print(f"  [cooperative] plan says {desired_heading}")
                    else:
                        if coop_step == "WAIT":
                            step = DIRECTIONS.get(ROBOT_HEADING, (0, 0))
                            wait_cell = (current_coord[0] + step[0], current_coord[1] + step[1])
                            blocked_cells.add(wait_cell)
                            print(f"  [cooperative] yielding at {current_coord} - routing around {wait_cell}")

                        desired_heading = next_heading(current_coord, target_node, GRID_NODES,
                                                       blocked=blocked_cells)
                        if desired_heading is None and blocked_cells:
                            # No detour exists (rare on an open grid). Fall back to the
                            # unblocked shortest path - the EMERGENCY stop still keeps
                            # the robots from actually touching.
                            print("  [avoidance] no detour - holding to shortest path, emergency-stop will guard")
                            desired_heading = next_heading(current_coord, target_node, GRID_NODES)
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
                            # exist - a full ADVANCE_TIME there drives the robot off
                            # the end of the grid, so the pivot happens away from the
                            # new line and it never re-acquires it (seen in the field
                            # at (8,0)). Use a short advance in that case.
                            _odx, _ody = DIRECTIONS.get(ROBOT_HEADING, (0, 0))
                            _ahead = (current_coord[0] + _odx, current_coord[1] + _ody)
                            advance_time_this_turn = ADVANCE_TIME if _ahead in GRID_NODES else ADVANCE_TIME_BOUNDARY
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
                            turned_since_prev_node = True   # skip re-sync at the next node (this turn is intentional)
                            self.state = "ADVANCING_TO_TURN"

                    break

                # ================= LINE FOLLOWING =================
                if self.state in ("FOLLOWING", "APPROACHING"):
                    if LAST_SEEN_ID is None and self.state == "FOLLOWING":
                        # No marker read yet -> we don't know which color line we're
                        # on (the default "pink" is just a placeholder). Try both
                        # colors and follow whichever is visible, so we can still
                        # reach a marker even if we start on the green path.
                        best_color, best_found = None, None
                        for probe_color in ("pink", "green"):
                            found = find_line(get_color_mask(frame, probe_color))
                            if found is not None and (best_found is None or found[1] > best_found[1]):
                                best_color, best_found = probe_color, found
                        if best_color is not None:
                            CURRENT_TARGET_COLOR = best_color

                    mask = get_color_mask(frame, CURRENT_TARGET_COLOR)          # RAW frame!
                    error, line_detected, _area = calculate_deviation(mask, display_frame)

                    # 2026-09-07 field observation: heading calibration needs a SECOND
                    # real node fix (see HEADING CALIBRATION above) to correct the
                    # initial pink/green guess - if that second fix never comes (line
                    # lost before reaching the next marker), CURRENT_TARGET_COLOR stays
                    # wrong forever and the robot wanders off on stray same-color noise
                    # until it runs out of anything to follow (observed: frodo1 "went
                    # diagonal", LINE LOST far from where it should have stopped).
                    # Recovery: once heading isn't known yet AND the current color just
                    # failed, try the OTHER color before giving up - this only fires on
                    # an actual failure (not every frame), so it can't reintroduce the
                    # per-frame flip-flop the LAST_SEEN_ID gating above was written to
                    # avoid.
                    if not line_detected and not heading_known:
                        other_color = "green" if CURRENT_TARGET_COLOR == "pink" else "pink"
                        other_found = find_line(get_color_mask(frame, other_color))
                        if other_found is not None:
                            print(f"  [color] {CURRENT_TARGET_COLOR} lost, heading not yet "
                                  f"known - switching to {other_color}")
                            CURRENT_TARGET_COLOR = other_color
                            mask = get_color_mask(frame, CURRENT_TARGET_COLOR)
                            error, line_detected, _area = calculate_deviation(mask, display_frame)

                    if line_detected:
                        line_lost_since = None
                        line_lost_reported = False
                        forward_speed, angular_speed = proportional_controller(error, kp, base_speed)
                        v_left, v_right = calculate_wheel_speeds(forward_speed, angular_speed, track_width)
                        frodo.control.setTrackSpeed(v_left, v_right)
                    elif self.state == "FOLLOWING" and \
                            (now - (line_lost_since or now)) < LINE_LOST_CREEP_TIME:
                        if line_lost_since is None:
                            line_lost_since = now
                        frodo.control.setTrackSpeed(base_speed, base_speed)
                        cv2.putText(display_frame, "LINE LOST - CREEPING", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    else:
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

                    if self.state == "APPROACHING":
                        # Only used for the target marker (see the TARGET REACHED
                        # decision) - since it's one-shot, the EKF-drift risk that
                        # exists for intermediate turns doesn't apply here.
                        if pending_node_xy is None:
                            dist_to_node = 0.0     # not in MARKER_WORLD_MAP - apply immediately
                        else:
                            dist_to_node = float(np.hypot(pending_node_xy[0] - pose_x, pending_node_xy[1] - pose_y))

                        approach_elapsed = now - approach_start
                        cv2.putText(display_frame, f"APPROACHING TARGET {dist_to_node:.2f}m",
                                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

                        if (now - last_approach_log) > 0.3:
                            last_approach_log = now
                            print(f"  [APPROACH {approach_elapsed:.2f}s] target_xy={pending_node_xy} "
                                  f"ekf=({pose_x:+.2f},{pose_y:+.2f}) distance={dist_to_node:.3f}m")

                        # Once the line is lost, the robot is already stopped (setTrackSpeed(0,0)
                        # above) - in that case the EKF distance also stops changing, so it may
                        # never drop below ARRIVE_DISTANCE_M and we'd wait uselessly until
                        # APPROACH_TIMEOUT (6s) (observed in the field: "DESTINATION REACHED"
                        # was delayed by 3-5s). If the robot is REALLY not moving (line lost),
                        # call it "arrived" after a short grace period instead - waiting longer buys nothing.
                        if line_detected:
                            approach_line_lost_since = None
                            stalled = False
                        else:
                            if approach_line_lost_since is None:
                                approach_line_lost_since = now
                            stalled = (now - approach_line_lost_since) > APPROACH_STALL_GRACE

                        arrived = dist_to_node <= ARRIVE_DISTANCE_M
                        timed_out = approach_elapsed > APPROACH_TIMEOUT
                        if timed_out and not arrived:
                            print(f"!!! APPROACH TIMEOUT ({APPROACH_TIMEOUT}s) - EKF distance stayed at "
                                  f"{dist_to_node:.3f}m, stopping anyway")
                        elif stalled and not arrived:
                            print(f"Line lost, robot stopped ({dist_to_node:.3f}m) - wait ended early")

                        if arrived or timed_out or stalled:
                            print(f"*** MARKER CENTER REACHED ({dist_to_node:.3f}m). "
                                  f"Parking for {PARKING_TIME}s ***")
                            frodo.control.setTrackSpeed(0.0, 0.0)
                            self.state = "PARKING"
                            stop_timer = time.time() + PARKING_TIME
                            pending_action = None
                            pending_node_xy = None
                            approach_line_lost_since = None

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
                    exit_err, exit_line_seen, _ = calculate_deviation(exit_mask, display_frame)
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

                # ================= SERVO: APPROACHING THE MARKER (automatic) =================
                # Same open-loop idea as ADVANCING_TO_TURN (driving WITHOUT looking at
                # the image, just a fixed forward speed) - the difference: ArUco
                # scanning/grid decisions KEEP RUNNING during this too (see the widened
                # status check above), the servo itself hasn't moved yet.
                elif self.state == "SERVO_APPROACHING":
                    servo_elapsed = now - servo_phase_start
                    frodo.control.setTrackSpeed(self.servo_trigger.forward_speed, self.servo_trigger.forward_speed)
                    cv2.putText(display_frame, f"SERVO APPROACH {servo_elapsed:.1f}s",
                                (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

                    if servo_elapsed >= self.servo_trigger.approach_time:
                        print(f"Reached the metronome ({servo_elapsed:.2f}s) -> triggering servo")
                        frodo.control.setTrackSpeed(0.0, 0.0)
                        time.sleep(self.servo_trigger.pre_stop_delay)   # robot is STOPPED - covers no distance
                        self.servo_trigger.rotate_to_trigger()          # blocking, short (~settle_time), robot stopped
                        servo_phase_start = now
                        self.state = "SERVO_ADVANCING"

                # ================= SERVO: SHORT FORWARD AFTER TRIGGERING (automatic) =================
                elif self.state == "SERVO_ADVANCING":
                    servo_elapsed = now - servo_phase_start
                    frodo.control.setTrackSpeed(self.servo_trigger.forward_speed, self.servo_trigger.forward_speed)
                    cv2.putText(display_frame, f"SERVO FORWARD {servo_elapsed:.1f}s",
                                (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

                    if servo_elapsed >= self.servo_trigger.forward_duration:
                        frodo.control.setTrackSpeed(0.0, 0.0)
                        self.servo_trigger.rotate_to_home()             # blocking, short (~settle_time), robot stopped
                        self.frodo.communication.send_event('art_project_servo_triggered', {'node': self.current_node})
                        # trigger_ids only holds our OWN assigned metronome marker, so
                        # getting here means we reached it -> mission complete.
                        arrived_node = self.current_node
                        with self._lock:
                            if self.target_node == arrived_node:
                                self.target_node = None
                        print(">>> SERVO ACTION COMPLETE - MISSION COMPLETE, holding until a new go_to_position().")
                        self.state = "DONE"

                # ================= SHORT STRAIGHT ADVANCE AT THE TARGET =================
                # Same idea as ADVANCING_TO_TURN: drive forward at a fixed speed
                # without looking at the image for PARKING_TIME, then stop - so we end
                # up at the exact center of the marker (not by line following, since
                # the line is usually lost right on top of the marker anyway).
                elif self.state == "PARKING":
                    frodo.control.setTrackSpeed(base_speed, base_speed)
                    cv2.putText(display_frame, "PARKING...", (60, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                    if time.time() > stop_timer:
                        frodo.control.setTrackSpeed(0.0, 0.0)
                        arrived_node = arriving_node
                        arriving_node = None

                        # --- trigger the metronome servo on arrival at the assigned target ---
                        # This is the whole point of the mission ("go to your marker,
                        # then run the servo"). Robot is stopped, so the cycle covers
                        # no distance and can't miss anything.
                        print("\n*** ARRIVED at target - triggering servo ***")
                        self.servo_trigger.rotate_to_trigger()
                        self.servo_trigger.rotate_to_home()
                        self.frodo.communication.send_event('art_project_servo_triggered',
                                                            {'node': list(arrived_node) if arrived_node else None})

                        with self._lock:
                            # Only clear the target if the host hasn't already set a new one.
                            if self.target_node == arrived_node:
                                self.target_node = None
                        self.state = "DONE"
                        print("*** MISSION COMPLETE - holding until a new go_to_position(). ***")
                        arrived_pose_x, arrived_pose_y, arrived_pose_psi = self.pose_est.get()
                        arrived_target = None
                        if arrived_node is not None:
                            ax, ay = grid_node_to_world(arrived_node)
                            arrived_target = {'x': ax, 'y': ay, 'psi': None, 'speed': None, 'tolerance': None}
                        self.frodo.communication.send_event('art_project', {
                            'type': 'position_reached',
                            'data': {
                                'pose': {'x': float(arrived_pose_x), 'y': float(arrived_pose_y),
                                         'psi': float(arrived_pose_psi), 'time': time.time()},
                                'target': arrived_target,
                                'node': list(arrived_node) if arrived_node is not None else None,  # FRODO-specific extra
                            },
                        })

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
                            psi_at_turn_start + (TURN_ANGLE if self.state == "TURNING_LEFT" else -TURN_ANGLE)
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
