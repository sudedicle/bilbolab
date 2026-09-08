# =====================================================================================
#  Robot-to-robot target sharing over a dedicated UDP broadcast - independent of the
#  host WiFi link (core/communication/wifi/udp/udp_socket.py is the same primitive the
#  rest of the codebase already uses for streaming, just on its own port/channel here).
#
#  hoca's Cooperative-A* multi-robot planning (path_planner.plan_cooperative_paths)
#  needs every robot's CURRENT node AND TARGET node up front - but the host only tells
#  each robot its OWN target (go_to_position(x, y) per robot, called separately per
#  robot - see art_project_frodo.py). This fills that gap WITHOUT touching the host
#  protocol at all: each robot periodically broadcasts
#  {robot_id, current_node, target_node, time} on the LAN; every robot listens for
#  every other robot's broadcasts and keeps the latest one per robot_id.
#
#  A peer is usable only within PEER_TIMEOUT_S of its last broadcast - if a robot goes
#  offline / out of WiFi range / hasn't been given a target yet, it silently drops out
#  and cooperative planning proceeds with whichever peers are still fresh. If NONE are
#  fresh (e.g. field-testing with a single robot, or the peer link is down), the
#  caller (art_project_frodo.py::_cooperative_next_step) falls back to the plain
#  camera-reactive Approach A - this module never blocks or is required for a robot to
#  drive.
# =====================================================================================
import json
import threading
import time

from core.communication.wifi.udp.udp_socket import UDP_Socket

PEER_SYNC_PORT = 37030     # separate from UDP_PORT_ADDRESS_STREAM (37020, host link)
BROADCAST_PERIOD_S = 0.5
PEER_TIMEOUT_S = 2.0       # a peer not heard from this long is dropped from planning


class PeerSync:
    """Broadcasts (current_node, target_node) for `robot_id` and collects the same
    from every other robot on the LAN.

    `state_fn`: callable -> (current_node, target_node), each a (col, row) tuple or
    None. Polled once per BROADCAST_PERIOD_S and re-broadcast (broadcast happens even
    when both are None, so peers can tell this robot is alive but idle)."""

    def __init__(self, robot_id: str, state_fn, port: int = PEER_SYNC_PORT, address: str = None):
        self.robot_id = robot_id
        self.state_fn = state_fn
        self._peers = {}   # robot_id -> {"current": node|None, "target": node|None, "time": float}
        self._lock = threading.Lock()
        self._exit = False

        # `address` is only used by UDP_Socket for the (disabled-by-default)
        # broadcast-echo filter, not for binding (it binds to all interfaces) - pass
        # the robot's WiFi IP (art_project_frodo.py uses getInterfaceIP("wlan0"), same
        # as the video stream) in production; any placeholder works off-robot.
        # `address=None` falls back to core.utils.network.getLocalIP_RPi(), which only
        # resolves on an actual Raspberry Pi.
        self._socket = UDP_Socket(address=address, port=port)
        self._socket.callbacks.rx.register(self._on_rx)

        self._thread = threading.Thread(target=self._broadcast_loop, daemon=True)

    # === METHODS =======================================================================
    def start(self):
        self._socket.start()
        self._thread.start()

    def stop(self):
        self._exit = True
        self._socket.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1)

    def get_fresh_peers(self, max_age: float = PEER_TIMEOUT_S) -> dict:
        """{robot_id: (current_node, target_node)} for every peer heard from within
        `max_age` seconds that also has BOTH a current_node and a target_node (i.e. is
        actually navigating, not idle/just booted/never given a target)."""
        now = time.time()
        out = {}
        with self._lock:
            for robot_id, info in self._peers.items():
                if now - info["time"] > max_age:
                    continue
                if info["current"] is None or info["target"] is None:
                    continue
                out[robot_id] = (info["current"], info["target"])
        return out

    # === PRIVATE ========================================================================
    def _broadcast_loop(self):
        while not self._exit:
            current, target = self.state_fn()
            msg = {
                "robot_id": self.robot_id,
                "current_node": list(current) if current is not None else None,
                "target_node": list(target) if target is not None else None,
                "time": time.time(),
            }
            try:
                self._socket.send(json.dumps(msg))
            except OSError:
                pass  # transient network hiccup - next tick retries
            time.sleep(BROADCAST_PERIOD_S)

    def _on_rx(self, data, address, port):
        try:
            payload = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
            msg = json.loads(payload)
            robot_id = msg["robot_id"]
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            return
        if robot_id == self.robot_id:
            return  # broadcast echo of our own message
        with self._lock:
            self._peers[robot_id] = {
                "current": tuple(msg["current_node"]) if msg.get("current_node") else None,
                "target": tuple(msg["target_node"]) if msg.get("target_node") else None,
                "time": msg.get("time", time.time()),
            }
