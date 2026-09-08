# =====================================================================================
#  Weighted-graph pathfinding (Dijkstra, A*) + a cooperative multi-agent planner
#  for the art-project floor grid.
#
#  grid_nav.py's shortest_path() is a plain BFS - correct today because every
#  grid edge costs the same (BFS == Dijkstra whenever all edge weights are
#  equal), but it can't take a per-edge cost or a heuristic. This module adds
#  both, plus the actual "multi-robot, both start AND goal known up front"
#  planner hoca described:
#
#    dijkstra(start, goal, nodes, cost_fn=None, blocked=None)
#      Same neighbors()/DIRECTIONS as grid_nav, Dijkstra's algorithm with an
#      arbitrary cost_fn(a, b) -> float (default: 1 per edge - with that
#      default this returns exactly what grid_nav.shortest_path() does).
#
#    astar(start, goal, nodes, cost_fn=None, heuristic=None, blocked=None)
#      Same, guided by `heuristic` (default: Manhattan distance, admissible
#      on this 4-connected unit grid) so it explores far fewer nodes than
#      Dijkstra/BFS - matters once the grid or the cost function grows.
#
#    plan_cooperative_paths(agents, nodes, priority_order=None)
#      hoca's scenario: agents = {agent_id: (start, goal)}, ALL known up
#      front (unlike Approach A in art_project_frodo.py, which only reacts to
#      whatever the camera sees one cell ahead, step by step, as it drives).
#      This is "Cooperative A*" (Silver, 2005): plan one agent at a time, in
#      priority order (same idea as ROBOT_PRIORITY), over a SPACE-TIME graph
#      - each state is (node, t), not just node - so an agent can also choose
#      to WAIT a timestep in place. Every already-planned higher-priority
#      agent reserves:
#        - the (node, t) cells it occupies, so a later agent can't be in the
#          same cell at the same time,
#        - both directions of the edge it crosses between t and t+1, so a
#          later agent can't swap places with it head-on,
#        - its goal cell for the rest of the horizon after it arrives (it
#          physically sits there - a later agent can't plan through it).
#      Returns {agent_id: [(node, t), ...] or None} - None means that agent
#      found no conflict-free path within the search horizon.
#
#  Nothing here touches hardware/camera/WiFi, so - like grid_nav.py - it can
#  be unit-tested/stress-tested on any machine:
#      python -m applications.art_project.path_planner
#      python -m applications.art_project.path_planner 500   (stress-test)
# =====================================================================================
import heapq
import itertools

from grid_nav import grid_nodes, neighbors, direction_between, DIRECTIONS  # noqa: F401 (DIRECTIONS re-exported)


# === SINGLE-AGENT: DIJKSTRA / A* ======================================================
def _unit_cost(_a, _b):
    return 1.0


def _reconstruct(came_from, end):
    path = [end]
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def dijkstra(start, goal, nodes, cost_fn=None, blocked=None):
    """Weighted shortest path start->goal over `nodes` (a set of (x, y)).
    cost_fn(a, b) -> float, defaults to 1 per edge (then this is identical to
    grid_nav.shortest_path()). `blocked`: nodes that may not be entered (start
    is always allowed, goal is not force-allowed). Returns [start, ..., goal]
    or None if unreachable."""
    cost_fn = cost_fn or _unit_cost
    blocked = set(blocked or ())
    blocked.discard(start)

    if start == goal:
        return [start]

    dist = {start: 0.0}
    came_from = {start: None}
    counter = itertools.count()
    heap = [(0.0, next(counter), start)]
    visited = set()

    while heap:
        d, _, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return _reconstruct(came_from, goal)

        for nxt in neighbors(node, nodes):
            if nxt in blocked or nxt in visited:
                continue
            nd = d + cost_fn(node, nxt)
            if nxt not in dist or nd < dist[nxt]:
                dist[nxt] = nd
                came_from[nxt] = node
                heapq.heappush(heap, (nd, next(counter), nxt))

    return None


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, nodes, cost_fn=None, heuristic=None, blocked=None):
    """Like dijkstra(), guided by `heuristic(node, goal) -> float` (default:
    Manhattan distance - admissible here as long as cost_fn never charges less
    than 1 per edge, true for the default uniform cost). Visits fewer nodes
    than dijkstra()/shortest_path() for the same result on a larger grid."""
    cost_fn = cost_fn or _unit_cost
    heuristic = heuristic or manhattan
    blocked = set(blocked or ())
    blocked.discard(start)

    if start == goal:
        return [start]

    g_score = {start: 0.0}
    came_from = {start: None}
    counter = itertools.count()
    heap = [(heuristic(start, goal), next(counter), start)]
    closed = set()

    while heap:
        _, _, node = heapq.heappop(heap)
        if node in closed:
            continue
        if node == goal:
            return _reconstruct(came_from, goal)
        closed.add(node)

        for nxt in neighbors(node, nodes):
            if nxt in blocked or nxt in closed:
                continue
            tentative_g = g_score[node] + cost_fn(node, nxt)
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                came_from[nxt] = node
                heapq.heappush(heap, (tentative_g + heuristic(nxt, goal), next(counter), nxt))

    return None


# === MULTI-AGENT: COOPERATIVE A* (space-time, prioritized) ===========================
def _space_time_astar(start, goal, nodes, reserved_cells, reserved_edges, max_time, heuristic):
    """A* over states (node, t). Actions: move to a 4-connected neighbor, or
    WAIT (stay on `node`, t -> t+1) - both cost 1. Rejects a move/wait that
    lands on a reserved (node, t), or a move whose edge is reserved (blocks
    head-on swaps). Returns [(node, 0), (node, 1), ...] or None if no
    conflict-free path is found within `max_time` steps."""
    start_state = (start, 0)
    g_score = {start_state: 0}
    came_from = {start_state: None}
    counter = itertools.count()
    heap = [(heuristic(start, goal), next(counter), start_state)]
    closed = set()

    while heap:
        _, _, state = heapq.heappop(heap)
        if state in closed:
            continue
        node, t = state
        if node == goal:
            path = [state]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        closed.add(state)
        if t >= max_time:
            continue

        candidates = [node] + list(neighbors(node, nodes))  # wait, or move
        for nxt in candidates:
            nxt_t = t + 1
            nxt_state = (nxt, nxt_t)
            if nxt_state in closed or (nxt, nxt_t) in reserved_cells:
                continue
            if nxt != node and (node, nxt, t) in reserved_edges:
                continue  # someone else crosses this edge the other way during [t, t+1]
            tentative_g = g_score[state] + 1
            if nxt_state not in g_score or tentative_g < g_score[nxt_state]:
                g_score[nxt_state] = tentative_g
                came_from[nxt_state] = state
                heapq.heappush(heap, (tentative_g + heuristic(nxt, goal), next(counter), nxt_state))

    return None


def plan_cooperative_paths(agents: dict, nodes, priority_order=None, max_horizon=None):
    """agents: {agent_id: (start, goal)}, both known up front for every agent.
    priority_order: iterable of agent_id, highest priority first - defaults to
    `agents`' insertion order (matches ROBOT_PRIORITY's convention: earlier =
    higher priority, planned first, never re-routed around a lower-priority
    agent). max_horizon: search cutoff in timesteps, defaults to a generous
    2 * len(nodes) + 5 (>= worst-case single-agent path length with room to
    wait out a conflict).

    Returns {agent_id: [(node, t), (node, t+1), ...] or None}. `None` for an
    agent means no conflict-free path existed within max_horizon (deadlocked
    by higher-priority agents/goals) - the caller should fall back to
    reactive avoidance (Approach A) or replan with a larger horizon."""
    order = list(priority_order) if priority_order is not None else list(agents.keys())
    max_horizon = max_horizon if max_horizon is not None else (2 * len(nodes) + 5)

    reserved_cells = set()   # {(node, t)}
    reserved_edges = set()   # {(from_node, to_node, t)} - crossed during [t, t+1]
    results = {}

    for agent_id in order:
        start, goal = agents[agent_id]
        path = _space_time_astar(start, goal, nodes, reserved_cells, reserved_edges, max_horizon, manhattan)
        results[agent_id] = path
        if path is None:
            continue

        for i, (node, t) in enumerate(path):
            reserved_cells.add((node, t))
            if i > 0:
                prev_node, prev_t = path[i - 1]
                reserved_edges.add((prev_node, node, prev_t))
                reserved_edges.add((node, prev_node, prev_t))  # block the swap too

        goal_node, arrival_t = path[-1]
        for t in range(arrival_t, max_horizon + 1):
            reserved_cells.add((goal_node, t))  # it physically sits there after arriving

    return results


def path_to_directions(path_with_time):
    """[(node, t), ...] -> ["EAST", "WAIT", "NORTH", ...], one entry per step
    (len(path) - 1 entries). Convenience for driving the FSM in
    art_project_frodo.py off a precomputed cooperative plan."""
    steps = []
    for (a, _), (b, _) in zip(path_with_time, path_with_time[1:]):
        steps.append("WAIT" if a == b else direction_between(a, b))
    return steps


# =====================================================================================
if __name__ == "__main__":
    import sys
    import random

    COLS, ROWS = 9, 6
    NODES = grid_nodes(COLS, ROWS)

    print("dijkstra() / astar() vs grid_nav.shortest_path() (uniform cost -> identical path length):")
    for start, goal in [((0, 0), (8, 5)), ((3, 2), (3, 2)), ((8, 5), (0, 0))]:
        pd = dijkstra(start, goal, NODES)
        pa = astar(start, goal, NODES)
        print(f"  {start} -> {goal}: dijkstra len={len(pd) if pd else None} "
              f"astar len={len(pa) if pa else None}")

    print("\ncooperative plan - two robots whose shortest paths cross head-on:")
    agents = {"frodo1": ((0, 0), (8, 0)), "frodo2": ((8, 0), (0, 0))}
    plans = plan_cooperative_paths(agents, NODES)
    for agent_id, path in plans.items():
        print(f"  {agent_id}: {path}")
        if path:
            print(f"    directions: {path_to_directions(path)}")

    # A cell/timestep must never be shared, and no edge may be crossed by two
    # agents in opposite directions during the same interval.
    occupied = {}
    ok = True
    for agent_id, path in plans.items():
        if path is None:
            continue
        for node, t in path:
            if (node, t) in occupied and occupied[(node, t)] != agent_id:
                print(f"  !!! COLLISION at {node}, t={t}: {agent_id} vs {occupied[(node, t)]}")
                ok = False
            occupied[(node, t)] = agent_id
    print("  no collisions" if ok else "  COLLISIONS FOUND")

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        print(f"\nstress-testing {n} random 2-robot scenarios...")
        nodes_list = list(NODES)
        failures = 0
        no_path = 0
        for i in range(n):
            s1, g1, s2, g2 = random.sample(nodes_list, 4)
            plans = plan_cooperative_paths({"a": (s1, g1), "b": (s2, g2)}, NODES)
            if plans["a"] is None or plans["b"] is None:
                no_path += 1
                continue
            occ = {}
            for aid, path in plans.items():
                for j, (node, t) in enumerate(path):
                    if (node, t) in occ:
                        print(f"  !!! scenario {i}: collision at {node}, t={t}")
                        failures += 1
                        break
                    occ[(node, t)] = aid
                    if j > 0:
                        pnode, pt = path[j - 1]
                        # opposite agent must not have crossed (node,pnode) during pt
        print(f"  done: {failures} collisions, {no_path} unreachable-within-horizon out of {n}")
