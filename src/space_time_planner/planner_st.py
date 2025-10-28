import math
import heapq
from typing import List, Dict, Tuple, Optional

import networkx as nx
from .agent_st import Agent

EPS = 1e-3
DEFAULT_HOLD_SEC = 1.0
DEFAULT_GAP_SEC  = 3.0
MAX_DEPART_PUSH_ITERS = 256

def _weight(G: nx.Graph, u: str, v: str) -> float:
    return float(G[u][v].get("weight", 1.0))

def _next_free_node_excluding(intervals: List[Tuple[float, float, str]], t: float, gap: float, who: Optional[str]) -> float:
    if not intervals:
        return t
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    cur = t
    for (s, e, name) in intervals:
        if name == who:
            continue
        if cur + gap <= s:
            break
        if cur < e + gap:
            cur = e + gap
    return cur

# --------------------------
# Small pretty helper
# --------------------------
def _fmt_path(nodes: List[str]) -> str:
    return " -> ".join(nodes) if nodes else "<empty>"

# ==========================
# Reservation table (with open-ended node occupancy)
# ==========================

class ReservationTable:
    """
    Node/edge reservations in seconds.
    - node_res[(node)] = [(t0, t1, who), ...]  (closed intervals; t1 can be +∞)
    - edge_res[(u,v)]  = [(t0, t1, who), ...]
    - open_nodes[node] = [(t0, who), ...]      (open-ended occupancy: [t0, +∞) until closed)
    """
    def __init__(self, gap_sec: float, hold_sec: float):
        self.node_res: Dict[str, List[Tuple[float, float, str]]] = {}
        self.edge_res: Dict[Tuple[str, str], List[Tuple[float, float, str]]] = {}
        self.open_nodes: Dict[str, List[Tuple[float, str]]] = {}
        self.gap = float(gap_sec)
        self.hold = float(hold_sec)

    # ---- book (closed)
    def book_node(self, node: str, t0: float, t1: float, who: str) -> None:
        lst = self.node_res.setdefault(node, [])
        lst.append((t0, t1, who))
        lst.sort(key=lambda x: (x[0], x[1]))

    def book_node_forever(self, node: str, t0: float, who: str, override: bool = True) -> None:
        """
        Reserve 'node' from t0 to +∞ for 'who'.
        If override=True, drop any previous +∞ reservation for the same who, then insert this one.
        """
        lst = self.node_res.setdefault(node, [])
        if override:
            lst[:] = [(s, e, w) for (s, e, w) in lst if not (w == who and math.isinf(e))]
        lst.append((t0, float("inf"), who))
        lst.sort(key=lambda x: (x[0], x[1]))


    def book_edge(self, edge: Tuple[str, str], t0: float, t1: float, who: str) -> None:
        lst = self.edge_res.setdefault(edge, [])
        lst.append((t0, t1, who))
        lst.sort(key=lambda x: (x[0], x[1]))

    # ---- open/close node occupancy (waiting parked at node)
    def open_node(self, node: str, t0: float, who: str) -> None:
        self.open_nodes.setdefault(node, []).append((t0, who))

    def close_node(self, node: str, who: str, t1: float) -> None:
        lst = self.open_nodes.get(node, [])
        for i in range(len(lst) - 1, -1, -1):
            t0, name = lst[i]
            if name == who:
                lst.pop(i)
                if t1 > t0:
                    self.book_node(node, t0, t1, who)
                break

# ==========================
# Earliest feasible depart
# ==========================

def earliest_depart(G: nx.Graph, res: ReservationTable, u: str, v: str, t: float, w: float, who: str) -> float:
    """
    Edge + reverse-edge + node checks. Higher-level logic also prevents committing
    when arrival node is currently open-occupied by someone else.
    """
    depart = t
    for _ in range(MAX_DEPART_PUSH_ITERS):
        conflict = False

        # departure node 'u' must be free for us (ignore our own closed holds)
        dep2 = _next_free_node_excluding(res.node_res.get(u, []), depart, res.gap, who)
        if dep2 > depart + 1e-12:
            depart = dep2
            conflict = True
        if conflict:
            continue

        # incoming edges into u
        incoming_lists = []
        for x in G.predecessors(u):
            incoming_lists.extend(res.edge_res.get((x, u), []))
        for (s, e, name) in sorted(incoming_lists, key=lambda x: (x[0], x[1])):
            if name == who:
                continue
            if not (depart >= e + res.gap or depart + res.gap <= s):
                depart = e + res.gap
                conflict = True
                break
        if conflict:
            continue

        # reverse edge must be free while we are on (u->v)
        for (s, e, _) in res.edge_res.get((v, u), []):
            s_q, e_q = s - res.gap, e + res.gap
            if not (depart + w <= s_q or e_q <= depart):
                depart = e_q
                conflict = True
                break
        if conflict:
            continue

        # forward edge (u->v)
        for (s, e, _) in res.edge_res.get((u, v), []):
            s_q, e_q = s - res.gap, e + res.gap
            if not (depart + w <= s_q or e_q <= depart):
                depart = e_q
                conflict = True
                break
        if conflict:
            continue

        # arrival node closed reservations (ignore our own reservations/forecasts)
        arr = depart + w
        arr2 = _next_free_node_excluding(res.node_res.get(v, []), arr, res.gap, who)
        if abs(arr2 - arr) < 1e-12:
            return depart
        depart = max(depart, arr2 - w)

    return depart

# ==========================
# Time-aware shortest path (utility)
# ==========================

def time_aware_shortest_path(G: nx.Graph, start: str, goal: str,
                             res: ReservationTable,
                             who: str,
                             t0: float = 0.0,
                             max_time: float = 1e6) -> List[str]:
    frontier: List[Tuple[float, str, List[str]]] = [(t0, start, [])]
    best_time: Dict[str, float] = {start: t0}

    while frontier:
        t, u, path = heapq.heappop(frontier)
        if t > best_time.get(u, float("inf")):
            continue
        path2 = path + [u]
        if u == goal:
            return path2
        for v in G.neighbors(u):
            w = _weight(G, u, v)
            depart = earliest_depart(G, res, u, v, t, w, who)
            arrive = depart + w
            if arrive > max_time:
                continue
            if arrive + 1e-9 < best_time.get(v, float("inf")):
                best_time[v] = arrive
                heapq.heappush(frontier, (arrive, v, path2))

    raise nx.NetworkXNoPath(f"No time-aware path from {start} to {goal}")

# ==========================
# Wait vs Detour (whole-path ETA, with tracing)
# ==========================

def _simulate_prefix_eta(nodes: List[str], t0: float, G: nx.Graph, res: ReservationTable,
                         who: str, edges: Optional[int] = None) -> Tuple[float, float, Optional[Tuple[str, str]]]:
    if not nodes or len(nodes) < 2:
        return t0, 0.0, None
    t = t0
    waited = 0.0
    first_conflict = None
    used = 0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if edges is not None and used >= edges:
            break
        w = _weight(G, u, v)
        depart = earliest_depart(G, res, u, v, t, w, who)
        if depart - t > 1e-9 and first_conflict is None:
            first_conflict = (u, v)
        waited += max(0.0, depart - t)
        t = depart + w
        used += 1
    return t, waited, first_conflict

# ==========================
# Spatial initial plan
# ==========================

def _spatial_shortest(G: nx.Graph, s: str, g: str) -> List[str]:
    return nx.shortest_path(G, source=s, target=g, weight="weight")

def _index_in_path(nodes: List[str], target: str) -> int:
    try:
        return nodes.index(target)
    except ValueError:
        return 10**9

def _last_safe_before(nodes: List[str], target: str) -> str:
    prev = None
    for n in nodes:
        if n == target:
            return prev if prev is not None else nodes[0]
        prev = n
    return nodes[0]

# ==========================
# Planner
# ==========================

class SpaceTimePlanner:
    def __init__(self,
                 positions: Dict[str, Tuple[float, float]],
                 fps: int = 10,
                 gap_sec: float = DEFAULT_GAP_SEC,
                 hold_sec: float = DEFAULT_HOLD_SEC,
                 debug: bool = False):
        self.positions = positions
        self.fps = int(fps)
        self.res = ReservationTable(gap_sec, hold_sec)
        self.debug = bool(debug)

    def _log(self, *a):
        if self.debug:
            print(*a)

    def dump_reservations(self) -> None:
        print("\n[ST] Reservations:")
        for (u, v), it in sorted(self.res.edge_res.items()):
            for (s, e, who) in it:
                print(f"  EDGE {u:>8} -> {v:<8}  [{s:6.2f},{e:6.2f})  by {who}")
        for n, it in sorted(self.res.node_res.items()):
            for (s, e, who) in it:
                e_str = "∞" if not math.isfinite(e) else f"{e:6.2f}"
                print(f"  NODE {n:>8}            [{s:6.2f},{e_str})  by {who}")
        for n, it in sorted(self.res.open_nodes.items()):
            for (s, who) in it:
                print(f"  OPEN {n:>8}            [{s:6.2f}, +inf) by {who}")

    def _forecast_goal_block(self, a: Agent, G: nx.Graph) -> None:
        """
        Startup forecast: reserve a.goal from simple path-weight ETA at t=0 to +∞.
        (Overridden later by dynamic forecasts and the actual arrival.)
        """
        path = getattr(a, "planned_path", None) or []
        if len(path) < 2:
            return
        eta = 0.0
        for u, v in zip(path[:-1], path[1:]):
            eta += _weight(G, u, v)
        self.res.book_node_forever(a.goal, eta, a.name)
        self._log(f"[forecast] {a.name}: {a.goal} reserved from t={eta:.2f} (goal forecast)")

    def _refresh_goal_forecast(self, a: Agent, G: nx.Graph, base_t: float) -> None:
        """
        Dynamic forecast each tick from *current anchor* to goal using current reservations.
        Books goal from predicted ETA to +∞ (overrides previous forecast).
        """
        cur_node, _ = self._current_anchor(a)
        try:
            # Find a time-aware path from 'now' to goal given current reservations
            p = time_aware_shortest_path(G, cur_node, a.goal, self.res, who=a.name, t0=base_t)
            eta, _, _ = _simulate_prefix_eta(p, base_t, G, self.res, a.name, edges=None)
            if math.isfinite(eta):
                self.res.book_node_forever(a.goal, eta, a.name)  # override previous forecast
                self._log(f"[forecast] {a.name}: {a.goal} reserved from t={eta:.2f} (dynamic)")
        except nx.NetworkXNoPath:
            # No feasible route right now → keep previous forecast (do nothing)
            pass


    def initial_plan(self, agents: List[Agent], G: nx.Graph) -> None:
        self.res.node_res.clear()
        self.res.edge_res.clear()
        self.res.open_nodes.clear()

        for a in agents:
            path = _spatial_shortest(G, a.start, a.goal)
            a.planned_path = path[:]
            a.full_route  = path[:]
            a.route = []
            a.fragments = []
            a.waiting = False
            a.finished = False
            a.active = True
            a.arrival_time = 0.0
            a.wait_frames = 0
            a.replans = 0
            a.gates = 0
            a.scheduled_departure_frame = 0
            a.finished_frame = 0
            self._log(f"[plan] {a.name}: {path}")
            # Agent is physically at its start node from t=0 until its first depart
            self.res.open_node(a.start, 0.0, a.name)

        # contested nodes → gate owners + waiters
        claims: Dict[str, List[Tuple[Agent, int]]] = {}
        for a in agents:
            for i, n in enumerate(a.planned_path):
                claims.setdefault(n, []).append((a, i))
        contested = [n for n, lst in claims.items() if len(lst) > 1]

        owners: Dict[str, Agent] = {}
        for n in contested:
            cands = claims[n]
            cands.sort(key=lambda t: (t[1], t[0].priority(), t[0].name))
            owners[n] = cands[0][0]

        for n in contested:
            owner = owners[n]
            for (a, _) in claims[n]:
                if a is owner:
                    continue
                a.waiting = True
                a.blocked_by_node = n
                a.blocker_owner = owner.name
                a.wait_node = _last_safe_before(a.planned_path, n)
                idx = _index_in_path(a.planned_path, a.wait_node)
                a.fragments.append(a.planned_path[:idx + 1])
                self._log(f"[gate] {a.name} waits @ {a.wait_node} (blocked by {owner.name} at {n})")

        # Seed goal-forecast reservations so others avoid future occupied goals from t=0
        for _a in agents:
            self._forecast_goal_block(_a, G)

    def _current_anchor(self, a: Agent) -> Tuple[str, float]:
        if a.route:
            last_arrival = float(a.route[-1][3])
            return a.route[-1][1], last_arrival
        return a.start, 0.0

    def _commit_one_edge(self, a: Agent, G: nx.Graph, prefix_nodes: List[str], now_t: float) -> Tuple[str, str, float, float]:
        if len(prefix_nodes) < 2:
            return "", "", 0.0, 0.0

        u, v = prefix_nodes[0], prefix_nodes[1]
        w = _weight(G, u, v)

        base_depart = max(now_t, float(getattr(a, "arrival_time", 0.0)))
        depart = earliest_depart(G, self.res, u, v, base_depart, w, a.name)
        arrive = depart + w

        # GUARD 1: if earliest_depart says unreachable, do not commit.
        if not math.isfinite(depart) or not math.isfinite(arrive):
            self._log(f"[skip] {a.name}: cannot schedule {u}->{v} (depart/arrive is ∞) — wait or detour")
            return "", "", 0.0, 0.0

        # We were 'open' at node u since our last arrival; close that open interval at 'depart'.
        self.res.close_node(u, a.name, depart)

        # Final arrival-node check using CLOSED reservations — IMPORTANT:
        # ignore *our own* reservations (including our goal-forever forecast), so we don't block ourselves.
        arrive2 = _next_free_node_excluding(self.res.node_res.setdefault(v, []), arrive, self.res.gap, a.name)
        if arrive2 > arrive:
            depart += (arrive2 - arrive)
            arrive = arrive2

        # GUARD 2: arrive2 might have pushed us into ∞ if someone else owns v forever.
        if not math.isfinite(depart) or not math.isfinite(arrive):
            self._log(f"[skip] {a.name}: {u}->{v} becomes ∞ after arrival adjustment — wait or detour")
            # Re-open our waiting at 'u' from the same last arrival time to now (visual only):
            # (No need to re-open here; the agent is logically still at 'u' until a real depart.)
            return "", "", 0.0, 0.0

        # Book short node blips + edge + a brief hold at v
        self.res.book_node(u, depart, depart + EPS, a.name)
        self.res.book_edge((u, v), depart, arrive, a.name)
        self.res.book_node(v, arrive, arrive + EPS, a.name)
        if self.res.hold > 0:
            self.res.book_node(v, arrive, arrive + self.res.hold, a.name)

        # Open-occupy v from arrival onward until our next depart (or forever if v==goal and we finish)
        self.res.open_node(v, arrive, a.name)

        # Update agent timeline
        timed = (u, v, depart, arrive)
        a.route = (a.route or []) + [timed]
        a.fragments.append([timed])
        a.arrival_time = arrive

        # These must only be computed for finite times
        a.scheduled_departure_frame = int(round(depart * self.fps))

        a.waiting = False
        a.replans = int(getattr(a, "replans", 0)) + 1
        a.current_fragment_idx = len(a.fragments) - 1
        a.replanned = True
        a.fragment_end_node = v

        if v == a.goal:
            a.finished = True
            a.active = False
            a.finished_frame = int(round(arrive * self.fps))
            # Confirm the goal-forever block with our *actual* arrival time (overrides forecasts).
            self.res.book_node_forever(v, arrive, a.name)

        return u, v, depart, arrive

        # ---------- Cooperative deadlock handling helpers ----------

    def _clone_reservation(self) -> ReservationTable:
        """Deep-copy the reservation table so we can simulate safely."""
        src = self.res
        dst = ReservationTable(src.gap, src.hold)
        # Deep copy closed reservations
        for k, v in src.node_res.items():
            dst.node_res[k] = list(v)
        for k, v in src.edge_res.items():
            dst.edge_res[k] = list(v)
        # Deep copy open occupancies
        for k, v in src.open_nodes.items():
            dst.open_nodes[k] = list(v)
        return dst
    
    def _permanent_block_set(self, who: str) -> set[str]:
        """Nodes reserved to +∞ by other agents (i.e., their goals or permanent holds)."""
        blocked: set[str] = set()
        for n, items in self.res.node_res.items():
            for (_, e, name) in items:
                if name != who and math.isinf(e):
                    blocked.add(n)
                    break
        return blocked


    def _try_make_detour_nodes(
        self,
        G: nx.Graph,
        current: str,
        goal: str,
        first_u: Optional[str],
        first_v: Optional[str],
        who: str,
        base_t: float,
        max_extra_avoids: int = 8,
    ) -> Optional[List[str]]:
        """
        Iteratively search for a spatial detour by removing:
        - all nodes permanently reserved by others (goals held to +∞), and
        - the first conflicted arrival node (preferred),
        - plus more culprits discovered along attempted paths,
        until we find a path whose *time-aware* ETA is finite, or we give up.
        """
        # Start with all permanent blocks (others' +∞ reservations)
        avoid: set[str] = self._permanent_block_set(who)

        # Also avoid the first conflicted arrival node if it isn't our current/goal
        if first_v and first_v not in (current, goal):
            avoid.add(first_v)

        attempts = 0
        while attempts <= max_extra_avoids:
            attempts += 1

            # Build a pruned graph
            H = G.copy()
            # Never remove our own endpoints
            safe_avoid = {n for n in avoid if n not in (current, goal)}
            H.remove_nodes_from(safe_avoid)

            # If either endpoint got removed (paranoia), bail
            if current not in H or goal not in H:
                return None

            # Pure spatial candidate on the pruned graph
            try:
                path = nx.shortest_path(H, source=current, target=goal, weight="weight")
            except nx.NetworkXNoPath:
                return None  # no spatial path even after pruning

            # Evaluate with current reservations (time-aware)
            eta, _, first_conf = _simulate_prefix_eta(path, base_t, G, self.res, who, edges=None)
            if math.isfinite(eta):
                return path  # success: this detour is time-feasible

            # Still ∞ → identify another culprit to avoid and try again
            culprit: Optional[str] = None

            # Prefer any node on this path that is permanently blocked by *others*
            for n in path[1:]:  # skip current
                if n in avoid:
                    continue
                for (_, e, name) in self.res.node_res.get(n, []):
                    if name != who and math.isinf(e):
                        culprit = n
                        break
                if culprit:
                    break

            # Otherwise fall back to the first conflicted arrival node from the ETA trace
            if culprit is None and first_conf is not None:
                _, v = first_conf
                if v not in (current, goal):
                    culprit = v

            if culprit is None:
                # We couldn't isolate a specific node to avoid further → stop
                return None

            avoid.add(culprit)

        return None  # ran out of tries


    def _simulate_commit_path(self, G: nx.Graph, res_copy: ReservationTable,
                              nodes: List[str], who: str, t0: float, goal_name: str) -> Tuple[float, float, bool]:
        """
        Simulate committing an entire path onto `res_copy`, using earliest_depart and
        booking node/edge/holds as in the real planner. Returns (ETA, waited, success).
        """
        if not nodes or len(nodes) < 2:
            return t0, 0.0, True

        t = t0
        waited = 0.0

        for u, v in zip(nodes[:-1], nodes[1:]):
            w = _weight(G, u, v)
            depart = earliest_depart(G, res_copy, u, v, t, w, who)
            arrive = depart + w
            if not math.isfinite(depart) or not math.isfinite(arrive):
                return float("inf"), float("inf"), False

            # Close any open occupancy up to depart, then book like the real commit
            res_copy.close_node(u, who, depart)
            res_copy.book_node(u, depart, depart + EPS, who)
            res_copy.book_edge((u, v), depart, arrive, who)
            res_copy.book_node(v, arrive, arrive + EPS, who)
            if res_copy.hold > 0:
                res_copy.book_node(v, arrive, arrive + res_copy.hold, who)
            # We don't need open_node for simulation of ETAs (earliest_depart uses closed reservations)

            waited += max(0.0, depart - t)
            t = arrive

        # If we reached our goal in this simulation, mirror the real "goal forever" behavior
        if nodes[-1] == goal_name:
            res_copy.book_node_forever(goal_name, t, who)

        return t, waited, True

    def _blocking_owners(self, node: Optional[str], who: str) -> set[str]:
        """
        Return set of agent names that currently 'block' a node for 'who':
        - open-occupancies by others
        - closed reservations to +∞ by others
        """
        if node is None:
            return set()
        owners = set()
        for (t0, name) in self.res.open_nodes.get(node, []):
            if name != who:
                owners.add(name)
        for (s, e, name) in self.res.node_res.get(node, []):
            if name != who and math.isinf(e):
                owners.add(name)
        return owners

    def _time_aware_blockers(self, G: nx.Graph, u: Optional[str], v: Optional[str],
                         base_t: float, who: str) -> set[str]:
        """
        Owners that actually block v *at our arrival time* if we take (u->v) at base_t.
        - Open occupancy by others: always blocks.
        - +∞ (goal) reservations by others: block only if our arrival >= their forecast start - gap.
        """
        if v is None or u is None:
            return set()

        owners: set[str] = set()

        # 1) Open-occupied: hard block
        for (_, name) in self.res.open_nodes.get(v, []):
            if name != who:
                owners.add(name)

        # 2) +∞ reservations (goal forecasts) by others: time-aware
        inf_res = [(s, name) for (s, e, name) in self.res.node_res.get(v, [])
                if name != who and math.isinf(e)]
        if inf_res:
            w = _weight(G, u, v)
            dep = earliest_depart(G, self.res, u, v, base_t, w, who)
            if math.isfinite(dep):
                arr = dep + w
                for s, name in inf_res:
                    if arr >= s - self.res.gap:
                        owners.add(name)
            else:
                # If we can't even get a finite depart, treat as blocked
                owners.update(name for (_, name) in inf_res)

        return owners

    def _compute_next_hop(self, nodes: List[str]) -> Tuple[Optional[str], Optional[str]]:
        if nodes and len(nodes) >= 2:
            return nodes[0], nodes[1]
        return None, None

    def replan_waiting_agents(self, agents: List[Agent], G: nx.Graph,
                              frame: int = 0, fps: Optional[int] = None,
                              release_delay_frames: int = 0,
                              ) -> bool | int:
        if fps is None:
            fps = self.fps
        now = frame / float(fps) + (release_delay_frames / float(fps))

        ready = [a for a in agents if not getattr(a, "finished", False)]
        # sort in your preferred priority
        ready.sort(key=lambda a: (a.priority(), a.name))

        # ----------------------------
        # Phase 0: snapshot per-agent facts at this tick
        # ----------------------------
        facts = {}  # name -> dict
        for a in ready:
            cur_node, anchor_t = self._current_anchor(a)
            base_t = max(now, anchor_t)

            # Update dynamic goal-forecast so others avoid this goal appropriately
            self._refresh_goal_forecast(a, G, base_t)

            target = a.goal
            if getattr(a, "waiting", False) and getattr(a, "wait_node", None) and cur_node != a.wait_node:
                target = a.wait_node

            try:
                cand_nodes = _spatial_shortest(G, cur_node, target)
            except nx.NetworkXNoPath:
                cand_nodes = [cur_node]  # no move possible

            u0, v0 = self._compute_next_hop(cand_nodes)
            facts[a.name] = {
                "agent": a,
                "cur": cur_node,
                "target": target,
                "base_t": base_t,
                "cand_nodes": cand_nodes,
                "u0": u0,
                "v0": v0,
                "blockers": self._time_aware_blockers(G, u0, v0, base_t, a.name) if v0 else set(),

            }

        # ----------------------------
        # Phase 1: detect simple mutual wait (2-agent cycle) and evaluate cooperative yields
        # ----------------------------
        forced_detour_for: Dict[str, List[str]] = {}

        # Build wait-for edges: A -> B if A's next arrival node is blocked by B (open or ∞)
        wait_edges = {(a_name, b_name)
                      for a_name, f in facts.items()
                      for b_name in f["blockers"]}

        # Look for mutually waiting pairs (A<->B)
        seen_pairs: set[Tuple[str, str]] = set()
        for (a_name, b_name) in list(wait_edges):
            if (b_name, a_name) in wait_edges:
                pair = tuple(sorted((a_name, b_name)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                A = facts[pair[0]]
                B = facts[pair[1]]
                a, b = A["agent"], B["agent"]

                # Prepare detours around the first blocked hop for each
                detour_A = self._try_make_detour_nodes(G, A["cur"], a.goal, A["u0"], A["v0"], a.name, A["base_t"])
                detour_B = self._try_make_detour_nodes(G, B["cur"], b.goal, B["u0"], B["v0"], b.name, B["base_t"])


                # Scenario: A yields (takes detour), B stays on candidate
                makespan_A_yields = float("inf")
                detail_A = ""
                if detour_A is not None:
                    resA = self._clone_reservation()
                    etaA, _, okA = self._simulate_commit_path(G, resA, detour_A, a.name, A["base_t"], a.goal)
                    if okA:
                        etaB, _, okB = self._simulate_commit_path(G, resA, B["cand_nodes"], b.name, B["base_t"], b.goal)
                        if okB:
                            makespan_A_yields = max(etaA, etaB)
                            detail_A = f"A yields → detourETA={etaA:.2f}s, B ETA={etaB:.2f}s, makespan={makespan_A_yields:.2f}s"

                # Scenario: B yields (takes detour), A stays on candidate
                makespan_B_yields = float("inf")
                detail_B = ""
                if detour_B is not None:
                    resB = self._clone_reservation()
                    etaB2, _, okB2 = self._simulate_commit_path(G, resB, detour_B, b.name, B["base_t"], b.goal)
                    if okB2:
                        etaA2, _, okA2 = self._simulate_commit_path(G, resB, A["cand_nodes"], a.name, A["base_t"], a.goal)
                        if okA2:
                            makespan_B_yields = max(etaB2, etaA2)
                            detail_B = f"B yields → detourETA={etaB2:.2f}s, A ETA={etaA2:.2f}s, makespan={makespan_B_yields:.2f}s"

                # Decide who should yield (pick smaller makespan)
                if makespan_A_yields < makespan_B_yields:
                    if detour_A is not None:
                        forced_detour_for[a.name] = detour_A
                        self._log(f"[coop] mutual-wait {a.name}↔{b.name}: choose {a.name} to YIELD.  {detail_A}")
                elif makespan_B_yields < float("inf"):
                    if detour_B is not None:
                        forced_detour_for[b.name] = detour_B
                        self._log(f"[coop] mutual-wait {a.name}↔{b.name}: choose {b.name} to YIELD.  {detail_B}")
                else:
                    # Neither scenario feasible → leave to normal logic (they'll keep holding)
                    self._log(f"[coop] mutual-wait {a.name}↔{b.name}: no feasible detour to break cycle")

        # ----------------------------
        # Phase 2: normal per-agent release, but honor any cooperative forced detours
        # ----------------------------
        release_events: List[Tuple[str, str, str, float, float]] = []

        for a in ready:
            f = facts[a.name]
            cur_node, base_t = f["cur"], f["base_t"]
            cand_nodes = f["cand_nodes"]

            # If a cooperative decision was made for this agent, override its plan
            if a.name in forced_detour_for:
                chosen_nodes = forced_detour_for[a.name]
                # sanity: ensure it starts from where we are
                if chosen_nodes and chosen_nodes[0] != cur_node:
                    # If the detour starts at our current, keep it; else, fallback to candidate
                    self._log(f"[coop] {a.name}: detour does not start at current node; fallback to candidate")
                    chosen_nodes = cand_nodes
                else:
                    self._log(f"[coop] {a.name}: FORCED DETOUR applied → {_fmt_path(chosen_nodes)}")
            else:
                # ----- normal candidate vs detour trace (unchanged from your version) -----
                eta_A, waited_A, first_conf = _simulate_prefix_eta(cand_nodes, base_t, G, self.res, a.name, edges=None)
                if len(cand_nodes) >= 2:
                    u0, v0 = cand_nodes[0], cand_nodes[1]
                    w0 = _weight(G, u0, v0)
                    dep0 = earliest_depart(G, self.res, u0, v0, base_t, w0, a.name)
                    wait0 = max(0.0, dep0 - base_t)
                    occ_note = " [blocked: arrival open-occupied]" if self._blocking_owners(v0, a.name) else ""
                    self._log(
                        f"[eta] {a.name} @ t={base_t:.2f}  candidate: {_fmt_path(cand_nodes)}  "
                        f"ETA={('∞' if not math.isfinite(eta_A) else f'{eta_A:.2f}s')} "
                        f"(Δ={('∞' if not math.isfinite(eta_A) else f'{(eta_A - base_t):.2f}s')} from now, "
                        f"waited≈{('∞' if not math.isfinite(waited_A) else f'{waited_A:.2f}s')}) ;  "
                        f"first hop {u0}->{v0} earliest_depart={dep0:.2f}s wait={wait0:.2f}s{occ_note}"
                    )
                else:
                    self._log(
                        f"[eta] {a.name} @ t={base_t:.2f}  candidate: {_fmt_path(cand_nodes)}  "
                        f"ETA={('∞' if not math.isfinite(eta_A) else f'{eta_A:.2f}s')} "
                        f"(Δ={('∞' if not math.isfinite(eta_A) else f'{(eta_A - base_t):.2f}s')} from now, "
                        f"waited≈{('∞' if not math.isfinite(waited_A) else f'{waited_A:.2f}s')})"
                    )

                detour = None
                eta_B = float("inf")
                if first_conf is not None:
                    u, v = first_conf
                    detour = self._try_make_detour_nodes(G, cur_node, a.goal, u, v, a.name, base_t)
                    if detour is not None:
                        eta_B, _, _ = _simulate_prefix_eta(detour, base_t, G, self.res, a.name, edges=None)
                        self._log(
                            f"[eta] {a.name} @ t={base_t:.2f}    detour : {_fmt_path(detour)}  "
                            f"ETA={('∞' if not math.isfinite(eta_B) else f'{eta_B:.2f}s')} "
                            f"(avoid first-conflict)"
                        )

                # --- STALL GUARD: candidate and detour both unreachable → do nothing this tick
                if not math.isfinite(eta_A) and (detour is None or not math.isfinite(eta_B)):
                    self._log(f"[stall] {a.name}: goal unreachable under current permanent reservations → hold at {cur_node}")
                    continue

                if detour is not None and math.isfinite(eta_B) and (eta_B + 1e-9 < eta_A or not math.isfinite(eta_A)):
                    better = "candidate unreachable" if not math.isfinite(eta_A) else f"{(eta_A - eta_B):.2f}s"
                    self._log(f"[choose] {a.name}: TAKE DETOUR (better by {better})")
                    chosen_nodes = detour
                else:
                    chosen_nodes = cand_nodes
                    if detour is not None:
                        worse = "∞" if not math.isfinite(eta_B) else f"{(eta_B - eta_A):.2f}s"
                        self._log(f"[choose] {a.name}: WAIT on candidate (detour worse by {worse})")
                    else:
                        self._log(f"[choose] {a.name}: WAIT on candidate (no feasible detour)")

            # Extra guard: unreachable first hop (because of ∞ reservations)
            if len(chosen_nodes) >= 2:
                uu, vv = chosen_nodes[0], chosen_nodes[1]
                dep_try = earliest_depart(G, self.res, uu, vv, base_t, _weight(G, uu, vv), a.name)
                if not math.isfinite(dep_try):
                    self._log(f"[hold] {a.name}: {uu}->{vv} unreachable (permanent reservation) → keep waiting/detour")
                    continue

            # HARD STOP (time-aware):
            if len(chosen_nodes) >= 2:
                uu, vv = chosen_nodes[0], chosen_nodes[1]

                # 1) Open-occupied by someone else → always wait
                if any(name != a.name for (_, name) in self.res.open_nodes.get(vv, [])):
                    self._log(f"[hold] {a.name}: next node {vv} is open-occupied by another agent → keep waiting\n")
                    continue

                # 2) +∞ goal reservations by others → only wait if we'd arrive after their forecast start
                inf_starts = [s for (s, e, name) in self.res.node_res.get(vv, [])
                            if name != a.name and math.isinf(e)]
                if inf_starts:
                    w0 = _weight(G, uu, vv)
                    dep_try = earliest_depart(G, self.res, uu, vv, base_t, w0, a.name)
                    if math.isfinite(dep_try):
                        arr = dep_try + w0
                        s_forecast = min(inf_starts)
                        if arr >= s_forecast - self.res.gap:
                            self._log(f"[hold] {a.name}: {vv} reserved from {s_forecast:.2f}s by others; our arrival {arr:.2f}s → wait\n")
                            continue
                        else:
                            self._log(f"[allow] {a.name}: will pass {vv} before forecast ({arr:.2f}s < {s_forecast:.2f}s) → proceed")
                    else:
                        self._log(f"[hold] {a.name}: {uu}->{vv} blocked by +∞ reservation (no finite depart) → wait\n")
                        continue

            # GATE RULE (unchanged)
            gate = getattr(a, "blocked_by_node", None)
            if getattr(a, "waiting", False) and getattr(a, "wait_node", None) == cur_node and gate:
                if len(chosen_nodes) >= 2 and chosen_nodes[1] == gate:
                    owner = getattr(a, "blocker_owner", None)
                    owner_arrivals = []
                    for (x, y), items in self.res.edge_res.items():
                        if y == gate:
                            for (s, e, who) in items:
                                if who == owner:
                                    owner_arrivals.append(e)
                    if not owner_arrivals:
                        self._log(f"[gate] {a.name}: gate {gate} owner={owner} has no booked arrival yet → wait")
                        continue
                    owner_clear_t = max(owner_arrivals)
                    if base_t < owner_clear_t + self.res.gap:
                        self._log(f"[gate] {a.name}: gate {gate} owned by {owner} clears@{owner_clear_t:.2f} → wait until ≥ {owner_clear_t + self.res.gap:.2f}")
                        continue

            # Commit one edge
            u, v, td, ta = self._commit_one_edge(a, G, chosen_nodes, base_t)
            if u and v:
                self._log(f"  [release] {a.name}: {u:>8} -> {v:<8}  [{td:6.2f}, {ta:6.2f})\n")
                release_events.append((a.name, u, v, td, ta))

        return len(release_events)
