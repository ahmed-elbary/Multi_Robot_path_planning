# fragment_planner.py
import math
from typing import List, Dict, Tuple, Optional
import networkx as nx
from .agent import Agent


# =============================== small helpers ===============================

def _fmt_path(p: Optional[List[str]], limit: int = 18) -> str:
    """Pretty-print a node list with optional truncation."""
    if not p:
        return "<no path>"
    if limit is None or len(p) <= limit:
        return " -> ".join(p)
    return " -> ".join(p[:limit]) + " -> ..."


def _node_claims(agents: List[Agent]) -> Dict[str, List[Agent]]:
    """
    Build a map: node -> [agents that pass through that node in full_route].
    (We ignore the start node because it's initially occupied by the agent.)
    """
    claims: Dict[str, List[Agent]] = {}
    for a in agents:
        if getattr(a, "full_route", None):
            for n in a.full_route[1:]:
                claims.setdefault(n, []).append(a)
    return claims


def _compute_block_constraints(agents: List[Agent], graph: nx.Graph) -> Dict[str, List[Tuple[str, str]]]:
    """
    For each node with multiple claimers, pick an owner with 'nearest-wins'
    (tie-break by agent.priority()). Return:
      blocked_agent_name -> [(blocked_node, owner_name), ...]
    """
    constraints: Dict[str, List[Tuple[str, str]]] = {}
    claims = _node_claims(agents)
    for node, claimers in claims.items():
        if len(claimers) < 2:
            continue
        winner = agent_has_priority(claimers, node, graph)
        if not winner:
            continue
        for a in claimers:
            if a is winner:
                continue
            constraints.setdefault(a.name, []).append((node, winner.name))
    return constraints


def _earliest_block_on_path(path: List[str], blocks: List[Tuple[str, str]]) -> Optional[Tuple[int, str, str]]:
    """
    Given a node path and a set of (node, owner) blocks, return the earliest
    blocked node index in the path (>=1) and the (node, owner).
    """
    if not blocks:
        return None
    node_to_owner = {n: o for (n, o) in blocks}
    for idx, n in enumerate(path[1:], start=1):
        if n in node_to_owner:
            return (idx, n, node_to_owner[n])
    return None


def _distance_to_node_in_route(agent: Agent, node: str, graph: nx.Graph) -> float:
    """Accumulate edge weights along agent.full_route up to 'node' (∞ if not present)."""
    if not agent.full_route or node not in agent.full_route:
        return math.inf
    idx = agent.full_route.index(node)
    dist = 0.0
    for i in range(idx):
        u, v = agent.full_route[i], agent.full_route[i + 1]
        dist += float(graph[u][v].get("weight", 1.0))
    return dist


def agent_has_priority(agents: List[Agent], node: str, graph: nx.Graph) -> Optional[Agent]:
    """
    Pick the agent with the smallest distance-to-node; tie-break by agent.priority().
    """
    claimers = [a for a in agents if a.full_route and node in a.full_route]
    if not claimers:
        return None
    scored = [(_distance_to_node_in_route(a, node, graph), a.priority(), a) for a in claimers]
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


# =============================== initial planning ==============================

def find_routes(agents: List[Agent], graph: nx.Graph):
    """
    Space-aware planner: plan each agent on a filtered copy of the map.
    Nodes occupied by already-planned agents are removed (except my own start/goal).
    """
    all_starts = {a.start for a in agents}
    for i, agent in enumerate(agents):
        print(f"\n[✓] Planning spatial route for {agent.name} from {agent.start} to {agent.goal}")
        occupied = all_starts - {agent.start}
        # for prev in agents[:i]:
        #     occupied.update(prev.full_route)

        filtered = graph.copy()
        for node in occupied:
            if node not in (agent.start, agent.goal) and node in filtered:
                filtered.remove_node(node)

        print(f"[Debug] {agent.name} sees filtered map with {len(filtered.nodes)} nodes, {len(filtered.edges)} edges.")
        print(f"[✓] Filtered map for {agent.name} — Occupied nodes removed: {sorted(list(occupied))}")

        try:
            path = nx.shortest_path(filtered, agent.start, agent.goal)
            agent.route = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
            agent.full_route = path
            agent.fragments = [agent.route.copy()]
            agent.waiting = False
            agent.wait_node = None
            agent.blocked_by_node = None
            agent.blocker_owner = None
            agent.resume_ready = False
            print(f"[{agent.name}] Planned path: {path}")
        except nx.NetworkXNoPath:
            agent.route = []
            agent.full_route = [agent.start]
            agent.fragments = []
            agent.waiting = True
            agent.wait_node = agent.start
            print(f"[{agent.name}] No spatial path due to blocked nodes, will wait at {agent.start}")


def find_critical_points(agents: List[Agent]) -> Dict[str, List[str]]:
    """
    Return all critical nodes: nodes shared by >1 agent in their current edge-route.
    """
    node_claims: Dict[str, List[str]] = {}
    for agent in agents:
        for _, v in agent.route:
            node_claims.setdefault(v, []).append(agent.name)
    return {n: names for n, names in node_claims.items() if len(names) > 1}


# =============================== initial safe split =============================

def split_critical_paths(graph: nx.Graph, agents: List[Agent], dangerous_points: List[str]):
    """
    Compute initial safe fragments and mark waiters.

    - For each agent, find the earliest blocked node along its full_route using
      nearest-wins ownership (_compute_block_constraints + _earliest_block_on_path).
    - Keep only the safe prefix up to the predecessor of that blocked node.
    - If no block applies, keep the full first fragment produced by find_routes.
    - Print a one-line summary showing the fragment end (safepoint).

    Note: 'dangerous_points' is currently unused; kept for interface parity.
    """
    all_constraints = _compute_block_constraints(agents, graph)

    for agent in agents:
        # No usable route yet
        if not getattr(agent, "full_route", None) or len(agent.full_route) < 2:
            if getattr(agent, "full_route", None):
                agent.fragment_end_node = agent.full_route[-1]
                print(f"[Fragment:init] {agent.name} → fragment end = {agent.fragment_end_node} "
                      f"(seg: {_fmt_path(agent.full_route)})")
            continue

        blocks = all_constraints.get(agent.name, [])

        # No constraints → keep full first fragment (what find_routes produced)
        if not blocks:
            agent.waiting = False
            agent.wait_node = None
            agent.blocked_by_node = None
            agent.blocker_owner = None
            agent.blocked_constraints = []
            # ensure first fragment exists
            if getattr(agent, "route", None):
                agent.fragments = [agent.route.copy()]
                seg_nodes = [agent.route[0][0]] + [v for (_, v) in agent.route]
            else:
                seg_nodes = agent.full_route
            agent.fragment_end_node = seg_nodes[-1]
            print(f"[Fragment:init] {agent.name} → fragment end = {agent.fragment_end_node} "
                  f"(seg: {_fmt_path(seg_nodes)})")
            continue

        # Find earliest blocked node along the current full route
        eb = _earliest_block_on_path(agent.full_route, blocks)

        # Constraints exist but not on the current remaining path → proceed unchanged
        if not eb:
            agent.waiting = False
            agent.wait_node = None
            agent.blocked_by_node = None
            agent.blocker_owner = None
            agent.blocked_constraints = blocks
            if getattr(agent, "route", None):
                seg_nodes = [agent.route[0][0]] + [v for (_, v) in agent.route]
            else:
                seg_nodes = agent.full_route
            agent.fragment_end_node = seg_nodes[-1]
            print(f"[Fragment:init] {agent.name} → fragment end = {agent.fragment_end_node} "
                  f"(seg: {_fmt_path(seg_nodes)})")
            continue

        # Cut before the first blocked node
        cut_idx, blocked_node, owner = eb
        last_safe = agent.full_route[cut_idx - 1]  # predecessor of the blocked node

        # Build new fragment from start -> last_safe (edges)
        new_frag: List[Tuple[str, str]] = []
        for i in range(cut_idx - 1):
            u, v = agent.full_route[i], agent.full_route[i + 1]
            new_frag.append((u, v))

        # Replace fragments/route/full_route to the safe prefix
        agent.fragments = [new_frag.copy()] if new_frag else []
        agent.route = new_frag[:]
        agent.full_route = [agent.start] + [v for (u, v) in new_frag] if new_frag else [agent.start]

        # Mark waiting at last_safe and store constraints
        agent.waiting = True
        agent.wait_node = last_safe
        agent.blocked_by_node = blocked_node
        agent.blocker_owner = owner
        agent.blocked_constraints = blocks
        agent.resume_ready = False

        # Log the safe fragment we’re taking now
        seg_nodes = [agent.start] + [v for (_, v) in new_frag] if new_frag else [agent.start]
        agent.fragment_end_node = seg_nodes[-1]
        print(f"[Fragment:init] {agent.name} → fragment end = {agent.fragment_end_node} "
              f"(seg: {_fmt_path(seg_nodes)}) | blocked ahead at {blocked_node} by {owner}")


# =============================== incremental replanning =========================

def replan_waiting_agents(agents: List[Agent], graph: nx.Graph, **kwargs) -> bool:
    """
    Fragment-to-next-safe policy with concise, deduped logging.

    For each waiting agent:
      1) Build a filtered shortest path (what is free *now*).
      2) Cut at the first currently-owned node → release only the safe prefix
         (source .. last-safe-before-critical). If none, FULL→goal.
      3) If prefixes still conflict, release ONE global winner and gate others
         behind the winner at the first shared node.

    Returns True iff at least one agent was released this call.
    """
    frame = kwargs.get("frame", 0)
    positions = kwargs.get("positions", {})
    release_delay_frames = int(kwargs.get("release_delay_frames", 0))
    reach_tol = float(kwargs.get("reach_tol", 0.20))
    start_frames = kwargs.get("start_frames", {})  # {agent_name: animation start frame}
    verbose_snapshots = bool(kwargs.get("verbose_snapshots", False))
    
    # ---- local helpers (logging + geometry) ----
    def _print_safepoints_snapshot():
        print("\n--- Safepoints snapshot ---")
        for a in agents:
            state = (
                "finished" if getattr(a, "finished", False)
                else ("waiting" if getattr(a, "waiting", False) else "active")
            )
            frag_end = getattr(a, "fragment_end_node", None)
            safepoint = frag_end or getattr(a, "wait_node", None) or "-"
            blocker = getattr(a, "blocker_owner", None)
            block_node = getattr(a, "blocked_by_node", None)
            block_txt = f"{blocker}@{block_node}" if (blocker and block_node) else "-"
            route_head = getattr(a, "full_route", []) or []
            print(f"  [{a.name}] {state:8s} | safe_until={safepoint:>6} | "
                  f"block={block_txt:>12} | goal={getattr(a, 'goal', None)} | "
                  f"route={_fmt_path(route_head, 12)}")
        print("—" * 48)

    def _log_gate(agent: Agent, owner_name: str, gate_node: str):
        """Print a gate line only when the owner@node pair changes."""
        key = (owner_name, gate_node)
        if getattr(agent, "_gate_state", None) != key:
            print(f"[i] {agent.name} gated by {owner_name} at {gate_node}")
            agent._gate_state = key

    def first_frame_at_node(blocker: Agent, node: str) -> Optional[int]:
        """Find the first animation frame where 'blocker' reaches 'node' (within tolerance)."""
        if not node or node not in positions or not getattr(blocker, "dynamic_coords", None):
            return None
        if not hasattr(blocker, "_node_frame_cache"):
            blocker._node_frame_cache = {}
        if node in blocker._node_frame_cache:
            return blocker._node_frame_cache[node]
        tx, ty = positions[node]
        tol2 = reach_tol * reach_tol
        for i, (x, y) in enumerate(blocker.dynamic_coords):
            dx, dy = x - tx, y - ty
            if dx * dx + dy * dy <= tol2:
                blocker._node_frame_cache[node] = i
                return i
        return None

    def has_reached_node(agent: Agent, node: str) -> bool:
        """
        Animation-aware reach check. Uses 'start_frames' to compute the agent's local
        timeline; returns True once the agent has reached 'node' plus any release delay.
        """
        local_start = int(start_frames.get(agent.name, 0))
        local_frame = frame - local_start
        if local_frame < 0:
            return False
        f = first_frame_at_node(agent, node)
        return f is not None and local_frame >= f + release_delay_frames

    def owner_has_cleared(owner: Agent, node: str) -> bool:
        """
        Edge-safe gating:
          - If 'node' is the owner's GOAL, treat it as occupied until 'finished'.
          - Otherwise require the owner to have reached the *next* node after 'node'.
        """
        if node == getattr(owner, "goal", None):
            return getattr(owner, "finished", False)

        fr = getattr(owner, "full_route", None) or []
        if node in fr:
            i = fr.index(node)
            if i + 1 < len(fr):
                next_node = fr[i + 1]
                return has_reached_node(owner, next_node)

        # If node not found in route, be conservative (still blocked).
        return False

    def _dist_in_path(path: List[str], node: str) -> float:
        """Distance along 'path' up to 'node' using graph edge weights (∞ if absent)."""
        if node not in path:
            return math.inf
        d = 0.0
        for i in range(path.index(node)):
            u, v = path[i], path[i + 1]
            d += float(graph[u][v].get("weight", 1.0))
        return d

    by_name = {a.name: a for a in agents}

    # ---- who is waiting now ----
    waiters: List[Agent] = [a for a in agents if getattr(a, "waiting", False) and not getattr(a, "replanned", False)]
    if not waiters:
        return False

    # ---- collect active owners (who currently "own" nodes) ----
    active_claims: Dict[str, List[Agent]] = {}
    for a in agents:
        if a in waiters:
            continue
        if getattr(a, "finished", False):
            g = getattr(a, "goal", None)
            if g:
                active_claims.setdefault(g, []).append(a)
        else:
            if getattr(a, "full_route", None):
                for n in a.full_route[1:]:
                    active_claims.setdefault(n, []).append(a)

    def owner_blocking(node: str) -> Optional[Agent]:
        """Return the active owner that is closest to 'node' (tie-break by priority)."""
        owners = active_claims.get(node, [])
        if not owners:
            return None
        return min(owners, key=lambda o: (_dist_in_path(getattr(o, "full_route", []) or [], node), o.priority()))

    # ---- build filtered shortest paths (free *now*) ----
    full_candidates: Dict[str, List[str]] = {}
    for a in waiters:
        # clear gate dedup once conditions may have changed
        if getattr(a, "_gate_state", None) is not None:
            a._gate_state = None

        Gf = graph.copy()

        # 1) Finished owners' goals are hard obstacles (except my goal & my source)
        finished_goals = {b.goal for b in agents if b is not a and getattr(b, "finished", False)}
        for g in list(finished_goals):
            if g != getattr(a, "goal", None) and g != getattr(a, "wait_node", None) and g in Gf:
                Gf.remove_node(g)

        # 2) Other waiters' current nodes are also hard obstacles (physical occupancy)
        other_wait_nodes = [
            getattr(w, "wait_node", None)
            for w in waiters
            if (w is not a and getattr(w, "wait_node", None))
        ]
        for wn in other_wait_nodes:
            if wn in Gf:
                Gf.remove_node(wn)

        # 3) Prune nodes blocked by ACTIVE owners that haven't cleared them yet
        to_remove = []
        for n in list(Gf.nodes):
            if n == getattr(a, "wait_node", None):
                continue
            owner = owner_blocking(n)
            if owner is not None and not owner_has_cleared(owner, n):
                to_remove.append(n)
        for n in to_remove:
            if n in Gf:
                Gf.remove_node(n)

        try:
            src = getattr(a, "wait_node", None)
            dst = getattr(a, "goal", None)
            if src not in Gf or dst not in Gf:
                raise nx.NodeNotFound
            p = nx.shortest_path(Gf, src, dst)
            full_candidates[a.name] = p
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    # optional debug dump
    if verbose_snapshots:
        print("\n[Replan] Candidates (filtered shortest paths):")
        for a in waiters:
            print(f"  - {a.name}: {_fmt_path(full_candidates.get(a.name))}")

    if not full_candidates:
        if verbose_snapshots:
            _print_safepoints_snapshot()
        return False

    # ---- cut each candidate to SAFE PREFIX (until first currently-owned node) ----
    safe_segments: Dict[str, List[str]] = {}
    gate_hint: Dict[str, Tuple[str, str]] = {}  # name -> (owner_name, gate_node)

    for name, path in full_candidates.items():
        block_index = None
        blocker_owner = None
        for idx, n in enumerate(path[1:], start=1):
            owner = owner_blocking(n)
            if owner is not None and not owner_has_cleared(owner, n):
                block_index = idx
                blocker_owner = owner
                break

        if block_index is None:
            safe_segments[name] = path  # FULL→goal
        else:
            # safe up to predecessor of first blocked node; record a gate hint
            if block_index - 1 >= 0:
                safe_segments[name] = path[:block_index]
                gate_hint[name] = (blocker_owner.name, path[block_index])

    if verbose_snapshots:
        print("[Replan] Safe segments (to next safepoint):")
        for a in waiters:
            if a.name in safe_segments:
                seg = safe_segments[a.name]
                end = seg[-1]
                suffix = "  [FULL→goal]" if end == getattr(a, "goal", None) else ""
                print(f"  - {a.name}: {_fmt_path(seg)}  (end={end}){suffix}")
            else:
                if a.name in gate_hint:
                    owner, node = gate_hint[a.name]
                    print(f"  - {a.name}: GATED by {owner} at {node} (hold at {getattr(a,'wait_node',None)})")
                else:
                    print(f"  - {a.name}: <no safe prefix now>")

    if not safe_segments:
        # Everyone blocked immediately: emit (deduped) gate info, then snapshot
        for name, hint in gate_hint.items():
            ag = by_name[name]
            _log_gate(ag, hint[0], hint[1])
        if verbose_snapshots:
            _print_safepoints_snapshot()
        return False

    # ---- arbitration among prefixes (release at most one conflicting group) ----
    claims: Dict[str, List[Agent]] = {}
    for name, seg in safe_segments.items():
        if len(seg) < 2:
            continue  # ignore no-op segments (single-node, no edge)
        ag = by_name[name]
        for n in seg[1:]:
            claims.setdefault(n, []).append(ag)

    conflict_nodes = {n for n, lst in claims.items() if len(lst) > 1}

    accepted: List[Agent] = []
    if conflict_nodes:
        contenders = set()
        for n in conflict_nodes:
            contenders.update(claims[n])

        def winner_key(ag: Agent):
            seg = safe_segments[ag.name]
            contested = [n for n in seg[1:] if n in conflict_nodes]
            d_first = min((_dist_in_path(seg, n) for n in contested), default=math.inf)
            return (d_first, ag.priority())

        winner = min(contenders, key=winner_key)
        accepted.append(winner)

        # Also accept disjoint segments
        used = set(safe_segments[winner.name][1:])
        for nm, seg in safe_segments.items():
            if len(seg) < 2:
                continue
            ag = by_name[nm]
            if ag in contenders or ag is winner:
                continue
            if set(seg[1:]).isdisjoint(used):
                accepted.append(ag)
                used |= set(seg[1:])
    else:
        # No conflicts: accept only segments that actually move
        accepted = [by_name[nm] for nm, seg in safe_segments.items() if len(seg) >= 2]

    # -------- apply releases --------
    any_resumed = False
    if accepted:
        print("[Replan] Accepted releases this pass:")
        for a in accepted:
            seg = safe_segments[a.name]             # <- node list of the released segment
            if len(seg) < 2:
                ...
                continue

            frag = [(seg[j], seg[j+1]) for j in range(len(seg)-1)]
            a.fragments.append(frag)
            a.route += frag

            # extend full_route without duplicating the boundary node
            if a.full_route and a.full_route[-1] == seg[0]:
                a.full_route += seg[1:]
            else:
                a.full_route += seg

            # 👇 add this line so the visualiser knows the resume anchor
            a.last_released_seg = seg[:]            # <— NEW

            a.replanned = True
            a.waiting = False
            a.blocked_by_node = None
            a.blocker_owner = None
            a.fragment_end_node = seg[-1]
            any_resumed = True
            print(f"  • {a.name} → fragment end = {a.fragment_end_node} (seg: {' -> '.join(seg)})")

        # Gate non-accepted candidates that had a direct gate hint
        for nm, hint in gate_hint.items():
            ag = by_name[nm]
            if ag in accepted:
                continue
            ag.waiting = True
            ag.blocker_owner, ag.blocked_by_node = hint
            _log_gate(ag, ag.blocker_owner, ag.blocked_by_node)

        if verbose_snapshots:
            _print_safepoints_snapshot()
        return any_resumed

    # Nobody accepted (rare) — still report gates and snapshot
    for nm, hint in gate_hint.items():
        ag = by_name[nm]
        ag.waiting = True
        ag.blocker_owner, ag.blocked_by_node = hint
        _log_gate(ag, ag.blocker_owner, ag.blocked_by_node)

    if verbose_snapshots:
        _print_safepoints_snapshot()
    return False
