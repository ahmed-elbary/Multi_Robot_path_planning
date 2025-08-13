# fragment_planner.py
import networkx as nx
from typing import List, Dict, Tuple, Optional
from .agent import Agent
import math


def _node_claims(agents: List[Agent]) -> Dict[str, List[Agent]]:
    """node -> agents whose full_route contains that node (excluding start)."""
    claims: Dict[str, List[Agent]] = {}
    for a in agents:
        if getattr(a, "full_route", None):
            for n in a.full_route[1:]:
                claims.setdefault(n, []).append(a)
    return claims

def _compute_block_constraints(agents: List[Agent], graph: nx.Graph) -> Dict[str, List[Tuple[str, str]]]:
    """
    For each agent, compute list of (node, owner_name) it must yield at,
    based on nearest-wins priority at that node.
    """
    constraints: Dict[str, List[Tuple[str, str]]] = {}
    claims = _node_claims(agents)
    for node, claimers in claims.items():
        if len(claimers) < 2:
            continue
        # decide the winner at this node
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

def find_routes(agents: List[Agent], graph: nx.Graph):
    """
    Space-aware: plan each agent sequentially on a filtered copy of the map.
    Nodes occupied by previously planned agents are removed.
    """
    all_starts = {a.start for a in agents}
    for i, agent in enumerate(agents):
        print(f"\n[✓] Planning spatial route for {agent.name} from {agent.start} to {agent.goal}")
        occupied = all_starts - {agent.start}
        for prev in agents[:i]:
            occupied.update(prev.full_route)

        filtered = graph.copy()
        for node in occupied:
            if node not in (agent.start, agent.goal) and node in filtered:
                filtered.remove_node(node)

        print(f"[Debug] {agent.name} sees filtered map with {len(filtered.nodes)} nodes, {len(filtered.edges)} edges.")
        print(f"[✓] Filtered map for {agent.name} — Occupied nodes removed: {sorted(list(occupied))}")

        try:
            path = nx.shortest_path(filtered, agent.start, agent.goal)
            agent.route = [(path[j], path[j+1]) for j in range(len(path)-1)]
            agent.full_route = path
            agent.fragments = [agent.route]
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
    Return all critical nodes shared by more than one agent on their route.
    """
    node_claims: Dict[str, List[str]] = {}
    for agent in agents:
        for _, v in agent.route:
            node_claims.setdefault(v, []).append(agent.name)
    return {n: names for n, names in node_claims.items() if len(names) > 1}

def _distance_to_node_in_route(agent: Agent, node: str, graph: nx.Graph) -> float:
    if not agent.full_route or node not in agent.full_route:
        return math.inf
    idx = agent.full_route.index(node)
    dist = 0.0
    for i in range(idx):
        u, v = agent.full_route[i], agent.full_route[i+1]
        dist += float(graph[u][v].get('weight', 1.0))
    return dist

def agent_has_priority(agents: List[Agent], node: str, graph: nx.Graph) -> Optional[Agent]:
    claimers = [a for a in agents if a.full_route and node in a.full_route]
    if not claimers:
        return None
    scored = [( _distance_to_node_in_route(a, node, graph), a.priority(), a) for a in claimers]
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]

def split_critical_paths(graph: nx.Graph, agents: List[Agent], dangerous_points: List[str]):
    """
    Multi-robot split: compute all critical ownerships, then for each agent
    cut its route to the earliest blocked node and mark it waiting there.
    """
    # Build full constraint sets across all nodes/agents
    all_constraints = _compute_block_constraints(agents, graph)

    for agent in agents:
        # full_route might be just [start] for waiting agents
        if not getattr(agent, "full_route", None) or len(agent.full_route) < 2:
            continue

        blocks = all_constraints.get(agent.name, [])
        if not blocks:
            # no constraints → can keep full route (first fragment is full)
            agent.waiting = False
            agent.wait_node = None
            agent.blocked_by_node = None
            agent.blocker_owner = None
            agent.blocked_constraints = []
            # ensure first fragment exists
            agent.fragments = [agent.route] if getattr(agent, "route", None) else agent.fragments
            continue

        # find earliest blocked node along the *current* full route
        eb = _earliest_block_on_path(agent.full_route, blocks)
        if not eb:
            # constraints exist but not on the current remaining path (rare) → proceed
            agent.waiting = False
            agent.wait_node = None
            agent.blocked_by_node = None
            agent.blocker_owner = None
            agent.blocked_constraints = blocks  # keep for future
            continue

        cut_idx, blocked_node, owner = eb
        # last safe node is the predecessor of the blocked node
        last_safe = agent.full_route[cut_idx-1]

        # Build new fragment from start -> last_safe (edges)
        new_frag: List[Tuple[str, str]] = []
        for i in range(cut_idx-1):
            u, v = agent.full_route[i], agent.full_route[i+1]
            new_frag.append((u, v))

        # Replace fragments/route/full_route to the safe prefix
        agent.fragments = [new_frag] if new_frag else []
        agent.route = new_frag
        agent.full_route = [agent.start] + [v for (u, v) in new_frag] if new_frag else [agent.start]

        # Mark waiting at last_safe and store constraints (all of them)
        agent.waiting = True
        agent.wait_node = last_safe
        agent.blocked_by_node = blocked_node           # for backward compat / logs
        agent.blocker_owner = owner
        agent.blocked_constraints = blocks             # NEW: keep all blockers
        agent.resume_ready = False

        print(f"[{agent.name}] Wait at {last_safe} due to collision at {blocked_node} (blocked by {owner})")

def assign_waiting_agents(agents):  # optional extension; currently handled in split
    pass

def replan_waiting_agents(agents: List[Agent], graph: nx.Graph, **kwargs) -> bool:
    """
    NO-PARTIALS release with blocker-aware gating.

    Strict policy (default): a waiter may be released only if there exists a full,
    currently-free path to its goal AND every owner whose route claims any node
    on that path is FINISHED (owner-level gate).
    """
    frame = kwargs.get("frame", 0)
    positions = kwargs.get("positions", {})
    release_delay_frames = int(kwargs.get("release_delay_frames", 0))
    reach_tol = float(kwargs.get("reach_tol", 0.20))
    finished_owners = set(kwargs.get("finished_owners", set()))
    strict_owner_gate = bool(kwargs.get("strict_owner_gate", True))
    start_frames = kwargs.get("start_frames", {})  # optional; ok if missing

    def _log_gate(agent: Agent, owner_name: str, gate_node: str):
        # print only if gate changed (debounce spam)
        key = (owner_name, gate_node)
        prev = getattr(agent, "_gate_state", None)
        if prev != key:
            print(f"[i] {agent.name} strictly gated by {owner_name} at {gate_node}")
            agent._gate_state = key

    def first_frame_at_node(blocker: Agent, node: str) -> Optional[int]:
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
            if dx*dx + dy*dy <= tol2:
                blocker._node_frame_cache[node] = i
                return i
        return None

    def has_reached_node(agent: Agent, node: str) -> bool:
        # terminal runs: finished agents count as having reached nodes on their route
        if getattr(agent, "finished", False):
            fr = getattr(agent, "full_route", []) or []
            return node in fr
        # animation runs: compare against *local* time since this path started (optional)
        local_start = int(start_frames.get(agent.name, 0)) if start_frames else 0
        local_frame = frame - local_start
        if local_frame < 0:
            return False
        f = first_frame_at_node(agent, node)
        return f is not None and local_frame >= f + release_delay_frames

    def _distance_to_node_in_route(agent: Agent, node: str, G: nx.Graph) -> float:
        if not agent.full_route or node not in agent.full_route:
            return math.inf
        idx = agent.full_route.index(node)
        dist = 0.0
        for i in range(idx):
            u, v = agent.full_route[i], agent.full_route[i+1]
            dist += float(G[u][v].get("weight", 1.0))
        return dist

    by_name = {a.name: a for a in agents}

    # ---- who is waiting this round?
    waiters: List[Agent] = [a for a in agents if getattr(a, "waiting", False) and not getattr(a, "replanned", False)]
    if not waiters:
        return False

    # ---- active claims: moving owners claim their future nodes; finished owners only their GOAL
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

    def blocked_by_active(node: str) -> Optional[Agent]:
        owners = active_claims.get(node, [])
        if not owners:
            return None
        owner = min(owners, key=lambda a: (_distance_to_node_in_route(a, node, graph), a.priority()))
        owner_at_node = has_reached_node(owner, node)
        # Goal: free until arrival; after arrival stays blocked (parking)
        if node == getattr(owner, "goal", None):
            return owner if owner_at_node else None
        # Non-goal: blocked until the owner reaches it; then it frees
        return owner if not owner_at_node else None

    # ---- losers gated behind a specific blocker until that blocker reaches its gate node
    gated: List[Agent] = []
    for a in waiters:
        owner_name = getattr(a, "blocker_owner", None)
        gate_node = getattr(a, "blocked_by_node", None)
        if not owner_name or not gate_node:
            continue
        owner = by_name.get(owner_name)
        if not owner:
            continue
        if gate_node == getattr(owner, "goal", None):
            if (owner.name not in finished_owners) and (not has_reached_node(owner, owner.goal)):
                gated.append(a)
                _log_gate(a, owner.name, gate_node)
        else:
            if not has_reached_node(owner, gate_node):
                gated.append(a)
                _log_gate(a, owner.name, gate_node)

    # ---- Step 1: build candidate FULL paths for not-gated waiters only (free *now*)
    candidate_paths: Dict[str, List[str]] = {}
    for a in waiters:
        if a in gated or not getattr(a, "wait_node", None):
            continue

        Gf = graph.copy()

        # Finished owners' goals are hard obstacles, except if it's my goal or my source
        finished_goals = {b.goal for b in agents if b is not a and getattr(b, "finished", False)}
        for g in list(finished_goals):
            if g != a.goal and g != a.wait_node and g in Gf:
                Gf.remove_node(g)

        # Remove every node currently blocked by an ACTIVE owner (keep my source)
        to_remove = []
        for n in list(Gf.nodes):
            if n == a.wait_node:
                continue
            if blocked_by_active(n) is not None:
                to_remove.append(n)
        for n in to_remove:
            if n in Gf:
                Gf.remove_node(n)

        try:
            if a.wait_node not in Gf or a.goal not in Gf:
                raise nx.NodeNotFound
            p = nx.shortest_path(Gf, a.wait_node, a.goal)
            candidate_paths[a.name] = p
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    if not candidate_paths:
        # clear gate spam only for agents that truly have no gate this round
        for a in waiters:
            if a not in gated and getattr(a, "_gate_state", None) is not None:
                a._gate_state = None
        return False

    # ---- Step 2: STRICT OWNER GATE (only release if ALL owners on the path are finished)
    gated_any = {a.name for a in gated}  # accumulate all gated this round (any rule)
    if strict_owner_gate:
        filtered: Dict[str, List[str]] = {}
        for name, p in candidate_paths.items():
            unfinished_owners_on_path = set()
            for n in p[1:]:
                for o in active_claims.get(n, []):
                    if not getattr(o, "finished", False):
                        unfinished_owners_on_path.add(o.name)

            if unfinished_owners_on_path - finished_owners:
                # gate by the earliest unfinished owner on the candidate path
                a = by_name[name]
                for n in p[1:]:
                    owners_here = [o for o in active_claims.get(n, []) if not getattr(o, "finished", False)]
                    if owners_here:
                        a.waiting = True
                        a.blocker_owner = owners_here[0].name
                        a.blocked_by_node = n
                        _log_gate(a, a.blocker_owner, a.blocked_by_node)
                        gated_any.add(a.name)
                        break
                continue

            filtered[name] = p
        candidate_paths = filtered

    if not candidate_paths:
        # clear gate spam only for those NOT gated by any rule this round
        for a in waiters:
            if a.name not in gated_any and getattr(a, "_gate_state", None) is not None:
                a._gate_state = None
        return False

    # ---- Step 3: waiter-vs-waiter arbitration (FULL only)
    this_round_path: Dict[str, List[str]] = dict(candidate_paths)

    claims: Dict[str, List[Agent]] = {}
    def _dist_in_path(path: List[str], node: str) -> float:
        if node not in path:
            return math.inf
        idx, d = path.index(node), 0.0
        for i in range(idx):
            u, v = path[i], path[i+1]
            d += float(graph[u][v].get('weight', 1.0))
        return d

    for name, p in this_round_path.items():
        ag = by_name[name]
        for n in p[1:]:
            claims.setdefault(n, []).append(ag)

    conflict_nodes = {n for n, ls in claims.items() if len(ls) > 1}

    any_resumed = False
    accepted: List[Agent] = []

    if conflict_nodes:
        # Choose ONE global winner nearest to its own earliest contested node (tie-break by priority)
        contenders = set()
        for n in conflict_nodes:
            contenders.update(claims[n])

        def winner_key(ag: Agent):
            p = this_round_path[ag.name]
            contested = [n for n in p[1:] if n in conflict_nodes]
            d_first = min((_dist_in_path(p, n) for n in contested), default=math.inf)
            return (d_first, ag.priority())

        winner = min(contenders, key=winner_key)
        accepted = [winner]  # <-- ONLY the winner this pass

        # Gate losers that overlap winner behind the winner's GOAL (log once)
        winner_nodes = set(this_round_path[winner.name][1:])
        for name, p in this_round_path.items():
            if name == winner.name:
                continue
            if not set(p[1:]).isdisjoint(winner_nodes):
                ag = by_name[name]
                ag.waiting = True
                ag.blocker_owner = winner.name
                ag.blocked_by_node = by_name[winner.name].goal
                _log_gate(ag, ag.blocker_owner, ag.blocked_by_node)

    else:
        # No conflicts → accept all (still require node-disjointness among them)
        used_nodes: set = set()
        for name, p in this_round_path.items():
            if set(p[1:]).isdisjoint(used_nodes):
                accepted.append(by_name[name])
                used_nodes |= set(p[1:])

    # ---- Step 4: apply releases
    if accepted:
        for a in accepted:
            p = this_round_path[a.name]
            frag = [(p[j], p[j+1]) for j in range(len(p)-1)]
            a.fragments.append(frag)
            a.route += frag
            if a.full_route and a.full_route[-1] == p[0]:
                a.full_route += p[1:]
            else:
                a.full_route += p
            a.replanned = True
            a.waiting = False
            a.blocked_by_node = None
            a.blocker_owner = None
            a._gate_state = None  # clear debounce state after release
            print(f"[FragmentPlanner] {a.name} appended fragment: {p}")
            any_resumed = True
        return any_resumed

    # clear stale gate states for any waiter not gated this round
    for a in waiters:
        if a.name not in gated_any and getattr(a, "_gate_state", None) is not None:
            a._gate_state = None
    return False
