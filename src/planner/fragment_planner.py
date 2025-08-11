# fragment_planner.py
import networkx as nx
from typing import List, Dict, Tuple, Optional
from .agent import Agent
import math

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
    Apply nearest-wins rule to each critical point.
    Non-priority agents truncate before the shared node and wait.
    """
    for cp in dangerous_points:
        winner = agent_has_priority(agents, cp, graph)
        if winner:
            print(f"[Critical] Node {cp} winner: {winner.name}")
        for agent in agents:
            if not agent.full_route or cp not in agent.full_route:
                continue
            if agent is winner:
                agent.waiting = False
                agent.wait_node = None
                agent.blocked_by_node = None
                agent.blocker_owner = None
                continue
            idx = agent.full_route.index(cp)
            if idx == 0:
                continue
            last_safe = agent.full_route[idx-1]
            new_frag = []
            for i in range(idx-1):
                u, v = agent.full_route[i], agent.full_route[i+1]
                new_frag.append((u, v))
            agent.fragments = [new_frag] if new_frag else []
            agent.route = new_frag
            agent.full_route = [agent.start] + [v for u, v in new_frag] if new_frag else [agent.start]
            agent.waiting = True
            agent.wait_node = last_safe
            agent.blocked_by_node = cp
            agent.blocker_owner = winner.name if winner else None
            agent.resume_ready = False
            print(f"[{agent.name}] Wait at {last_safe} due to collision at {cp} (blocked by {agent.blocker_owner})")

def assign_waiting_agents(agents):  # optional extension; currently handled in split
    pass

def replan_waiting_agents(agents: List[Agent], graph: nx.Graph, **kwargs) -> bool:
    """
    During animation: if the blocking agent has passed the shared node,
    the blocked agent replans from its wait_node on the full map.
    Fallback: if an agent is waiting without an explicit blocker (no critical point),
    try to plan on the full graph once conditions likely changed.
    """
    frame = kwargs.get("frame", 0)
    positions = kwargs.get("positions", {})
    release_delay_frames = int(kwargs.get("release_delay_frames", 0))

    def first_frame_at_node(blocker: Agent, node: str) -> Optional[int]:
        if not node or node not in positions or not getattr(blocker, "dynamic_coords", None):
            return None
        if not hasattr(blocker, "_node_frame_cache"):
            blocker._node_frame_cache = {}
        if node in blocker._node_frame_cache:
            return blocker._node_frame_cache[node]
        target_xy = positions[node]
        for i, (x, y) in enumerate(blocker.dynamic_coords):
            if math.isclose(x, target_xy[0], abs_tol=1e-6) and math.isclose(y, target_xy[1], abs_tol=1e-6):
                blocker._node_frame_cache[node] = i
                return i
        return None

    by_name = {a.name: a for a in agents}
    any_resumed = False

    for agent in agents:
        if not agent.waiting or agent.replanned:
            continue

        blocked_node = getattr(agent, "blocked_by_node", None)
        blocker_owner = getattr(agent, "blocker_owner", None)

        # Case A: explicit blocker at a critical node
        if blocked_node and blocker_owner:
            blocker = by_name.get(blocker_owner)
            if blocker is None:
                continue
            f_node = first_frame_at_node(blocker, blocked_node)
            if f_node is not None and frame >= f_node + release_delay_frames:
                resume_now = True
            else:
                resume_now = False
        else:
            # Case B: no explicit blocker (just "no path" earlier).
            # Try a normal full-graph path now.
            resume_now = True  # allow attempts as soon as caller triggers a replan

        if resume_now:
            try:
                path = nx.shortest_path(graph, agent.wait_node, agent.goal)
                frag = [(path[j], path[j+1]) for j in range(len(path)-1)]
                # append the new fragment
                agent.fragments.append(frag)
                agent.route += frag
                # extend full route (avoid duplicating source if matches last)
                if agent.full_route and agent.full_route[-1] == path[0]:
                    agent.full_route += path[1:]
                else:
                    agent.full_route += path
                agent.waiting = False
                agent.replanned = True
                agent.blocked_by_node = None
                agent.blocker_owner = None
                any_resumed = True
            except nx.NetworkXNoPath:
                # still blocked; keep waiting
                pass

    return any_resumed
