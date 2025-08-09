import networkx as nx
from typing import List, Dict, Tuple
from .agent import Agent
import heapq
from bisect import bisect_right

# === Reservation-table helpers ===

def interval_conflicts(existing: list, start: float, end: float) -> bool:
    """Check if [start, end) conflicts with any existing reservation intervals."""
    intervals = sorted(
        (t[0], t[1]) if len(t) >= 2 else t for t in existing
    )
    from bisect import bisect_right
    i = bisect_right(intervals, (start, end))
    if i > 0 and intervals[i-1][1] > start:
        return True
    if i < len(intervals) and intervals[i][0] < end:
        return True
    return False

def edge_swap_conflict(u, v, t_start, t_end, edge_res):
    """Check if the edge (v, u) is reserved during [t_start, t_end)."""
    reverse = (v, u)
    if reverse in edge_res:
        for booked_start, booked_end, _ in edge_res[reverse]:
            # Intervals overlap if not (end <= start2 or start >= end2)
            if not (t_end <= booked_start or t_start >= booked_end):
                return True
    return False

node_reservations: Dict[str, List[Tuple[float,float,str]]] = {}
edge_reservations: Dict[Tuple[str,str], List[Tuple[float,float,str]]] = {}

def book_node(node, t_start, t_end, agent, node_res):
    if node not in node_res:
        node_res[node] = []
    node_res[node].append((t_start, t_end, agent))

def book_edge(edge, t_start, t_end, agent, edge_res):
    if edge not in edge_res:
        edge_res[edge] = []
    edge_res[edge].append((t_start, t_end, agent))

EPS = 1e-3

def next_free(intervals, t, eps=EPS):
    intervals = sorted(intervals)
    t_curr = t
    for (s,e,_) in intervals:
        if t_curr + eps <= s:
            break
        if t_curr < e:
            t_curr = e
    return t_curr

# === MAIN PLANNER ===
def find_routes(agents, graph, reservation_table=None, edge_reservations_arg=None):
    global node_reservations, edge_reservations
    if reservation_table is not None:
        node_reservations = reservation_table
    if edge_reservations_arg is not None:
        edge_reservations = edge_reservations_arg

    for agent in agents:
        print(f"\n[✓] Planning path for {agent.name} from {agent.start} to {agent.goal}")
        found = False
        t0 = 0.0
        max_wait = 100
        while t0 <= max_wait:
            try:
                path, times = time_aware_shortest_path(graph, agent.start, agent.goal, start_time=t0)
                found = True
                break
            except Exception:
                t0 += 1.0
        if not found:
            print(f"[!] {agent.name} could not find any feasible time-aware path from {agent.start} to {agent.goal}")
            agent.route = []
            agent.full_route = []
            agent.arrival_time = None
            continue

        timed_path = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            depart = times[i]
            arrival = times[i + 1]
            if not graph.has_edge(u, v):
                print(f"[Warning] Edge '{u} → {v}' does not exist in the map. Skipping.")
                continue

            # Book node u from depart to arrival (i.e. robot is occupying node u until it leaves)
            book_node(u, depart, arrival, agent.name, node_reservations)
            print(f"[{agent.name}] Booked node '{u}' from {depart:.3f} to {arrival:.3f}")

            # Book node v from arrival to arrival + 1.0 (robot sits at arrival node for 1s after arriving)
            book_node(v, arrival, arrival + 1.0, agent.name, node_reservations)
            print(f"[{agent.name}] Booked node '{v}' from {arrival:.3f} to {arrival+1.0:.3f}")

            # Book the edge only in the direction traversed
            book_edge((u, v), depart, arrival, agent.name, edge_reservations)
            book_edge((v, u), depart, arrival, agent.name, edge_reservations)
            print(f"[{agent.name}] Booked edge '{u} → {v}' from {depart:.3f} to {arrival:.3f}")
            
            timed_path.append((u, v, depart, arrival))
        agent.route = timed_path
        agent.full_route = path
        agent.arrival_time = times[-1] if times else 0.0
        print(f"[{agent.name}] Final arrival time: {agent.arrival_time:.3f}")

    return node_reservations, edge_reservations

def time_aware_shortest_path(graph: nx.Graph, start: str, goal: str,
                             max_time: float = 100.0,
                             start_time: float = 0.0,
                             reservation_table=None,
                             edge_reservations_arg=None) -> Tuple[List[str], List[float]]:
    node_res = reservation_table if reservation_table is not None else node_reservations
    edge_res = edge_reservations_arg if edge_reservations_arg is not None else edge_reservations

    # (cost, t, node, path, times)
    frontier = [(0, start_time, start, [start], [start_time])]
    visited = set()
    while frontier:
        cost, t, cur, path, times = heapq.heappop(frontier)
        if (cur, t) in visited or t > max_time:
            continue
        visited.add((cur, t))
        if cur == goal:
            return path, times
        # Try to wait at cur
        if not interval_conflicts(node_res.get(cur, []), t, t+1):
            heapq.heappush(frontier, (cost+0.1, t+1, cur, path[:], times + [t+1]))
        for nbr in graph.neighbors(cur):
            w = graph[cur][nbr].get('weight', 1)
            arrival = t + w
            # Block if node/edge/reverse edge not available for the whole period
            if interval_conflicts(node_res.get(nbr, []), arrival, arrival+1):
                continue
            if interval_conflicts(edge_res.get((cur, nbr), []), t, arrival):
                continue
            if interval_conflicts(edge_res.get((nbr, cur), []), t, arrival):
                continue
            heapq.heappush(frontier, (cost + w, arrival, nbr, path + [nbr], times + [arrival]))

    raise Exception("No valid time-aware path found")

def assign_waiting_agents(agents): pass
def replan_waiting_agents(agents, graph: nx.Graph, frame=0, fps=10): pass
def split_critical_paths(*args, **kwargs): pass  # Disabled for now
