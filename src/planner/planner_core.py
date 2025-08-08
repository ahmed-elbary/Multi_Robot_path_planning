import networkx as nx
from typing import List, Dict, Tuple
from .agent import Agent
import heapq
from bisect import bisect_right

def interval_conflicts(existing: List[Tuple[float,float]], start: float, end: float) -> bool:
    i = bisect_right(existing, (start, end))
    if i > 0 and existing[i-1][1] > start:
        return True
    if i < len(existing) and existing[i][0] < end:
        return True
    return False

node_reservations: Dict[str, List[Tuple[float,float]]] = {}
edge_reservations: Dict[Tuple[str,str], List[Tuple[float,float]]] = {}

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

def find_routes(agents, graph, reservation_table=None, edge_reservations=None):
    node_reservations = reservation_table if reservation_table is not None else {}
    edge_reservations = edge_reservations if edge_reservations is not None else {}

    for agent in agents:
        t = 0.0
        print(f"\n[✓] Planning path for {agent.name} from {agent.start} to {agent.goal}")
        path = time_aware_shortest_path(graph, agent.start, agent.goal)
        agent.original_path_length = len(agent.full_route)

        timed_path = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            weight = graph[u][v].get('weight', 1.0)
            depart = t
            arrival = depart + weight

            t_node = next_free(node_reservations.setdefault(v, []), depart)
            rev = (v, u)
            t_edge_rev = next_free(edge_reservations.setdefault(rev, []), depart)
            wait_until = max(t_node, t_edge_rev)
            if wait_until > depart:
                depart = wait_until
                arrival = depart + weight

            book_node(v, depart, depart + EPS, agent.name, node_reservations)
            print(f"[{agent.name}] Booked node '{v}' from {depart:.3f} to {depart+EPS:.3f}")

            book_edge((u, v), depart, arrival, agent.name, edge_reservations)
            print(f"[{agent.name}] Booked edge '{u} → {v}' from {depart:.3f} to {arrival:.3f}")

            book_node(v, arrival, arrival + 1.0, agent.name, node_reservations)
            print(f"[{agent.name}] Booked node '{v}' (hold) from {arrival:.3f} to {arrival+1.0:.3f}")

            timed_path.append((u, v, depart, arrival))
            t = arrival

        agent.route = timed_path
        agent.full_route = [agent.start] + [v for _, v, _, _ in timed_path]
        agent.arrival_time = t
        print(f"[{agent.name}] Final arrival time: {t:.3f}")

    return node_reservations, edge_reservations

def time_aware_shortest_path(graph: nx.Graph, start: str, goal: str,
                             max_time: float = 100.0,
                             reservation_table=None,
                             edge_reservations_arg=None) -> List[str]:
    node_res = reservation_table if reservation_table is not None else node_reservations
    edge_res = edge_reservations_arg if edge_reservations_arg is not None else edge_reservations

    frontier = [(0, 0.0, start, [])]
    visited = set()
    while frontier:
        cost, t, cur, path = heapq.heappop(frontier)
        if (cur, t) in visited or t > max_time:
            continue
        visited.add((cur, t))
        path2 = path + [cur]
        if cur == goal:
            return path2
        if not interval_conflicts(node_res.get(cur, []), t, t+1):
            heapq.heappush(frontier, (cost+0.1, t+1, cur, path2))
        for nbr in graph.neighbors(cur):
            if interval_conflicts(node_res.get(nbr, []), t+1, t+1+1e-3):
                continue
            if interval_conflicts(edge_res.get((cur,nbr), []), t, t+1):
                continue
            if interval_conflicts(edge_res.get((nbr,cur), []), t, t+1):
                continue
            w = graph[cur][nbr].get('weight', 1)
            heapq.heappush(frontier, (cost + w, t+1, nbr, path2))
    raise Exception("No valid time-aware path found")

def find_critical_points(agents: List[Agent]) -> Dict[str, List[str]]:
    point_to_agents = {}
    for agent in agents:
        visited_nodes = set()
        for u, v, _, _ in agent.route:
            visited_nodes.add(u)
            visited_nodes.add(v)
        for node in visited_nodes:
            point_to_agents.setdefault(node, []).append(agent.name)
    return {k: v for k, v in point_to_agents.items() if len(v) > 1}

def agent_has_priority(agent: Agent, other_agents: List[Agent], node: str) -> bool:
    def node_index(a: Agent, n: str) -> int:
        for i, (_, v, _, _) in enumerate(a.route):
            if v == n:
                return i
        return float('inf')

    competing = [a for a in other_agents if any(v == node for _, v, _, _ in a.route)]
    competing.sort(key=lambda a: (node_index(a, node), len(a.route), a.priority()))
    return competing and competing[0].name == agent.name

def split_critical_paths(graph, agents: List[Agent], dangerous_points: List[str]):
    for agent in agents:
        agent.fragments = []
        frag = []

        for i, (u, v, t1, t2) in enumerate(agent.route):
            frag.append((u, v, t1, t2))

            if v in dangerous_points:
                if not agent_has_priority(agent, agents, v):
                    # Identify safest previous node
                    wait_node = agent.start
                    safe_index = None
                    for j in range(i - 1, -1, -1):
                        if frag[j][1] not in dangerous_points:
                            wait_node = frag[j][1]
                            safe_index = j
                            break

                    trimmed_path = frag[:safe_index + 1] if safe_index is not None else frag[:1]

                    # Build up reservation tables from earlier agents only
                    node_res, edge_res = build_reservations_for_replan(agents, agent)

                    # Attempt temporary replan WITH reservations
                    temp_path = time_aware_shortest_path(
                        graph, wait_node, agent.goal, 
                        reservation_table=node_res, 
                        edge_reservations_arg=edge_res
                    )
                    replanned_length = len(temp_path)

                    # Decision: wait or replan
                    if replanned_length >= agent.original_path_length + 3:
                        agent.waiting = True
                        agent.wait_node = wait_node
                        agent.route = trimmed_path
                        agent.fragments.append(trimmed_path)
                        print(f"[Wait] {agent.name} will wait at {wait_node} (replanned path too long).")
                    else:
                        agent.waiting = False
                        agent.route = frag + [step for step in agent.route[len(frag):]]
                        agent.fragments.append(agent.route)
                        print(f"[Replan] {agent.name} will replan instead of wait.")

                    agent.arrival_time = trimmed_path[-1][3] if trimmed_path else 0.0
                    agent.resume_time = frag[-1][2] if frag else agent.arrival_time + 1.0
                    break  # Stop further path building after deciding
        else:
            agent.fragments.append(frag)


def assign_waiting_agents(agents):
    pass

def replan_waiting_agents(agents, graph: nx.Graph, frame=0, fps=10):
    global node_reservations, edge_reservations
    start_time = frame / fps
    print(f"[Replan] Triggered at frame {frame}, time {start_time:.2f}")
    for agent in agents:
        if not agent.waiting or agent.replanned:
            continue
        print(f"[Replan] Replanning for {agent.name} from {agent.wait_node} to {agent.goal}")
        try:
            path = time_aware_shortest_path(graph, agent.wait_node, agent.goal)
            print(f"[Replan] {agent.name} replanning path: {path}")
        except Exception as e:
            print(f"[Error] Could not replan path for {agent.name}: {e}")
            agent.replanned = True
            continue
        timed_path = []
        t = start_time
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u not in graph or v not in graph[u]:
                print(f"[Error] Edge ({u} → {v}) not found in graph for {agent.name}. Skipping replanning.")
                agent.replanned = True
                agent.waiting = False
                break
            weight = graph[u][v].get('weight', 1.0)
            depart = t
            arrival = depart + weight
            t_node = next_free(node_reservations.setdefault(v, []), depart)
            rev = (v, u)
            t_edge_rev = next_free(edge_reservations.setdefault(rev, []), depart)
            wait_until = max(t_node, t_edge_rev)
            if wait_until > depart:
                depart = wait_until
                arrival = depart + weight
            book_node(v, depart, depart + EPS, agent.name, node_reservations)
            book_edge((u, v), depart, arrival, agent.name, edge_reservations)
            book_node(v, arrival, arrival + 1.0, agent.name, node_reservations)
            timed_path.append((u, v, depart, arrival))
            t = arrival
        if timed_path:
            agent.fragments.append(timed_path)
            agent.route += timed_path
            agent.full_route += [v for _, v, _, _ in timed_path]
            agent.arrival_time = t
            agent.replanned = True
            agent.waiting = False
            print(f"[Replan] {agent.name} new arrival time: {t:.2f}")

def build_temporal_occupancy_table(agents: List[Agent]) -> Dict[int, Dict[str, str]]:
    table = {}
    max_len = max(len(agent.full_route) for agent in agents if agent.full_route)
    for t in range(max_len):
        table[t] = {}
        for agent in agents:
            if t < len(agent.full_route):
                node = agent.full_route[t]
                table[t][node] = agent.name
    return table

def build_reservations_for_replan(agents, current_agent):
    node_res = {}
    edge_res = {}
    for a in agents:
        if a is current_agent:
            break  # Only build up to the current agent
        for (u, v, t1, t2) in a.route:
            node_res.setdefault(v, []).append((t1, t2, a.name))
            edge_res.setdefault((u, v), []).append((t1, t2, a.name))
    return node_res, edge_res
