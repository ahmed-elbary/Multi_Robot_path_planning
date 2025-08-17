# utils.py
import yaml
import networkx as nx
import math
from typing import Dict, Tuple, List

def load_map(yaml_path: str) -> Dict:
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def build_graph_from_yaml(data: list) -> nx.Graph:
    graph = nx.DiGraph()
    # add nodes with pos
    for entry in data:
        node_data = entry['node']
        name = node_data['name']
        x = node_data['pose']['position']['x']
        y = node_data['pose']['position']['y']
        graph.add_node(name, pos=(x, y))

    # add directed edges with euclidean weight
    for entry in data:
        u = entry['node']['name']
        ux, uy = graph.nodes[u]['pos']
        for edge in entry['node'].get('edges', []):
            v = edge['node']
            vx, vy = graph.nodes[v]['pos']
            w = math.hypot(vx - ux, vy - uy)
            graph.add_edge(u, v, weight=w)
    return graph


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_occupied_nodes(agents, graph, proximity_thresh: float = 0.5) -> list:
    occupied = set()
    for agent in agents:
        if agent.active:
            for step in agent.route:
                if isinstance(step, tuple):
                    occupied.update(step[:2])
    return list(occupied)


def generate_filtered_map(graph: nx.Graph, occupied_nodes: List[str], start: str, goal: str) -> nx.Graph:
    G = graph.copy()
    G.remove_nodes_from(n for n in occupied_nodes if n not in [start, goal])
    return G

# ======================== Evaluation helpers ========================

def path_length(nodes: List[str], graph: nx.Graph) -> float:
    if not nodes or len(nodes) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(nodes, nodes[1:]):
        if graph.has_edge(u, v):
            total += float(graph[u][v].get("weight", euclidean_distance(graph.nodes[u]["pos"], graph.nodes[v]["pos"])))
        else:
            # If edge was pruned but existed geometrically, fall back to straight-line
            total += euclidean_distance(graph.nodes[u]["pos"], graph.nodes[v]["pos"])
    return total

def sum_of_costs(agents, graph: nx.Graph) -> float:
    return sum(path_length(getattr(a, "full_route", []) or [], graph) for a in agents)

def makespan_frames(agents) -> int:
    return max(int(getattr(a, "finished_frame", 0)) for a in agents) if agents else 0

def makespan_seconds(agents, fps: int) -> float:
    mf = makespan_frames(agents)
    return mf / float(fps) if fps > 0 else float(mf)

def success_rate(agents) -> float:
    return 1.0 if agents and all(getattr(a, "finished", False) for a in agents) else 0.0

def total_wait_time_frames(agents) -> int:
    return sum(int(getattr(a, "wait_frames", 0)) for a in agents)

def per_agent_stats(agents, graph: nx.Graph) -> List[dict]:
    rows = []
    for a in agents:
        init_len = float(getattr(a, "initial_len", 0.0))
        final_len = path_length(getattr(a, "full_route", []) or [], graph)
        rows.append({
            "agent": a.name,
            "finished": bool(getattr(a, "finished", False)),
            "replans": int(getattr(a, "replans", 0)),
            "gates": int(getattr(a, "gates", 0)),
            "wait_frames": int(getattr(a, "wait_frames", 0)),
            "finished_frame": int(getattr(a, "finished_frame", 0)),
            "initial_len": round(init_len, 3),
            "final_len": round(final_len, 3),
            "extra_len": round(final_len - init_len, 3),
        })
    return rows

def print_eval_summary(agents, graph: nx.Graph, fps: int = 10):
    print("\n==== Evaluation Summary ====")
    sr = success_rate(agents)
    ms_f = makespan_frames(agents)
    ms_s = makespan_seconds(agents, fps)
    soc = sum_of_costs(agents, graph)
    wt = total_wait_time_frames(agents)

    print(f"Success Rate      : {sr*100:.1f}%")
    print(f"Makespan          : {ms_f} frames  (~{ms_s:.2f} s @ {fps} fps)")
    print(f"Sum of Costs      : {soc:.3f} (graph-weighted)")
    print(f"Total Wait Time   : {wt} frames")

    print("\nPer-Agent:")
    for r in per_agent_stats(agents, graph):
        print(f"  - {r['agent']}: finished={r['finished']}  replans={r['replans']}  gates={r['gates']}  "
              f"wait={r['wait_frames']}f  finish@{r['finished_frame']}  "
              f"len(initial→final)={r['initial_len']}→{r['final_len']}  (+{r['extra_len']})")
