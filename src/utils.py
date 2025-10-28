# utils.py
import yaml
import networkx as nx
import math
from typing import Dict, Tuple, List

# utils.py
import yaml
import networkx as nx
import math

def load_map(yaml_path: str):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def _iter_node_entries(data):
    """Normalize map to an iterable of node entries."""
    if isinstance(data, dict) and "nodes" in data:
        return data["nodes"]                      # big-map format
    if isinstance(data, list):
        return data                               # small-map format
    raise ValueError("Unsupported map schema: expected list or dict with 'nodes'.")

def _get_node_dict(entry):
    """Some files store fields under entry['node'], others directly at entry-level."""
    return entry.get("node", entry)

def build_graph_from_yaml(data) -> nx.DiGraph:
    G = nx.DiGraph()
    entries = _iter_node_entries(data)

    # 1) add nodes with positions
    for entry in entries:
        node = _get_node_dict(entry)
        name = node["name"]
        pos = node.get("pose", {}).get("position", {})
        x = float(pos.get("x", 0.0))
        y = float(pos.get("y", 0.0))
        # keep original entry for things like orientation lookup later
        G.add_node(name, pos=(x, y), raw_entry=entry)

    # 2) add directed edges with Euclidean weights
    entries = _iter_node_entries(data)  # re-iterate
    for entry in entries:
        node = _get_node_dict(entry)
        u = node["name"]
        ux, uy = G.nodes[u]["pos"]
        for edge in node.get("edges", []) or []:
            v = edge.get("node")
            if not v or v not in G:
                continue
            vx, vy = G.nodes[v]["pos"]
            w = math.hypot(vx - ux, vy - uy)
            G.add_edge(u, v, weight=w, edge_id=edge.get("edge_id"))
    return G



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

