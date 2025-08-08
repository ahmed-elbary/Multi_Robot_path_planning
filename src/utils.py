import yaml
import networkx as nx
import math
from typing import Dict, Tuple, List

def load_map(yaml_path: str) -> Dict:
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def build_graph_from_yaml(data: list) -> nx.Graph:
    graph = nx.DiGraph()
    for entry in data:
        node_data = entry['node']
        name = node_data['name']
        x = node_data['pose']['position']['x']
        y = node_data['pose']['position']['y']
        graph.add_node(name, pos=(x, y))
        for edge in node_data.get('edges', []):
            graph.add_edge(name, edge['node'])
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