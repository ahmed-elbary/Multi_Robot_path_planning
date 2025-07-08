import networkx as nx
import math

def build_graph_from_yaml(top_map):
    """
    Converts a YAML-formatted topological map into a NetworkX graph.
    Each node contains its position as attributes. Each edge has Euclidean distance as weight.
    """
    G = nx.DiGraph()
    positions = {}

    for item in top_map:
        node = item["node"]
        name = node["name"]
        pos = node["pose"]["position"]
        x, y = pos["x"], pos["y"]

        # Add node with position attribute
        G.add_node(name, pos=(x, y))
        positions[name] = (x, y)

        for edge in node.get("edges", []):
            neighbor = edge["node"]

            # Prevent duplicate edges
            if not G.has_edge(name, neighbor):
                neighbor_pos = find_node_position(top_map, neighbor)
                distance = math.hypot(x - neighbor_pos[0], y - neighbor_pos[1])
                G.add_edge(name, neighbor, weight=distance)

    return G, positions

def find_node_position(top_map, node_name):
    """
    Finds the (x, y) position of a given node name from the topological map.
    """
    for item in top_map:
        node = item["node"]
        if node["name"] == node_name:
            pos = node["pose"]["position"]
            return (pos["x"], pos["y"])
    raise ValueError(f"Node '{node_name}' not found in YAML.")

def heuristic(u, v, positions):
    """
    Heuristic function for A* — Euclidean distance between nodes.
    """
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    return math.hypot(x2 - x1, y2 - y1)

def find_path(graph, start, goal, positions):
    """
    Finds a path between 'start' and 'goal' using A* on the graph.
    """
    try:
        path = nx.astar_path(graph, start, goal,
                             heuristic=lambda u, v: heuristic(u, v, positions),
                             weight='weight')
        return path
    except nx.NetworkXNoPath:
        print(f"No path found between {start} and {goal}.")
        return []
