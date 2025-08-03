# src/main.py
import argparse
from planner.agent import Agent
from planner.planner_core import find_routes, find_critical_points, split_critical_paths
from planner.visualiser import animate_paths
from utils import load_map, build_graph_from_yaml, get_occupied_nodes, generate_filtered_map


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Robot Fragment Planner")
    parser.add_argument('--map', type=str, default='data/map.yaml', help='Path to YAML map file')
    return parser.parse_args()


def create_agents() -> list:
    # Replace this with actual agent configuration or loading logic
    return [
        Agent("Robot1", "Park1", "T01"),
        Agent("Robot2", "Park2", "T52"),
        Agent("Robot3", "Park3", "T00"),
        Agent("Robot4", "Spare2", "T51"),

    ]


def main():
    args = parse_args()
    print("[✓] Loading map...")
    map_data = load_map(args.map)
    graph = build_graph_from_yaml(map_data)

    print("[✓] Creating agents...")
    agents = create_agents()

    print("[✓] Finding initial routes...")
    find_routes(agents, graph)

    print("[✓] Identifying occupied nodes...")
    occupied = get_occupied_nodes(agents, graph)

    print("[✓] Filtering map and replanning if needed...")
    for agent in agents:
        if agent.active:
            filtered = generate_filtered_map(graph, occupied, agent.start, agent.goal)
            find_routes([agent], filtered)

    print("[✓] Detecting critical points...")
    criticals = find_critical_points(agents)

    print("[✓] Splitting paths at critical points...")
    split_critical_paths(agents, list(criticals.keys()))

    print("==== Final Agent Routes ====")
    for agent in agents:
        status = "Active" if agent.goal else "Inactive (No goal)"
        print(f"- {agent.name}: full_route={agent.full_route} | {status}")

    print("\n[Shared Critical Points]")
    for point, robots in criticals.items():
        print(f"- {point}: used by {', '.join(sorted(robots))}")

    print("\n[Detected Wait Conditions]")
    for agent in agents:
        if agent.waiting and agent.wait_node:
            print(f"- {agent.name} will wait at '{agent.wait_node}' due to lower priority")

    print("\n[Fragmented Segments]")
    for agent in agents:
        if agent.fragments:
            print(f"- {agent.name}: fragments = {agent.fragments}")

    print("[✓] Launching animation...")
    positions = {node: tuple(data['pos']) for node, data in graph.nodes(data=True)}

    def get_orientation_from_map(node_name):
        for entry in map_data:
            if entry['node']['name'] == node_name:
                q = entry['node']['pose']['orientation']
                z, w = q['z'], q['w']
                return math.atan2(2.0 * w * z, 1.0 - 2.0 * (z ** 2))
        return None

    animate_paths(agents, positions, map_data, get_orientation_from_map, show_legend=True)


if __name__ == "__main__":
    import math
    main()
