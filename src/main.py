import argparse
import math
from planner.agent import Agent
from planner.planner_core import find_routes
from planner.visualiser import animate_paths
from utils import load_map, build_graph_from_yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Robot Reservation-Based Path Planner")
    parser.add_argument('--map', type=str, default='data/map.yaml', help='Path to YAML map file')
    parser.add_argument('--save', action='store_true', help='Save animation to file')
    return parser.parse_args()

def create_agents() -> list:
    return [
        Agent("Robot1", "Start1", "Spare1"),
        Agent("Robot2", "T02", "Park2"),
        # Agent("Robot3", "Park3", "T00"),
        # Agent("Robot4", "Spare2", "T51"),
    ]

def main():
    args = parse_args()
    print("[✓] Loading map...")
    map_data = load_map(args.map)
    graph = build_graph_from_yaml(map_data)

    print("[✓] Creating agents...")
    agents = create_agents()

    print("[✓] Planning all routes sequentially using reservation-based time-aware planner...")

    # --- Sequential Planning: Shared reservation tables ---
    node_reservations = {}
    edge_reservations = {}

    for agent in agents:
        find_routes([agent], graph, reservation_table=node_reservations, edge_reservations_arg=edge_reservations)

    print("\n==== Final Agent Routes and Timings ====")
    for agent in agents:
        status = "Active" if agent.goal else "Inactive (No goal)"
        print(f"\nAgent: {agent.name} ({status})")
        if not getattr(agent, "route", None):
            print("  No route assigned.")
            continue
        for u, v, depart, arrive in agent.route:
            print(f"  {u} → {v}: depart at {depart:.2f}, arrive at {arrive:.2f}")
        print(f"  Final full route: {agent.full_route}")
        print(f"  Final arrival time: {agent.arrival_time:.2f}")

    print("[✓] Launching animation...")
    positions = {node: tuple(data['pos']) for node, data in graph.nodes(data=True)}

    def get_orientation_from_map(node_name):
        for entry in map_data:
            if entry['node']['name'] == node_name:
                q = entry['node']['pose']['orientation']
                z, w = q['z'], q['w']
                return math.atan2(2.0 * w * z, 1.0 - 2.0 * (z ** 2))
        return None

    animate_paths(
        agents=agents,
        positions=positions,
        topo_map=map_data,
        get_orientation_from_map=get_orientation_from_map,
        show_legend=True,
        graph=graph,
        save=args.save
    )

if __name__ == "__main__":
    main()
