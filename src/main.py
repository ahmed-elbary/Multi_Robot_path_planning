import argparse
import math
from planner.agent import Agent
from planner.planner_core import (
    find_routes, find_critical_points, split_critical_paths,
    assign_waiting_agents, replan_waiting_agents
)
from planner.visualiser import animate_paths
from utils import load_map, build_graph_from_yaml, get_occupied_nodes, generate_filtered_map

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Robot Fragment Planner")
    parser.add_argument('--map', type=str, default='data/map.yaml', help='Path to YAML map file')
    parser.add_argument('--save', action='store_true', help='Save animation to file')
    return parser.parse_args()

def create_agents() -> list:
    return [
        Agent("Robot1", "Park1", "T01"),
        Agent("Robot2", "Park2", "T52"),
        Agent("Robot3", "Park3", "T00"),
        # Agent("Robot4", "Spare2", "T51"),
    ]

def main():
    reservation_table = {}
    edge_reservations = {}

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
    print("\n==== Occupied Nodes for Filtering ====")
    print(occupied)

    print("[✓] Filtering map and replanning if needed...")
    for i, agent in enumerate(agents):
        if agent.active:
            prev_agents = agents[:i]
            occupied = get_occupied_nodes(prev_agents, graph)
            print(f"[{agent.name}] Occupied nodes (from previous agents): {occupied}")
            filtered = generate_filtered_map(graph, occupied, agent.start, agent.goal)

            try:
                find_routes([agent], filtered)
                agent.original_path_length = len(agent.full_route)
            except Exception as e:
                print(f"[!] {agent.name} could not plan with filtered map. Scheduling wait and retrying with full map...")
                wait_time = 3.0
                agent.start_delay = agent.arrival_time + wait_time if hasattr(agent, "arrival_time") else wait_time
                find_routes([agent], graph)
                agent.original_path_length = len(agent.full_route)

    print("[✓] Detecting critical points...")
    criticals = find_critical_points(agents)

    print("[✓] Splitting paths at critical points...")
    split_critical_paths(graph, agents, list(criticals.keys()))

    print("[✓] Resolving waiting agents...")
    assign_waiting_agents(agents)

    print("\n==== Final Agent Routes and Timings ====")
    for agent in agents:
        status = "Active" if agent.goal else "Inactive (No goal)"
        print(f"\nAgent: {agent.name} ({status})")
        print(f"  Fragments: {agent.fragments}")
        if not agent.route:
            print("  No route assigned.")
            continue
        for u, v, depart, arrive in agent.route:
            print(f"  {u} → {v}: depart at {depart:.2f}, arrive at {arrive:.2f}")
        print(f"  Final full route: {agent.full_route}")
        print(f"  Final arrival time: {agent.arrival_time:.2f}")
        if agent.waiting:
            print(f"  Wait Node: {agent.wait_node}, Arrival Time: {agent.arrival_time:.2f}, Resume Time: {agent.resume_time:.2f}")

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
