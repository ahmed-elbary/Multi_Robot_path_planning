# src/main.py
import argparse
from planner.agent import Agent
from planner.planner_core import find_routes, find_critical_points, split_critical_paths, assign_waiting_agents, replan_waiting_agents

from planner.visualiser import animate_paths
from utils import load_map, build_graph_from_yaml, get_occupied_nodes, generate_filtered_map


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Robot Fragment Planner")
    parser.add_argument('--map', type=str, default='data/map.yaml', help='Path to YAML map file')
    parser.add_argument('--save', action='store_true', help='Save animation as video')
    return parser.parse_args()



def create_agents() -> list:
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
    for agent in agents:
        route = agent.route
        if not route or not agent.active:
            continue

        fragments = []
        fragment = []
        for i, node in enumerate(route):
            fragment.append(node)
            if node in criticals:
                if not any(a.name != agent.name and node in a.route and a.priority() < agent.priority() for a in agents):
                    continue
                else:
                    agent.waiting = True
                    if i > 0:
                        agent.wait_node = route[i - 1]
                    fragments.append(fragment[:-1])
                    fragment = [node]
        if fragment:
            fragments.append(fragment)
        agent.fragments = fragments
        agent.route = fragments[0] if fragments else []

    print("[✓] Resolving waiting agents...")
    any_waiting = True
    while any_waiting:
        before = {a.name for a in agents if a.waiting}
        assign_waiting_agents(agents)
        after = {a.name for a in agents if a.waiting}
        new_waiters = after - before
        any_waiting = bool(new_waiters)

    print("==== Final Agent Routes ====")
    for agent in agents:
        status = "Active" if agent.goal else "Inactive (No goal)"
        print(f"- {agent.name}: route={agent.route} | full_route={agent.full_route} | fragments={agent.fragments} | {status}")

    print("[✓] Launching animation...")
    positions = {node: tuple(data['pos']) for node, data in graph.nodes(data=True)}

    def get_orientation_from_map(node_name):
        for entry in map_data:
            if entry['node']['name'] == node_name:
                q = entry['node']['pose']['orientation']
                z, w = q['z'], q['w']
                return math.atan2(2.0 * w * z, 1.0 - 2.0 * (z ** 2))
        return None

    # Pass graph to visualiser for dynamic replanning during animation
    replan_waiting_agents(agents, graph, frame=0)


    animate_paths(agents, positions, map_data, get_orientation_from_map, show_legend=True, graph=graph, save=args.save)


if __name__ == "__main__":
    import math
    main()
