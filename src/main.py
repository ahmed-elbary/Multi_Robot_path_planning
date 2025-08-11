import argparse
import math
from planner.agent import Agent
from planner.fragment_planner import (
    find_routes,
    find_critical_points,
    split_critical_paths,
    assign_waiting_agents,
    replan_waiting_agents,
)
from utils import load_map, build_graph_from_yaml
### NEW
from planner.visualiser import animate_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Robot Fragment-Based Path Planner")
    parser.add_argument('--map', type=str, default='data/map.yaml', help='Path to YAML map file')
    parser.add_argument('--max_loops', type=int, default=5, help='Max replan loops after a finisher event')
    ### NEW
    parser.add_argument('--animate', action='store_true', help='Show matplotlib animation')
    parser.add_argument('--fps', type=int, default=10, help='Animation FPS')
    parser.add_argument('--save', action='store_true', help='Save animation to file (uses default name)')
    return parser.parse_args()


def create_agents() -> list:
    return [
        Agent("Robot1", "Start1", "Spare1"),
        Agent("Robot2", "T02", "Park2"),
        Agent("Robot3", "Park3", "T00"),
        Agent("Robot4", "Spare2", "T51"),
    ]


def pretty_frag(frag):
    return " -> ".join([frag[0][0]] + [v for (_, v) in frag]) if frag else ""


def print_summary(agents, title="\n==== Agent Routes (fragments) ===="):
    print(title)
    for agent in agents:
        status = []
        status.append("waiting" if agent.waiting else "active")
        if agent.finished:
            status.append("finished")
        if agent.replanned:
            status.append("replanned")

        print(f"\nAgent: {agent.name} [{', '.join(status)}]")
        if not agent.fragments:
            print("  No route assigned.")
        else:
            for frag in agent.fragments:
                if frag:
                    print(f"  Fragment: {pretty_frag(frag)}")
        if getattr(agent, "full_route", None):
            print(f"  Full route: {agent.full_route}")


def main():
    args = parse_args()
    print("[✓] Loading map...")
    map_data = load_map(args.map)
    graph = build_graph_from_yaml(map_data)

    print("[✓] Creating agents...")
    agents = create_agents()

    # ===== 1) Initial space-aware planning on filtered maps =====
    print("[✓] Space-aware planning on filtered maps...")
    find_routes(agents, graph)

    critical = find_critical_points(agents)
    dangerous_points = list(critical.keys())
    if dangerous_points:
        print(f"[✓] Critical points found: {dangerous_points}")
        split_critical_paths(graph, agents, dangerous_points)
        assign_waiting_agents(agents)

    print_summary(agents, "\n==== Initial Plan (with fragments) ====")

    ### NEW — optional live animation for debugging
    if args.animate:
        # positions: node -> (x,y) from the graph
        positions = {node: tuple(data['pos']) for node, data in graph.nodes(data=True)}

        # orientation helper (quaternion z,w to yaw) from YAML
        def get_orientation_from_map(node_name):
            for entry in map_data:
                if entry['node']['name'] == node_name:
                    q = entry['node']['pose']['orientation']
                    z, w = q['z'], q['w']
                    return math.atan2(2.0 * w * z, 1.0 - 2.0 * (z ** 2))
            return None

        print("[✓] Launching animation...")
        animate_paths(
            agents=agents,
            positions=positions,
            topo_map=map_data,
            get_orientation_from_map=get_orientation_from_map,
            fps=args.fps,
            show_legend=True,
            graph=graph,     # enables in-animation replanning for waiting agents
            save=args.save
        )
        # After animation returns, also dump a summary of whatever changed
        print_summary(agents, "\n==== After Animation ====")
        return  # skip the terminal-only loop when animating

    # ===== 2) Terminal-only “event -> replan” loop (kept for non-animated runs) =====
    robot1 = next((a for a in agents if a.name == "Robot1"), None)

    if robot1 and robot1.full_route:
        robot1.current_node = robot1.full_route[-1]
        robot1.finished = True
        print(f"\n[✓] Simulating {robot1.name} has reached goal at {robot1.current_node}")

    loop = 0
    while any(a.waiting for a in agents) and loop < args.max_loops:
        loop += 1
        print(f"\n[→] Replan loop {loop}: attempting to resume all waiting agents...")
        resumed = replan_waiting_agents(agents, graph)
        if resumed:
            print("[✓] One or more waiting agents successfully replanned.")
            print_summary(agents, "\n==== After Replan ====")
        else:
            print("[x] No waiting agent could replan this loop (still blocked).")
            break

    print_summary(agents, "\n==== Final Summary ====")


if __name__ == "__main__":
    main()
