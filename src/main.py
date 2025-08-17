import argparse
import math
from planner.agent import Agent
from planner.fragment_planner import (
    find_routes,
    find_critical_points,
    split_critical_paths,
    replan_waiting_agents,
)
from utils import load_map, build_graph_from_yaml
from planner.visualiser import animate_paths
from planner.metrics import collect_metrics, print_metrics
from utils import print_eval_summary





def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Robot Fragment-Based Path Planner")
    parser.add_argument('--map', type=str, default='data/map.yaml', help='Path to YAML map file')
    parser.add_argument('--max_loops', type=int, default=5, help='Max replan loops per cascade (terminal mode)')
    parser.add_argument('--animate', action='store_true', help='Show matplotlib animation')
    parser.add_argument('--fps', type=int, default=10, help='Animation FPS')
    parser.add_argument('--save', action='store_true', help='Save animation to file (uses default name)')
    parser.add_argument('--quiet', action='store_true', help='Reduce replanning logs/snapshots')
    return parser.parse_args()


# ====================== choose a test set ======================
def create_agents() -> list:
    return [
        Agent("Robot1", "Park1", "T11"),
        Agent("Robot2", "Spare2", "Park2"),
        Agent("Robot3", "Park3", "T33"),
        Agent("Robot4", "T41", "T52"),
        Agent("Robot5", "T53", "T40"),
       
    ]


# def create_agents() -> list:
#     return [
#         Agent("Robot1", "Park1", "D2"),
#         Agent("Robot2", "Start2", "Start2"),
#         Agent("Robot3", "C1", "C1"),
#         Agent("Robot4", "A1", "Park1"),
#         Agent("Robot5", "Park5", "Park5"),
#     ]

# def create_agents() -> list:
#     return [
#         Agent("Robot1", "Park1", "W"),
#         Agent("Robot2", "Park2", "S"),
#         Agent("Robot3", "Park3", "N"),
#         Agent("Robot4", "Park4", "E"),
#         Agent("Robot5", "Park5", "NE"),
#     ]
# ===============================================================


def pretty_frag(frag):
    return " -> ".join([frag[0][0]] + [v for (_, v) in frag]) if frag else ""


def print_summary(agents, title="\n==== Agent Routes (fragments) ===="):
    print(title)
    for agent in agents:
        status = []
        status.append("waiting" if agent.waiting else "active")
        if getattr(agent, "finished", False):
            status.append("finished")
        if getattr(agent, "replanned", False):
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

    # ===== 1) Load map & build graph =====
    print("[✓] Loading map...")
    map_data = load_map(args.map)
    graph = build_graph_from_yaml(map_data)

    # ===== 2) Create agents =====
    print("[✓] Creating agents...")
    agents = create_agents()

    # ===== 3) Initial space-aware planning on filtered maps =====
    print("[✓] Space-aware planning on filtered maps...")
    find_routes(agents, graph)

    critical = find_critical_points(agents)
    dangerous_points = list(critical.keys())
    if dangerous_points:
        print(f"\n[✓] Critical points found: {dangerous_points}\n")
        # Computes initial safe prefixes & waiting states
        split_critical_paths(graph, agents, dangerous_points)

    print_summary(agents, "\n==== Initial Plan (with fragments) ====")

    # ===== 4) Animation mode =====
    if args.animate:
        positions = {node: tuple(data['pos']) for node, data in graph.nodes(data=True)}

        def get_orientation_from_map(node_name):
            # Pull yaw from the YAML map (z,w quaternion → yaw)
            for entry in map_data:
                if entry['node']['name'] == node_name:
                    q = entry['node']['pose']['orientation']
                    z, w = q['z'], q['w']
                    return math.atan2(2.0 * w * z, 1.0 - 2.0 * (z ** 2))
            return None

        print("\n[✓] Launching animation...")
        animate_paths(
            agents=agents,
            positions=positions,
            topo_map=map_data,
            get_orientation_from_map=get_orientation_from_map,
            fps=args.fps,
            show_legend=True,
            graph=graph,
            save=args.save
            # (Visualizer is expected to call replan_waiting_agents internally per frame.)
        )

        print_summary(agents, "\n==== After Animation ====")
        print_eval_summary(agents, graph, fps=args.fps)

        return

    # ===== 5) Terminal mode (no --animate) =====
    # Simple cascade runner that simulates finishers so dependent waiters can release.
    def run_cascade(finishers: set[str], label: str):
        """
        Run up to max_loops of replan calls, releasing waiters if their blockers are in `finishers`
        (or if they can find an alternate path around any active owner). Winners are simulated
        to their goals and added to `finishers` to allow chained releases within the cascade.
        """
        print(f"\n--- Replan cascade after {label} ---\n")
        finished_owners = set(finishers)
        loop = 0
        while loop < args.max_loops:
            if not any(a.waiting for a in agents):
                break
            loop += 1
            print(f"[→] Replan loop {loop}: attempting to resume all waiting agents...")

            resumed = replan_waiting_agents(
                agents,
                graph,
                verbose_snapshots=not args.quiet
            )

            winners = [a for a in agents if getattr(a, "replanned", False) and not a.waiting]
            advanced_only = [a for a in agents if getattr(a, "replanned", False) and a.waiting]

            if not resumed:
                print("[x] No new agents resumed.")
                break

            if not winners and advanced_only:
                # Clear transient "replanned" flags on advanced-but-still-gated agents
                for a in advanced_only:
                    a.replanned = False
                print("[i] All newly-advanced agents are still gated; waiting for a blocker to finish.")
                break

            # Simulate winners reaching their goals immediately (terminal demo convenience)
            for w in winners:
                w.current_node = w.goal
                w.finished = True
                w.replanned = False
                finished_owners.add(w.name)
                print(f"[✓] Simulating {w.name} reached goal at {w.goal}")

            print_summary(agents, "\n==== After Replan ====")

            if all(getattr(a, "finished", False) for a in agents):
                break

    # Demo trigger: simulate Robot1 finishing first, then cascade.
    robot1 = next((a for a in agents if a.name == "Robot1"), None)
    if robot1 and robot1.full_route:
        robot1.current_node = robot1.goal
        robot1.finished = True
        print(f"\n[✓] Simulating {robot1.name} has reached goal at {robot1.current_node}")

    run_cascade({robot1.name} if robot1 else set(), label="Robot1")

    # If anyone is still waiting, identify an unfinished blocker and finish it next, then cascade again.
    safety = 0
    while any(a.waiting for a in agents) and safety < len(agents) * 3:
        safety += 1
        waiter_blockers = {}
        for a in agents:
            if getattr(a, "waiting", False) and getattr(a, "blocker_owner", None):
                waiter_blockers[a.blocker_owner] = waiter_blockers.get(a.blocker_owner, 0) + 1

        candidates = [a for a in agents if a.name in waiter_blockers and not getattr(a, "finished", False)]
        if not candidates:
            print("[i] No unfinished blockers identified; stopping.")
            break

        # Prefer the one blocking the most waiters; tie-break by agent.priority()
        owner = min(candidates, key=lambda o: (-waiter_blockers[o.name], o.priority()))
        owner.current_node = owner.goal
        owner.finished = True
        print(f"\n[✓] Simulating {owner.name} has reached goal at {owner.goal}")

        run_cascade({owner.name}, label=owner.name)

    print_summary(agents, "\n==== Final Summary ====")
    print_eval_summary(agents, graph, fps=args.fps)






if __name__ == "__main__":
    main()
