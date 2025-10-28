# src/main_st.py
from __future__ import annotations

import argparse
import logging
import math
from typing import List, Dict, Tuple

import networkx as nx

from utils import load_map, build_graph_from_yaml
from metrics import collect_metrics, print_metrics, append_run_to_csv

from space_time_planner.planner_st import SpaceTimePlanner
from space_time_planner.agent_st import Agent


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("st.main")


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    # basicConfig is a no-op if logging is already configured elsewhere
    logging.basicConfig(level=level, format="%(message)s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    p = argparse.ArgumentParser(description="Space–Time Multi-Robot Planner (terminal-first)")
    p.add_argument("--map", type=str, default="data/map2.yaml", help="Path to YAML map file")
    p.add_argument("--max_loops", type=int, default=20, help="Max replan loops in terminal mode")
    p.add_argument("--fps", type=int, default=10, help="Logical FPS for timing")
    p.add_argument("--animate", action="store_true", help="Show animation (optional)")
    p.add_argument("--save", action="store_true", help="Save animation to file (optional)")
    p.add_argument("--debug", action="store_true", help="Verbose planner logs (ETAs, releases, gate holds)")
    p.add_argument('--dump_csv', type=str, default=None, help='Append metrics (global + per-agent) to this CSV after the run')

    return p.parse_args()


# ---------------------------------------------------------------------------
# create agents and their starting points and goals 
# ---------------------------------------------------------------------------

## For map.yaml
def create_agents() -> List[Agent]:
    """
    Define initial agents and goals. Adjust to force/avoid conflicts.
    For real experiments, move scenarios into a separate module or YAML.
    """
    return [
        Agent("Robot1", "Park1", "T31"),
        Agent("Robot2", "Park2", "T53"),
        Agent("Robot3", "T53", "T40"),
        # Agent("Robot4", "T42", "T01"),
        # Agent("Robot5", "T01", "T42"),

    ]

# For map_bigger.yaml
# def create_agents() -> List[Agent]:
#     """
#     Define initial agents and goals. Adjust to force/avoid conflicts.
#     For real experiments, move scenarios into a separate module or YAML.
#     """
#     return [
        # Agent("Robot1", "Park1", "T27"),
        # Agent("Robot2", "Park2", "T36"),
        # Agent("Robot3", "Park3", "T61"),
        # Agent("Robot4", "Park4", "T51"),
        # Agent("Robot5", "Park5", "T74"),
        # Agent("Robot6", "Park6", "T79"),

        # Agent("Robot1", "T35", "T27"),
        # Agent("Robot2", "Park2", "T36"),
        # Agent("Robot3", "T78", "T61"),
        # Agent("Robot4", "T42", "T51"),
        # Agent("Robot5", "T62", "T74"),
        # Agent("Robot6", "Park6", "T65"),
    # ]

## For map2.yaml
# def create_agents() -> list[Agent]:
#     return [
#         Agent("Robot1", "dock-0", "r5.7-ca"),
#         Agent("Robot2", "dock-1", "r6.5-c0"),
#         Agent("Robot3", "r7.5-c3", "r6.5-c3"),
#         Agent("Robot4", "r3.5-c0", "r5.7-c1"),
#         Agent("Robot5", "r10.3-ca", "r1.5-c1"),
#         Agent("Robot6", "r8.5-c1", "r9.5-c0"),
#         Agent("Robot7", "r1.5-c3", "r3.5-c3"),
#     ]

## For map3.yaml
# def create_agents() -> list[Agent]:
#     return [
#         Agent("Robot1", "Park1", "A7"),
#         Agent("Robot2", "Park2", "B2"),
#         Agent("Robot3", "Park3", "C1"),

#     ]


# ---------------------------------------------------------------------------
# Terminal print helpers 
# ---------------------------------------------------------------------------

def _edge_nodes_from_frag(frag) -> List[str]:
    """Normalize a 'fragment' to a list of node names."""
    if not frag:
        return []
    first = frag[0]
    if isinstance(first, str):
        return list(frag)
    if isinstance(first, tuple) and len(first) >= 2:
        nodes = [first[0]]
        nodes.extend(step[1] for step in frag if isinstance(step, tuple) and len(step) >= 2)
        return nodes
    return list(frag)


def pretty_nodes(seq) -> str:
    nodes = _edge_nodes_from_frag(seq)
    return " -> ".join(nodes) if nodes else ""


def print_spatial_plans(agents: List[Agent], title: str = "\n==== Initial spatial plans ====") -> None:
    print(title)
    for a in agents:
        path = getattr(a, "planned_path", None) or getattr(a, "full_route", None) or []
        print(f"- {a.name}: {path}")


def print_status(agents: List[Agent], title: str = "\n==== Status ====") -> None:
    print(title)
    for a in agents:
        flags = ["waiting" if getattr(a, "waiting", False) else "active"]
        if getattr(a, "finished", False):
            flags.append("finished")
        print(f"- {a.name} [{', '.join(flags)}]  at {getattr(a, 'wait_node', a.start)}  → {a.goal}")


def print_all_timed(agents: List[Agent], title: str = "\n==== Timed routes (u->v @ [depart, arrive)) ====") -> None:
    print(title)
    for a in agents:
        r = getattr(a, "route", []) or []
        if not r:
            print(f"- {a.name}: <no timed edges yet>")
            continue
        print(f"- {a.name}:")
        for (u, v, td, ta) in r:
            print(f"    {u:>8} -> {v:<8}  [{td:6.2f}, {ta:6.2f})")


def print_global_timeline(agents: List[Agent], fps: int = 10,
                          title: str = "\n==== Global second-by-second timeline ====") -> None:
    """
    One shared clock for all agents. For each second [t,t+1), show each agent's action.
    Useful to synchronization with animation.
    """
    # Find the last arrival among all agents
    max_end = 0.0
    for a in agents:
        for _, _, _, ta in (a.route or []):
            if ta > max_end:
                max_end = ta
    horizon = int(math.ceil(max_end))

    def state_at(a: Agent, t: float):
        """Return ('moving', u, v, progress, duration) or ('waiting', node)."""
        r = a.route or []
        if not r:
            return ("waiting", getattr(a, "start", None))
        # moving if inside any edge interval
        for (u, v, td, ta) in r:
            if td <= t < ta:
                return ("moving", u, v, t - td, ta - td)
        # otherwise waiting
        if t < r[0][2]:
            return ("waiting", r[0][0])
        last_node = r[0][0]
        for (u, v, td, ta) in r:
            if ta <= t:
                last_node = v
            else:
                break
        return ("waiting", last_node)

    if horizon == 0:
        print("\n(no timed movement yet — nothing to show)")
        return

    print(title)
    for t in range(horizon):
        print(f"[t={t:2d}–{t+1:d}s]")
        for a in agents:
            st = state_at(a, t)
            if st[0] == "moving":
                _, u, v, prog, dur = st
                print(f"  - {a.name:6}: moving  {u:>6} → {v:<6}  ( {prog:4.1f}s / {dur:4.1f}s )")
            else:
                _, node = st
                node = node if node is not None else "<unknown>"
                print(f"  - {a.name:6}: waiting at {node}")
        print("------------------------------------------")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_animation(args: argparse.Namespace,
                  agents: List[Agent],
                  positions: Dict[str, Tuple[float, float]],
                  topo: list,
                  graph: nx.DiGraph,
                  planner: SpaceTimePlanner) -> None:
    """Animate space–time planning from t=0 using the live planner."""
    from space_time_planner.visualiser_st import animate_paths  # lazy import to keep CLI snappy

    def get_orientation_from_map(node_name):
        entries = topo.get("nodes", []) if isinstance(topo, dict) else topo
        for entry in entries:
            node = entry.get("node", entry)
            if node.get("name") == node_name:
                q = node.get("pose", {}).get("orientation", {})
                z = float(q.get("z", 0.0)); w = float(q.get("w", 1.0))
                return math.atan2(2.0 * w * z, 1.0 - 2.0 * (z ** 2))
        return None


    logger.info("\n[✓] Launching time-aware animation...\n")
    animate_paths(
        agents=agents,
        positions=positions,
        topo_map=topo,
        get_orientation_from_map=get_orientation_from_map,
        fps=args.fps,
        show_legend=True,
        graph=graph,
        save=args.save,
        planner=planner,
        live_timeline=False,  # keep the second-by-second clock visible during animation
    )

    print_all_timed(agents, "\n==== Timed routes after animation ====")
    report = collect_metrics(agents, graph, fps=args.fps)
    print_metrics(report, fps=args.fps)


    if args.dump_csv:
        meta = {
            "map_id": args.map,
            "planner": "space_time",
            "fps": args.fps,
            "gap_sec": planner.res.gap,
            "hold_sec": planner.res.hold,
            "n_agents": len(agents),
        }
        out_path = append_run_to_csv(report, args.dump_csv, meta, agents)
        print(f"[✓] Appended metrics to {out_path}")

    return



def run_terminal_cascade(args: argparse.Namespace,
                         agents: List[Agent],
                         graph: nx.DiGraph,
                         planner: SpaceTimePlanner) -> None:
    """
    Terminal (non-animated) loop:
    - each iteration asks the planner to release one or more edges,
      respecting reservations and gaps;
    - we advance the logical frame counter accordingly.
    """
    logger.info("\n[✓] Running terminal cascade (space–time replan)...")
    frame, safety = 0, 0

    while safety < max(5, args.max_loops):
        safety += 1
        prev_counts = {a.name: len(getattr(a, "route", []) or []) for a in agents}

        released = planner.replan_waiting_agents(agents, graph, frame=frame)
        tick_releases = sum(len((a.route or [])) - prev_counts[a.name] for a in agents)

        # Show tick header only in debug mode (planner prints its own detailed logs)
        if released and args.debug:
            logger.debug(f"\n--- Tick {safety} @ frame {frame}  (released {tick_releases} edge(s)) ---")

        # Advance logical time
        if released:
            frame += max(1, args.fps // 2)
        else:
            if all(getattr(a, "finished", False) for a in agents):
                break
            frame += 1

        if all(getattr(a, "finished", False) for a in agents):
            break

    # Final report
    print_all_timed(agents, "\n==== Final timed routes ====")
    print_global_timeline(agents, fps=args.fps)

    report = collect_metrics(agents, graph, fps=args.fps)
    print_metrics(report, fps=args.fps)

    if args.dump_csv:
        meta = {
            "map_id": args.map,
            "planner": "space_time",
            "seed": getattr(args, "seed", None),
            "fps": args.fps,
            "gap_sec": planner.res.gap,
            "hold_sec": planner.res.hold,
            "n_agents": len(agents),
        }
        out_path = append_run_to_csv(report, args.dump_csv, meta, agents)
        print(f"[✓] Appended metrics to {out_path}")
    return




# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    setup_logging(args.debug)

    # 1) Load map & graph
    logger.info("[✓] Loading map...")
    topo = load_map(args.map)
    graph: nx.DiGraph = build_graph_from_yaml(topo)
    positions = {n: tuple(d["pos"]) for n, d in graph.nodes(data=True)}

    # 2) Agents
    logger.info("[✓] Creating agents...")
    agents = create_agents()

    # 3) Planner
    st = SpaceTimePlanner(positions=positions, fps=args.fps, debug=args.debug)

    # 4) Initial spatial plan + waiter marking
    logger.info("[✓] Initial spatial planning & waiter marking...")
    st.initial_plan(agents, graph)
    print_spatial_plans(agents)
    print_status(agents, "\n==== After waiter marking ====")

    # 5) Run
    if args.animate:
        run_animation(args, agents, positions, topo, graph, st)
    else:
        run_terminal_cascade(args, agents, graph, st)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
