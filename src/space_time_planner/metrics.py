# planner/metrics.py
import math
import networkx as nx
from typing import Dict, List, Tuple
from .agent_st import Agent

def _edge_w(graph: nx.Graph, u: str, v: str) -> float:
    # prefer stored weight; fall back to Euclidean from pos
    w = graph[u][v].get("weight")
    if w is not None:
        return float(w)
    (x1, y1), (x2, y2) = graph.nodes[u]["pos"], graph.nodes[v]["pos"]
    return math.hypot(x2 - x1, y2 - y1)

def path_length(nodes: List[str], graph: nx.Graph) -> float:
    if not nodes or len(nodes) < 2:
        return 0.0
    return sum(_edge_w(graph, nodes[i], nodes[i+1]) for i in range(len(nodes)-1))

def shortest_length(graph: nx.Graph, start: str, goal: str) -> float:
    # uses the 'weight' we set in utils
    return float(nx.shortest_path_length(graph, start, goal, weight="weight"))

def potential_edge_swaps(agents: List[Agent]) -> int:
    # count (u,v) in Ai and (v,u) in Aj across full routes
    edge_sets: Dict[str, set] = {}
    for a in agents:
        r = getattr(a, "full_route", None) or []
        edge_sets[a.name] = set(zip(r[:-1], r[1:])) if len(r) >= 2 else set()
    swaps = 0
    names = [a.name for a in agents]
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            Ei, Ej = edge_sets[names[i]], edge_sets[names[j]]
            for (u, v) in Ei:
                if (v, u) in Ej:
                    swaps += 1
    return swaps

def critical_nodes_count(agents: List[Agent]) -> int:
    claims: Dict[str, int] = {}
    for a in agents:
        r = getattr(a, "full_route", None) or []
        for n in r:
            claims[n] = claims.get(n, 0) + 1
    return sum(1 for n, c in claims.items() if c > 1)

def collect_metrics(agents: List[Agent], graph: nx.Graph) -> Dict:
    per_agent = {}
    lengths = []
    shorts  = []
    for a in agents:
        full = getattr(a, "full_route", None) or [a.start]
        L_actual = path_length(full, graph)
        L_short  = shortest_length(graph, a.start, a.goal)
        lengths.append(L_actual)
        shorts.append(L_short)
        per_agent[a.name] = {
            "finished": bool(getattr(a, "finished", False)),
            "actual_length": round(L_actual, 3),
            "shortest_length": round(L_short, 3),
            "detour_ratio": (L_actual / L_short) if L_short > 0 else float("inf"),
            "num_fragments": len(getattr(a, "fragments", []) or []),
            "replans": int(getattr(a, "replans", 0)),
            "gates": int(getattr(a, "gates", 0)),
            "wait_frames": int(getattr(a, "wait_frames", 0)),
        }
    report = {
        "success": all(per_agent[n]["finished"] for n in per_agent),
        "sum_of_costs": round(sum(lengths), 3),
        "makespan_distance": round(max(lengths) if lengths else 0.0, 3),
        "avg_detour_ratio": (sum((per_agent[n]["detour_ratio"] for n in per_agent)) / len(per_agent)) if per_agent else 0.0,
        "potential_edge_swaps": potential_edge_swaps(agents),
        "critical_nodes": critical_nodes_count(agents),
        "per_agent": per_agent,
    }
    return report

def print_metrics(report: Dict, fps: int | None = None):
    print("\n==== Metrics ====")
    print(f"Success: {report['success']}")
    print(f"Sum-of-Costs (distance): {report['sum_of_costs']}")
    print(f"Makespan (distance): {report['makespan_distance']}")
    if fps:
        print(f"~Makespan (frames@fps={fps} if speed=1 unit/s): approx {int(report['makespan_distance']*fps)}")
    print(f"Avg Detour Ratio: {report['avg_detour_ratio']:.3f}")
    print(f"Critical nodes (shared): {report['critical_nodes']}")
    print(f"Potential edge swaps: {report['potential_edge_swaps']}")
    for name, m in report["per_agent"].items():
        print(f"  - {name}: len={m['actual_length']} (shortest={m['shortest_length']}, detour={m['detour_ratio']:.3f}), "
              f"frags={m['num_fragments']}, replans={m['replans']}, gates={m['gates']}, wait_frames={m['wait_frames']}")
