# metrics.py  (agent-agnostic)
import math, csv, os, uuid, datetime
import networkx as nx
from typing import Dict, List, Tuple, Protocol, runtime_checkable, Any, Optional


@runtime_checkable
class SupportsAgent(Protocol):
    name: str
    start: str
    goal: str
    # Path-ish attributes used by different planners
    full_route: List[str] | None
    fragments: List[List[str]] | None
    finished: bool

    # optional counters (present in both planners / visualisers)
    replans: int
    gates: int
    wait_frames: int

# -----------------------------
# Graph helpers
# -----------------------------
def _edge_w(graph: nx.Graph, u: str, v: str) -> float:
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
    return float(nx.shortest_path_length(graph, start, goal, weight="weight"))

# -----------------------------
# Extraction helpers (planner-agnostic)
# -----------------------------
def _extract_timed_edges(agent: SupportsAgent) -> List[Tuple[str, str, float, float]]:
    """
    Return [(u,v,td,ta), ...] if agent.route is time-annotated; else [].
    """
    r = getattr(agent, "route", None)
    if not r:
        return []
    timed = []
    for e in r:
        if (
            isinstance(e, tuple)
            and len(e) >= 4
            and isinstance(e[0], str)
            and isinstance(e[1], str)
            and isinstance(e[2], (int, float))
            and isinstance(e[3], (int, float))
        ):
            u, v, td, ta = e[:4]
            timed.append((u, v, float(td), float(ta)))
    return timed

def _route_nodes_from_timed(timed: List[Tuple[str, str, float, float]]) -> List[str]:
    if not timed:
        return []
    nodes = [timed[0][0]]
    nodes.extend(v for (_, v, _, _) in timed)
    return nodes

def _extract_executed_nodes(agent: SupportsAgent) -> List[str]:
    """
    Best-effort executed node sequence:
    - Prefer nodes derived from timed edges
    - Else fall back to full_route (initial plan or updated)
    - Else just [start]
    """
    timed = _extract_timed_edges(agent)
    if timed:
        return _route_nodes_from_timed(timed)
    full = getattr(agent, "full_route", None)
    if full:
        return list(full)
    return [getattr(agent, "start", "<unknown>")]

def _initial_path_nodes(agent: SupportsAgent) -> List[str]:
    """
    Initial spatial path nodes:
    - Prefer 'planned_path' if available (most explicit)
    - Else 'full_route' (often set to initial plan)
    - Else [start, goal] if distinct, or [start]
    """
    planned = getattr(agent, "planned_path", None)
    if planned:
        return list(planned)
    full = getattr(agent, "full_route", None)
    if full:
        return list(full)
    s = getattr(agent, "start", None)
    g = getattr(agent, "goal", None)
    if s and g and s != g:
        return [s, g]
    return [s] if s else []

# -----------------------------
# Higher-level metrics
# -----------------------------
def actual_route_length(agent: SupportsAgent, graph: nx.Graph) -> float:
    """
    True movement distance/time (sum of edge weights actually traversed).
    If timed edges exist, sum (ta - td). Else, sum weights along executed nodes.
    """
    timed = _extract_timed_edges(agent)
    if timed:
        return sum(max(0.0, ta - td) for (_, _, td, ta) in timed)
    nodes = _extract_executed_nodes(agent)
    return path_length(nodes, graph)

def initial_path_length(agent: SupportsAgent, graph: nx.Graph) -> float:
    nodes = _initial_path_nodes(agent)
    if nodes and len(nodes) >= 2:
        return path_length(nodes, graph)
    # Fallback to on-the-fly shortest (should rarely be needed)
    s, g = getattr(agent, "start", None), getattr(agent, "goal", None)
    if s and g:
        try:
            return shortest_length(graph, s, g)
        except Exception:
            return 0.0
    return 0.0

def waiting_time_seconds(agent: SupportsAgent) -> float:
    """
    Sum of stationary time:
    - Pre-first depart: td0 - 0
    - Between edges: max(0, td_k - ta_{k-1})
    Requires timed edges; else returns 0.0 (or rely on wait_frames in printing).
    """
    timed = _extract_timed_edges(agent)
    if not timed:
        return 0.0
    wait = 0.0
    # pre-first depart
    first_td = timed[0][2]
    wait += max(0.0, first_td - 0.0)
    # gaps
    for i in range(1, len(timed)):
        prev_ta = timed[i-1][3]
        this_td  = timed[i][2]
        if this_td > prev_ta:
            wait += (this_td - prev_ta)
    return wait

def finish_time_seconds(agent: SupportsAgent) -> Optional[float]:
    """
    Last arrival time ta (if timed edges exist). If start==goal and no edges, returns 0.0.
    Else None if unknown.
    """
    timed = _extract_timed_edges(agent)
    if timed:
        return float(timed[-1][3])
    # Handle trivial success where no movement is needed
    if getattr(agent, "start", None) == getattr(agent, "goal", None):
        return 0.0
    # If planner tracks finished_frame, caller may convert with fps outside
    return None

# -----------------------------
# Extra diagnostics you already had
# -----------------------------
def critical_nodes_count(agents: List[SupportsAgent]) -> int:
    claims: Dict[str, int] = {}
    for a in agents:
        r = getattr(a, "full_route", None) or []
        for n in r:
            claims[n] = claims.get(n, 0) + 1
    return sum(1 for _, c in claims.items() if c > 1)

# -----------------------------
# Public API
# -----------------------------
def collect_metrics(agents: List[SupportsAgent], graph: nx.Graph, fps: int | None = None) -> Dict:
    per_agent: Dict[str, Dict[str, Any]] = {}
    actuals: List[float] = []
    initials: List[float] = []
    waits_sec: List[float] = []
    finishes_sec: List[float] = []

    total = len(agents)
    successes = 0

    for a in agents:
        L_actual  = actual_route_length(a, graph)
        L_initial = initial_path_length(a, graph)
        extra     = L_actual - L_initial

        actuals.append(L_actual)
        initials.append(L_initial)

        # Waiting time
        w_sec = waiting_time_seconds(a)
        waits_sec.append(w_sec)
        w_frames = int(round(w_sec * fps)) if fps else getattr(a, "wait_frames", 0)

        # Finish time (seconds if possible)
        ft_sec = finish_time_seconds(a)
        if ft_sec is not None:
            finishes_sec.append(ft_sec)

        # Success condition (finished OR trivial start==goal)
        finished_flag = bool(getattr(a, "finished", False))
        if finished_flag or getattr(a, "start", None) == getattr(a, "goal", None):
            successes += 1

        per_agent[a.name] = {
            "finished": finished_flag,
            "start_equals_goal": getattr(a, "start", None) == getattr(a, "goal", None),
            "actual_length": round(L_actual, 3),
            "initial_length": round(L_initial, 3),
            "extra_length": round(extra, 3),
            "detour_ratio": (L_actual / L_initial) if L_initial > 0 else float("inf"),
            "wait_time_sec": round(w_sec, 3),
            "wait_time_frames": int(w_frames),
            "replans": int(getattr(a, "replans", 0)),
            "gates": int(getattr(a, "gates", 0)),
            "route_nodes": _extract_executed_nodes(a),  # optional visibility
        }

    # Totals
    sum_actual = sum(actuals)
    sum_initial = sum(initials)
    sum_extra = sum_actual - sum_initial
    total_wait_sec = sum(waits_sec)
    total_wait_frames = int(round(total_wait_sec * fps)) if fps else sum(int(getattr(a, "wait_frames", 0)) for a in agents)

    # Makespan in seconds (prefer time-aware), else try frames, else distance-as-time
    makespan_sec: Optional[float] = max(finishes_sec) if finishes_sec else None
    makespan_frames: Optional[int] = None
    if makespan_sec is None:
        # Try finished_frame if present
        frames = [int(getattr(a, "finished_frame", 0)) for a in agents if hasattr(a, "finished_frame")]
        if frames:
            makespan_frames = max(frames)
            makespan_sec = (makespan_frames / float(fps)) if fps else None
    else:
        makespan_frames = int(round(makespan_sec * fps)) if fps else None

    # For backwards compatibility with existing code that expects these keys:
    sum_of_costs = round(sum_actual, 3)
    makespan_distance = round(max(actuals) if actuals else 0.0, 3)  # legacy (distance-based proxy)

    success_rate = (100.0 * successes / total) if total > 0 else 0.0

    report = {
        # New summary
        "success_rate_pct": success_rate,
        "success_count": successes,
        "total_agents": total,

        "sum_actual_length": round(sum_actual, 3),
        "sum_initial_length": round(sum_initial, 3),
        "sum_extra_length": round(sum_extra, 3),

        "total_wait_time_sec": round(total_wait_sec, 3),
        "total_wait_time_frames": int(total_wait_frames),
        "makespan_sec": round(makespan_sec, 3) if makespan_sec is not None else None,

        # Legacy fields kept for compatibility
        "success": (successes == total and total > 0),
        "sum_of_costs": sum_of_costs,
        "makespan_distance": makespan_distance,

        "avg_detour_ratio": (
            sum((per_agent[n]["detour_ratio"] for n in per_agent)) / len(per_agent)
        ) if per_agent else 0.0,

        "critical_nodes": critical_nodes_count(agents),

        "per_agent": per_agent,
    }
    return report

def print_metrics(report: Dict, fps: int | None = None):
    print("\n==== Metrics ====")

    # 5) Success rate
    print(f"Success Rate        : {report['success_rate_pct']:.1f}%  "
          f"({report['success_count']}/{report['total_agents']})")

    # 4) Makespan
    ms_frames = report.get("makespan_frames")
    ms_sec    = report.get("makespan_sec")
    if ms_sec is not None:
        print(f"Makespan            : {ms_sec:.2f} s")
    else:
        # Legacy fallback when no timing exists
        print(f"Makespan (distance) : {report['makespan_distance']} (no timing available)")

    # 1) Actual route length
    print(f"Sum Route Length    : {report['sum_actual_length']}  (graph-weighted)")
    # 2) Initial path length
    print(f"Sum Initial Length  : {report['sum_initial_length']}")
    # Extra
    extra = report['sum_extra_length']
    sign  = "+" if extra >= 0 else "-"
    print(f"Extra Length Total  : {extra:.3f} ({sign}{abs(extra):.3f} vs initial)")

    # 3) Waiting time
    if fps:
        print(f"Total Waiting       : {report['total_wait_time_frames']} frames  (~{report['total_wait_time_sec']:.2f} s)")
    else:
        print(f"Total Waiting (s)   : {report['total_wait_time_sec']:.2f} s")

    # Diagnostics retained
    print(f"Avg Detour Ratio    : {report['avg_detour_ratio']:.3f}")
    print(f"Critical Nodes      : {report['critical_nodes']}")

    # Per-agent breakdown
    for name, m in report["per_agent"].items():
        if fps:
            wt = f"{m['wait_time_frames']}f (~{m['wait_time_sec']:.2f}s)"
        else:
            wt = f"{m['wait_time_sec']:.2f}s"
        print(f"  - {name}: finished={m['finished']}  "
              f"len={m['actual_length']}  init={m['initial_length']}  extra={m['extra_length']}  "
              f"detour={m['detour_ratio']:.3f}  wait={wt}  "
              f"replans={m['replans']}  gates={m['gates']}")






# A single schema that can hold both run-level and per-agent rows
_FIELDNAMES = [
    "row_type", "run_id", "ts",
    "map_id", "planner", "n_agents",
    "success_rate", "makespan_sec",
    "sum_of_costs_actual", "sum_of_costs_initial", "sum_of_extra_length",
    "total_wait_sec",
    "avg_detour_ratio",
    "critical_nodes",
    # per-agent fields (filled only for row_type='agent')
    "agent", "start", "goal", "finished",
    "initial_len", "actual_len", "extra_len", "detour_ratio",
    "wait_sec", "replans", "gates",
]

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _maybe_round(x: Optional[float], nd=3):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return x

def append_run_to_csv(
    report: dict,
    csv_path: str,
    meta: dict,
    agents: Optional[list] = None,
) -> str:
    """
    Append one run's results to `csv_path`.
    - `report`: from collect_metrics(agents, graph)
    - `meta`: dict with keys like:
        {"map_id": "...", "planner": "...", "n_agents": len(agents)}
      (Fill what you have; missing keys become None.)
    - `agents`: list of Agent; used to backfill start/goal if report doesn't include them.
    Returns: absolute path to the CSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    new_file = not os.path.exists(csv_path)

    run_id = meta.get("run_id") or str(uuid.uuid4())
    fps = meta.get("fps")
    seed = meta.get("seed")

    # Global row
    global_row = {
        "row_type": "global",
        "run_id": run_id,
        "ts": _now_iso(),
        "map_id": meta.get("map_id"),
        "planner": meta.get("planner"),
        "n_agents": meta.get("n_agents"),
        "success_rate": report.get("success_rate_pct"),
        "makespan_sec": report.get("makespan_sec"),
        "sum_of_costs_actual": _maybe_round(report.get("sum_actual_length")),
        "sum_of_costs_initial": _maybe_round(report.get("sum_initial_length")),
        "sum_of_extra_length": _maybe_round(report.get("sum_extra_length")),
        "total_wait_sec": _maybe_round(report.get("total_wait_time_sec")),
        "avg_detour_ratio": _maybe_round(report.get("avg_detour_ratio")),
        "critical_nodes": report.get("critical_nodes"),
        # agent fields left blank on purpose
        "agent": None, "start": None, "goal": None, "finished": None,
        "initial_len": None, "actual_len": None, "extra_len": None,
        "detour_ratio": None, "wait_sec": None,
        "replans": None, "gates": None,
    }

    # Build a quick lookup for start/goal if needed
    a_lookup = {getattr(a, "name", f"a{i}"): a for i, a in enumerate(agents or [])}

    # Per-agent rows
    agent_rows = []
    per_agent = report.get("per_agent", {}) or {}
    for name, m in per_agent.items():
        # fallbacks if metrics didn’t include start/goal/waits in seconds
        start = m.get("start") or getattr(a_lookup.get(name), "start", None)
        goal = m.get("goal") or getattr(a_lookup.get(name), "goal", None)
        wait_sec = m.get("wait_sec")
        if wait_sec is None and fps:
            wait_sec = (m.get("wait_frames") or 0) / float(fps)

        row = {
            "row_type": "agent",
            "run_id": run_id,
            "ts": _now_iso(),
            "map_id": meta.get("map_id"),
            "planner": meta.get("planner"),
            "n_agents": meta.get("n_agents"),
            # copy some globals for convenience; others left None
            "success_rate": None,
            "makespan_sec": None,
            "sum_of_costs_actual": None,
            "sum_of_costs_initial": None,
            "sum_of_extra_length": None,
            "total_wait_sec": None,
            "avg_detour_ratio": None,
            "critical_nodes": None,
            # per-agent
            "agent": name,
            "start": start,
            "goal": goal,
            "finished": bool(m.get("finished", False)),
            "initial_len": _maybe_round(m.get("initial_length")),
            "actual_len": _maybe_round(m.get("actual_length")),
            "extra_len": _maybe_round(m.get("extra_length")),
            "detour_ratio": _maybe_round(m.get("detour_ratio")),
            "wait_sec": _maybe_round(wait_sec),
            "replans": m.get("replans"),
            "gates": m.get("gates"),
        }
        agent_rows.append(row)

    # Write/append
    abs_path = os.path.abspath(csv_path)
    with open(abs_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerow(global_row)
        for r in agent_rows:
            writer.writerow(r)

    return abs_path