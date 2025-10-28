# src/space_time_planner/agent_st.py
from typing import Optional, Callable, Dict, Tuple, List

TimedEdge = Tuple[str, str, float, float]  # (u, v, depart_s, arrive_s)

class Agent:
    def __init__(self, name, start, goal,
             steer_model: Optional[Callable[[Optional[str], str, str, Dict[str, Tuple[float, float]]], float]] = None,
             **kwargs):
        super().__init__(**kwargs) if hasattr(super(), "__init__") else None

        self.steer_model = steer_model

        self.name = name
        self.start = start
        self.goal = goal

        # Spatial plan (nodes)
        self.planned_path: List[str] = []      # initial spatial shortest path
        self.full_route: List[str] = []        # grows as fragments commit (nodes)

        # Time-aware plan (edges with times)
        self.route: List[TimedEdge] = []       # appended as edges are released
        self.fragments: List[List[TimedEdge]] = []  # optional: one-edge fragments

        # State flags
        self.active: bool = True
        self.finished: bool = False
        self.waiting: bool = False
        self.replanned: bool = False

        # Waiting / gating info
        self.wait_node: Optional[str] = None
        self.blocked_by_node: Optional[str] = None
        self.blocker_owner: Optional[str] = None

        # Timing / metrics
        self.arrival_time: float = 0.0
        self.scheduled_departure_frame: int = 0
        self.finished_frame: int = 0
        self.wait_frames: int = 0

        # Book-keeping
        self.current_fragment_idx: int = 0
        self.replans: int = 0
        self.gates: int = 0

    def priority(self) -> int:
        """Lower number = higher priority. Parses digits in name; fallback large."""
        try:
            return int(''.join(ch for ch in self.name if ch.isdigit()))
        except ValueError:
            return 10_000
