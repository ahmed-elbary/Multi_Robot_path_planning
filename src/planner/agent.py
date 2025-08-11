# --- add near top of file ---
from typing import List, Optional

class Agent:
    def __init__(self, name: str, start: str, goal: str):
        self.name = name
        self.start = start
        self.goal = goal
        self.route: List[str] = []
        self.full_route: List[str] = []
        self.fragments: List[List[str]] = []
        self.active: bool = True
        self.waiting: bool = False
        self.wait_node: Optional[str] = None
        self.finished: bool = False
        self.replanned: bool = False
        self.dynamic_coords = []
        self.dynamic_angles = []
        self.wait_counter = 0
        self.original_path_length: int = 0
        self.arrival_time: float = 0.0
        self.resume_time: float = 0.0
        self.current_fragment_idx = 0
        self.current_node = self.start

        # --- NEW: critical-point signaling / ownership ---
        self.blocked_by_node: Optional[str] = None   # collision node (critical point) we're waiting on
        self.blocker_owner: Optional[str] = None     # agent name that owns the node by priority
        self.resume_ready: bool = False              # set True when blocker has passed critical node

    def priority(self) -> int:
        return int(self.name.replace("Robot", ""))
