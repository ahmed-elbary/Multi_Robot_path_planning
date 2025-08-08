from typing import List

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
        self.wait_node: str = None
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

    def priority(self) -> int:
        return int(self.name.replace("Robot", ""))
