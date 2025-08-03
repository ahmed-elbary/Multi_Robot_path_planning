
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

    def priority(self) -> int:
        # Defines agent priority (lower value = higher priority)
        return int(self.name.replace("Robot", ""))