import yaml
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Tuple
import math
import os
from itertools import combinations
from matplotlib.patches import FancyArrowPatch

class Agent:
    def __init__(self, name: str, start: str, goal: str):
        self.name = name
        self.start = start
        self.goal = goal
        self.route: List[str] = []
        self.full_route: List[str] = [] 
        self.active: bool = True
        self.fragments: List[List[str]] = []
        self.waiting: bool = False
        self.wait_node: str = None
        self.finished: bool = False
        self.replanned: bool = False
        self.dynamic_coords = []
        self.dynamic_angles = []
        self.wait_counter = 0
        

        
    def priority(self):
        return int(self.name.replace("Robot", ""))

class FragmentPlanner:
    def __init__(self, yaml_file: str, agents: List[Agent]):
        self.yaml_file = yaml_file
        self.topo_map = self.load_map(yaml_file)
        self.graph, self.positions = self.build_graph_from_yaml(self.topo_map)
        self.agents = agents
        self.routes = {}
        self.occupied_nodes = set()
        self.critical_points = set()
        self.dangerous_points = set()
        self.stopped_agents: Dict[str, str] = {}
        self.collision_window = 5.0  # seconds
        self.deadlock_logs = []

    def load_map(self, yaml_file: str):
        with open(yaml_file, 'r') as f:
            return yaml.safe_load(f)

    def build_graph_from_yaml(self, topo_map):
        G = nx.DiGraph()
        positions = {}
        for entry in topo_map:
            node_data = entry['node']
            name = node_data['name']
            x = node_data['pose']['position']['x']
            y = node_data['pose']['position']['y']
            G.add_node(name)
            positions[name] = (x, y)
            for edge in node_data.get('edges', []):
                G.add_edge(name, edge['node'])
        return G, positions

    def run(self):
        self.find_routes()
        self.update_goal_occupancy()
        self.process_routes()
        self.find_critical_points()
        self.detect_dangerous_points()
        self.resolve_all_waits()
        self.assign_waiting_agents()
        self.split_critical_paths()
        self.print_report()
        # self.replan_waiting_agents()
        # self.detect_deadlocks()
        self.animate_paths()



    
    def find_routes(self):
        for agent in self.agents:
            if agent.goal is None or agent.start == agent.goal:
                agent.route = [agent.start]
                agent.active = False
                continue
            try:
                agent.route = nx.shortest_path(self.graph, agent.start, agent.goal)
            except nx.NetworkXNoPath:
                agent.route = []
                agent.active = False
            self.routes[agent.name] = agent.route

    def update_goal_occupancy(self):
        for agent in self.agents:
            if agent.goal is None or agent.start == agent.goal:
                self.occupied_nodes.add(agent.start)
            elif agent.route and agent.route[-1] == agent.goal:
                self.occupied_nodes.add(agent.goal)

    def process_routes(self):
        for agent in self.agents:
            if agent.active:
                self.process_route_for_active(agent)
            else:
                self.process_route_for_inactive(agent)

    def process_route_for_active(self, agent: Agent):
        if not agent.route:
            self.execute_recovery_behaviour(agent)
            return
        filtered_map = self.generate_filtered_map(agent)
        try:
            route = nx.shortest_path(filtered_map, agent.start, agent.goal)
            agent.route = route
        except nx.NetworkXNoPath:
            self.execute_recovery_behaviour(agent)

    def process_route_for_inactive(self, agent: Agent):
        if not agent.goal:
            agent.route = [agent.start]
            return
        closest = self.find_closest_node(agent.start)
        agent.route = [agent.start, closest]

    def generate_filtered_map(self, agent: Agent) -> nx.Graph:
        filtered = self.graph.copy()
        for node in self.occupied_nodes:
            if node != agent.goal and node in filtered:
                filtered.remove_node(node)
        return filtered

    def execute_recovery_behaviour(self, agent: Agent):
        agent.route = [agent.start]
        agent.active = False

    def find_closest_node(self, start: str) -> str:
        return min(self.graph.nodes, key=lambda n: nx.shortest_path_length(self.graph, start, n))

    def find_critical_points(self):
        point_count: Dict[str, int] = {}
        for agent in self.agents:
            for node in agent.route:
                point_count[node] = point_count.get(node, 0) + 1
        self.critical_points = {node for node, count in point_count.items() if count > 1}

    def detect_dangerous_points(self):
        self.detected_dangerous_points = {}
        node_timings = {}

        # Build arrival/departure windows for each node
        for agent in self.agents:
            route = agent.route
            positions = self.positions
            fps = 10  # same as used in animation
            speed = 1.0  # 1 m/s

            time = 0.0
            for i in range(len(route) - 1):
                a, b = route[i], route[i+1]
                x1, y1 = positions[a]
                x2, y2 = positions[b]
                dist = math.hypot(x2 - x1, y2 - y1)
                duration = dist / speed
                start_time = time
                end_time = time + duration
                node_timings.setdefault(b, []).append((agent.name, start_time, end_time))
                time = end_time

        # Check for overlapping windows at each node
        for node, visits in node_timings.items():
            for i in range(len(visits)):
                name_i, start_i, end_i = visits[i]
                for j in range(i+1, len(visits)):
                    name_j, start_j, end_j = visits[j]
                    if max(start_i, start_j) <= min(end_i, end_j):
                        self.detected_dangerous_points.setdefault(node, set()).update([name_i, name_j])


    def resolve_all_waits(self):
        any_waiting = True
        while any_waiting:
            before = {a.name for a in self.agents if a.waiting}
            self.assign_waiting_agents()
            after = {a.name for a in self.agents if a.waiting}
            new_waiters = after - before
            any_waiting = bool(new_waiters)
            if new_waiters:
                self.replan_waiting_agents()
                
    
    def split_critical_paths(self):
        for agent in self.agents:
            route = agent.route
            if not route or not agent.active:
                continue
            fragments = []
            fragment = []
            for i, node in enumerate(route):
                if node in self.dangerous_points:
                    if not self.agent_has_priority(agent, node):
                        agent.waiting = True
                        if i > 0:
                            agent.wait_node = route[i - 1]
                        break
                fragment.append(node)
            if fragment:
                agent.fragments.append(fragment)

    def agent_has_priority(self, agent: Agent, node: str) -> bool:
        competing_agents = [a for a in self.agents if a.name != agent.name and node in a.route]
        agent_index = agent.route.index(node) if node in agent.route else float('inf')
        agent_route_len = len(agent.route)

        for other in competing_agents:
            other_index = other.route.index(node)
            other_len = len(other.route)
            if agent_index > other_index:
                return False
            elif agent_index == other_index:
                if agent_route_len > other_len:
                    return False
                elif agent_route_len == other_len:
                    if agent.name > other.name:
                        return False
        return True



    def replan_waiting_agents(self):
        for agent in self.agents:
            if agent.waiting and agent.wait_node and not agent.replanned:
                if not any(agent.wait_node in a.route for a in self.agents if a != agent):
                    filtered_map = self.generate_filtered_map(agent)
                    try:
                        new_path = nx.shortest_path(filtered_map, agent.wait_node, agent.goal)
                        agent.route = [agent.start] + new_path[1:]
                        agent.waiting = False
                        agent.replanned = True
                        print(f"[✓] {agent.name} replanned from '{agent.wait_node}' to '{agent.goal}': {agent.route}")
                    except nx.NetworkXNoPath:
                        print(f"[x] {agent.name} could not replan from '{agent.wait_node}' to '{agent.goal}'")


    def assign_waiting_agents(self):
        if not hasattr(self, 'detected_dangerous_points'):
            return

        for node, robot_names in self.detected_dangerous_points.items():
            if len(robot_names) <= 1:
                continue

            # Get fragments for each involved robot
            robot_objs = [a for a in self.agents if a.name in robot_names]
            robot_fragments = {a.name: a.route for a in robot_objs if a.route}

            # Prioritize by robot ID
            sorted_robots = sorted(robot_objs, key=lambda a: a.priority())
            winner = sorted_robots[0]
            others = sorted_robots[1:]

            for loser in others:
                # Identify earliest intersection between loser's route and winner's route
                intersection_idx = None
                for idx, node in enumerate(loser.route):
                    if node in winner.route:
                        intersection_idx = idx
                        break

                if intersection_idx is not None:
                    # Set wait node as the node before the intersection point
                    wait_idx = max(intersection_idx - 1, 0)
                    loser.wait_node = loser.route[wait_idx]
                    loser.waiting = True

                    if not loser.full_route:
                        loser.full_route = loser.route[:]
                    loser.route = loser.route[:wait_idx + 1]


    def detect_deadlocks(self):
        deadlocks = []
        seen_pairs = set()
        for a1 in self.agents:
            path1 = a1.full_route if a1.full_route else a1.route
            for i in range(len(path1) - 1):
                edge1 = (path1[i], path1[i+1])
                for a2 in self.agents:
                    if a1.name == a2.name:
                        continue
                    path2 = a2.full_route if a2.full_route else a2.route
                    for j in range(len(path2) - 1):
                        edge2 = (path2[j], path2[j+1])
                        if edge1 == edge2[::-1]:
                            key = tuple(sorted([a1.name, a2.name]))
                            if key not in seen_pairs:
                                seen_pairs.add(key)
                                deadlocks.append((a1.name, a2.name, edge1))
                                if a1.priority() < a2.priority():
                                    msg = f"[PRIORITY] {a1.name} allowed to proceed due to higher priority"
                                else:
                                    msg = f"[PRIORITY] {a2.name} allowed to proceed due to higher priority"
                                self.deadlock_logs.append(msg)

        for agent in self.agents:
            if agent.waiting and agent.wait_node and agent.full_route:
                try:
                    idx = agent.full_route.index(agent.wait_node)
                    for offset in range(1, 4):  # Check next 3 segments (coming path)
                        if idx + offset < len(agent.full_route):
                            my_edge = (agent.full_route[idx + offset - 1], agent.full_route[idx + offset])
                            for other in self.agents:
                                if other.name == agent.name or not other.full_route:
                                    continue
                                for i in range(len(other.full_route) - 1):
                                    other_edge = (other.full_route[i], other.full_route[i+1])
                                    if my_edge == other_edge[::-1]:
                                        key = tuple(sorted([agent.name, other.name]))
                                        if key not in seen_pairs:
                                            seen_pairs.add(key)
                                            deadlocks.append((agent.name, other.name, my_edge))
                                            if agent.priority() < other.priority():
                                                msg = f"[PRIORITY] {agent.name} allowed to proceed due to higher priority"
                                            else:
                                                msg = f"[PRIORITY] {other.name} allowed to proceed due to higher priority"
                                            self.deadlock_logs.append(msg)
                                    # Also check if other agent's upcoming segment matches our direction (not just inverse)
                                    elif my_edge == other_edge:
                                        # This might mean both are trying to go through same edge in same direction
                                        # This isn't deadlock but still could imply congestion
                                        continue

                    # NEW: Check if other robots' paths include our full_route (in reverse)
                    reversed_path = list(reversed(agent.full_route[idx:idx+4]))
                    for other in self.agents:
                        if other.name == agent.name or not other.full_route:
                            continue
                        for i in range(len(other.full_route) - len(reversed_path) + 1):
                            if other.full_route[i:i+len(reversed_path)] == reversed_path:
                                key = tuple(sorted([agent.name, other.name]))
                                if key not in seen_pairs:
                                    seen_pairs.add(key)
                                    deadlocks.append((agent.name, other.name, (reversed_path[1], reversed_path[0])))
                                    if agent.priority() < other.priority():
                                        msg = f"[PRIORITY] {agent.name} allowed to proceed due to higher priority"
                                    else:
                                        msg = f"[PRIORITY] {other.name} allowed to proceed due to higher priority"
                                    self.deadlock_logs.append(msg)

                except ValueError:
                    continue

        for a, b, (u, v) in deadlocks:
            msg = f"[DEADLOCK DETECTED] {a} ↔ {b} at edge ({u} ↔ {v})"
            self.deadlock_logs.append(msg)
        return deadlocks

    def print_report(self):
        print("\n==== Fragment Planner Report ====\n")
        print("[Agents]")
        for agent in self.agents:
            print(f"- {agent.name}: start={agent.start}, goal={agent.goal}")

        print("\n[Agent Status]")
        for agent in self.agents:
            if agent.finished:
                status = "Finished"
            elif not agent.route or agent.goal == agent.start:
                status = "Inactive (No goal)"
            elif agent.waiting:
                status = "Waiting (Stalled at dangerous point)"
            else:
                status = "Active"
            print(f"- {agent.name}: {status}")

        print("\n[Occupied Nodes]")
        occupied = sorted({agent.route[-1] for agent in self.agents if agent.route})
        print(", ".join(occupied))

        print("\n[Initial Routes]")
        for agent in self.agents:
            route_to_print = agent.full_route if agent.full_route else agent.route
            print(f"- {agent.name}: {route_to_print}")

        # Intersected Nodes
        node_map = {}
        for agent in self.agents:
            nodes = agent.full_route if agent.full_route else agent.route
            for node in nodes:
                node_map.setdefault(node, set()).add(agent.name)
        intersected = {k: v for k, v in node_map.items() if len(v) > 1}
        print("\n[Intersected Nodes]")
        if intersected:
            for node, robots in intersected.items():
                print(f"- {node}: shared by {', '.join(sorted(robots))}")
        else:
            print("(none)")

        # Dangerous points based on temporal overlaps
        print("\n[Dangerous Collision Points (Time-Based)]")
        if hasattr(self, 'detected_dangerous_points') and self.detected_dangerous_points:
            for node, robots in self.detected_dangerous_points.items():
                print(f"- {node}: time conflict between {', '.join(sorted(robots))}")
        else:
            print("(none detected)")

        print("\n[Stopped Agents at Critical Points]")
        for agent in self.agents:
            if agent.waiting and agent.wait_node and hasattr(self, 'detected_dangerous_points') and any(agent.name in v for v in self.detected_dangerous_points.values()):
                print(f"- {agent.name} will wait at '{agent.wait_node}' to avoid dangerous point ahead")

        if hasattr(self, 'replanning_logs'):
            for line in self.replanning_logs:
                print(line)

        if hasattr(self, 'deadlock_logs'):
            for log in self.deadlock_logs:
                print(log)

            
    def get_orientation_from_map(self, node_name: str):
        for entry in self.topo_map:
            node = entry['node']
            if node['name'] == node_name:
                q = node['pose']['orientation']
                z, w = q['z'], q['w']
                return math.atan2(2.0 * w * z, 1.0 - 2.0 * (z ** 2))
        return None








    def animate_paths(self):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title("Multi-Robot Path Animation")
        ax.axis('equal')
        ax.grid(False)

        for entry in self.topo_map:
            node = entry['node']
            name = node['name']
            x, y = self.positions[name]
            ax.scatter(x, y, c='blue')
            ax.text(x + 0.1, y - 0.1, name, fontsize=9)
            for edge in node.get('edges', []):
                to_node = edge['node']
                if to_node in self.positions:
                    x2, y2 = self.positions[to_node]
                    ax.plot([x, x2], [y, y2], color='gray')
                    dx, dy = x2 - x, y2 - y
                    ax.add_patch(FancyArrowPatch((x + 0.25 * dx, y + 0.25 * dy),
                                                 (x + 0.30 * dx, y + 0.30 * dy),
                                                 arrowstyle='-|>', mutation_scale=15, color='gray'))

        colors = ['red', 'green', 'blue', 'orange', 'purple']
        trails = []
        max_frames = 0
        fps = 10
        wait_time = 3.0

        for idx, agent in enumerate(self.agents):
            path = agent.full_route if agent.full_route else agent.route
            if not path or len(path) < 2:
                continue
            coords = []
            angles = []
            for i in range(len(path) - 1):
                x1, y1 = self.positions[path[i]]
                x2, y2 = self.positions[path[i + 1]]
                steps = max(int(math.hypot(x2 - x1, y2 - y1) * fps), 1)
                for t in range(steps):
                    alpha = t / steps
                    x = (1 - alpha) * x1 + alpha * x2
                    y = (1 - alpha) * y1 + alpha * y2
                    coords.append((x, y))
                    angles.append(math.atan2(y2 - y1, x2 - x1))

            final_pos = self.positions[path[-1]]
            coords.append(final_pos)
            final_angle = self.get_orientation_from_map(path[-1]) or angles[-1]
            while len(angles) < len(coords):
                angles.append(final_angle)

            if agent.wait_node:
                wait_pos = self.positions[agent.wait_node]
                try:
                    wait_idx = next(i for i, p in enumerate(coords)
                                    if math.isclose(p[0], wait_pos[0], abs_tol=1e-3)
                                    and math.isclose(p[1], wait_pos[1], abs_tol=1e-3))
                    wait_frames = int(wait_time * fps)
                    coords[wait_idx:wait_idx] = [wait_pos] * wait_frames
                    angles[wait_idx:wait_idx] = [angles[wait_idx]] * wait_frames
                except StopIteration:
                    pass

            xs, ys = zip(*[self.positions[n] for n in path if n in self.positions])
            ax.plot(xs, ys, '--', color=colors[idx % len(colors)], linewidth=2)

            dot, = ax.plot([], [], 'o', color=colors[idx % len(colors)], markersize=8)
            arrow = None
            trails.append({'coords': coords, 'angles': angles, 'dot': dot, 'arrow': arrow, 'color': colors[idx % len(colors)], 'agent': agent})
            agent.dynamic_coords = coords.copy()
            agent.dynamic_angles = angles.copy()
            max_frames = max(max_frames, len(coords))

        def update(frame):
            for trail in trails:
                agent = trail['agent']
                coords = agent.dynamic_coords
                angles = agent.dynamic_angles

                if agent.waiting and agent.wait_node:
                    current_idx = agent.full_route.index(agent.wait_node)
                    try:
                        next_node = agent.full_route[current_idx + 1]
                    except IndexError:
                        continue
                    blocking_agents = [a for a in self.agents if a.name != agent.name and next_node in a.route]
                    should_wait = any(a.priority() < agent.priority() for a in blocking_agents)

                    if not should_wait:
                        agent.waiting = False
                        coords = []
                        angles = []
                        path = agent.full_route[current_idx:]
                        for i in range(len(path) - 1):
                            x1, y1 = self.positions[path[i]]
                            x2, y2 = self.positions[path[i + 1]]
                            steps = max(int(math.hypot(x2 - x1, y2 - y1) * fps), 1)
                            for t in range(steps):
                                alpha = t / steps
                                x = (1 - alpha) * x1 + alpha * x2
                                y = (1 - alpha) * y1 + alpha * y2
                                coords.append((x, y))
                                angles.append(math.atan2(y2 - y1, x2 - x1))
                        coords.append(self.positions[path[-1]])
                        final_angle = self.get_orientation_from_map(path[-1]) or angles[-1]
                        while len(angles) < len(coords):
                            angles.append(final_angle)
                        agent.dynamic_coords = coords
                        agent.dynamic_angles = angles

                idx = min(frame, len(agent.dynamic_coords) - 1)
                x, y = agent.dynamic_coords[idx]
                yaw = agent.dynamic_angles[idx]
                dx = 0.5 * math.cos(yaw)
                dy = 0.5 * math.sin(yaw)
                trail['dot'].set_data([x], [y])
                if trail['arrow']:
                    trail['arrow'].remove()
                trail['arrow'] = FancyArrowPatch((x, y), (x + dx, y + dy),
                                                 arrowstyle='-|>', mutation_scale=20,
                                                 color=trail['color'])
                ax.add_patch(trail['arrow'])
            return [obj for trail in trails for obj in (trail['dot'], trail['arrow']) if obj is not None]

        ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=100, blit=True, repeat=False)
        plt.show()



if __name__ == "__main__":
    yaml_file = "data/map.yaml"
    agents = [
        Agent("Robot1", "Park1", "T01"),
        Agent("Robot2", "Park2", "T52"),
        Agent("Robot3", "Park3", "T00"),
        # Agent("Robot4", "Spare2", "T51"),

    ]
    planner = FragmentPlanner(yaml_file, agents)
    planner.run()

