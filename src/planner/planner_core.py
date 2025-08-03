import networkx as nx
from typing import List, Dict
from .agent import Agent

def find_routes(agents: List[Agent], graph: nx.Graph):
    for agent in agents:
        if not agent.active or agent.start == agent.goal:
            agent.active = False
            continue
        try:
            path = nx.shortest_path(graph, source=agent.start, target=agent.goal)
            agent.route = path
            agent.full_route = path.copy()
            agent.timestamps = list(range(len(path)))  # Initialize timestamps per node
        except nx.NetworkXNoPath:
            agent.active = False

def find_critical_points(agents: List[Agent]) -> Dict[str, List[str]]:
    point_to_agents = {}
    for agent in agents:
        for node in agent.route:
            point_to_agents.setdefault(node, []).append(agent.name)
    return {k: v for k, v in point_to_agents.items() if len(v) > 1}

def agent_has_priority(agent: Agent, other_agents: List[Agent], node: str) -> bool:
    competing = [a for a in other_agents if node in a.route]
    competing.sort(key=lambda a: (a.route.index(node), len(a.route), a.priority()))
    return competing[0].name == agent.name

def split_critical_paths(agents: List[Agent], dangerous_points: List[str]):
    for agent in agents:
        agent.fragments = []
        frag = []
        for i, node in enumerate(agent.route):
            if node in dangerous_points and not agent_has_priority(agent, agents, node):
                agent.waiting = True
                agent.wait_node = agent.route[i-1] if i > 0 else agent.start
                break
            frag.append(node)
        agent.route = frag
        agent.fragments.append(frag)

def assign_waiting_agents(agents: List[Agent]):
    point_to_agents = {}
    for agent in agents:
        for node in agent.route:
            point_to_agents.setdefault(node, set()).add(agent.name)

    for node, names in point_to_agents.items():
        if len(names) <= 1:
            continue

        involved = [a for a in agents if a.name in names]
        involved.sort(key=lambda a: a.priority())
        leader = involved[0]

        for follower in involved[1:]:
            if node in follower.route:
                idx = follower.route.index(node)
                wait_idx = max(0, idx - 1)
                wait_node = follower.route[wait_idx]
                follower.wait_node = wait_node
                follower.waiting = True
                if not follower.full_route:
                    follower.full_route = follower.route[:]
                follower.route = follower.route[:wait_idx + 1]

def replan_waiting_agents(agents: List[Agent], graph: nx.Graph, frame: int = 0, fps: int = 10):
    time_occupancy = {}
    goal_locks = set()

    for other in agents:
        if not other.full_route:
            continue
        for t, node in enumerate(other.full_route):
            time_occupancy.setdefault((node, t), set()).add(other.name)
        if other.goal:
            goal_locks.add(other.goal)

    for agent in agents:
        if agent.waiting and agent.wait_node and not agent.replanned:
            start_time = len(agent.route)
            future_conflicts = set()

            if agent.goal and agent.goal in graph:
                for offset, node in enumerate(agent.full_route[len(agent.route):]):
                    t = start_time + offset
                    if (node, t) in time_occupancy:
                        competing_agents = time_occupancy[(node, t)]
                        if any(name != agent.name for name in competing_agents):
                            future_conflicts.add(node)

                filtered = graph.copy()
                for node in future_conflicts.union(goal_locks - {agent.goal}):
                    if node in filtered:
                        filtered.remove_node(node)

                try:
                    new_path = nx.shortest_path(filtered, source=agent.wait_node, target=agent.goal)
                    agent.route = [agent.wait_node] + new_path[1:]
                    agent.waiting = False
                    agent.replanned = True
                    agent.full_route = agent.route.copy()
                    agent.timestamps = list(range(start_time, start_time + len(agent.route)))
                    print(f"[✓] {agent.name} replanned from '{agent.wait_node}' to '{agent.goal}': {agent.route}")
                except nx.NetworkXNoPath:
                    print(f"[x] {agent.name} could not replan from '{agent.wait_node}' to '{agent.goal}'")
