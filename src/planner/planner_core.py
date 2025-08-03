import networkx as nx
from typing import List, Dict, Tuple
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