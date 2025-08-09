# planner_core_fragment.py
import networkx as nx
from typing import List, Dict, Tuple
from .agent import Agent
import heapq

def find_routes(agents, graph):
    """
    Each agent plans its route using a spatially filtered map.
    If blocked, the agent will be assigned a 'wait' at a safe node.
    """
    for i, agent in enumerate(agents):
        print(f"\n[✓] Planning spatial route for {agent.name} from {agent.start} to {agent.goal}")
        # Gather occupied nodes (by all previous agents)
        occupied = set()
        for prev in agents[:i]:
            occupied.update(prev.full_route)
        # Remove occupied except agent's own start/goal
        filtered = graph.copy()
        for node in occupied:
            if node not in (agent.start, agent.goal):
                if node in filtered:
                    filtered.remove_node(node)
        try:
            path = nx.shortest_path(filtered, agent.start, agent.goal)
            agent.route = [(path[j], path[j+1]) for j in range(len(path)-1)]
            agent.full_route = path
            agent.fragments = [agent.route]
            agent.waiting = False
            print(f"[{agent.name}] Planned path: {path}")
        except nx.NetworkXNoPath:
            # Blocked: send to a wait node (here: just wait at start)
            agent.route = []
            agent.full_route = [agent.start]
            agent.fragments = []
            agent.waiting = True
            agent.wait_node = agent.start
            print(f"[{agent.name}] No spatial path available, will wait at {agent.start}")

def find_critical_points(agents: List[Agent]) -> Dict[str, List[str]]:
    """
    Identify critical points (nodes) shared by more than one agent.
    """
    node_claims = {}
    for agent in agents:
        for u, v in agent.route:
            node_claims.setdefault(v, []).append(agent.name)
    return {n: names for n, names in node_claims.items() if len(names) > 1}

def agent_has_priority(agent: Agent, agents: List[Agent], node: str) -> bool:
    """
    Example: lower-numbered robot gets priority, but you can swap this logic.
    """
    claimers = [a for a in agents if any(v == node for _, v in a.route)]
    claimers.sort(key=lambda a: a.priority())
    return claimers and claimers[0].name == agent.name

def split_critical_paths(graph, agents: List[Agent], dangerous_points: List[str]):
    """
    Split routes at collision points. Lower-priority agents truncate their path at the last safe node.
    """
    for agent in agents:
        frag = []
        for u, v in agent.route:
            frag.append((u, v))
            if v in dangerous_points and not agent_has_priority(agent, agents, v):
                # Only claim up to last safe node; mark agent as waiting
                agent.fragments = [frag]
                agent.route = frag
                agent.full_route = [agent.start] + [step[1] for step in frag]
                agent.waiting = True
                agent.wait_node = u
                print(f"[{agent.name}] Wait at {u} due to collision at {v}")
                break
        else:
            agent.fragments = [frag]
            agent.waiting = False
            agent.wait_node = None

def assign_waiting_agents(agents):
    # In this version, nothing extra is needed; planner already set wait_node.
    pass

def replan_waiting_agents(agents, graph, **kwargs):
    """
    For fragment planner: periodically (in main/anim loop), try to replan for agents still waiting.
    """
    for agent in agents:
        if agent.waiting:
            try:
                print(f"[FragmentPlanner] Re-attempting to plan for {agent.name} from {agent.wait_node} to {agent.goal}")
                path = nx.shortest_path(graph, agent.wait_node, agent.goal)
                # Append new fragment
                frag = [(path[j], path[j+1]) for j in range(len(path)-1)]
                agent.fragments.append(frag)
                agent.route += frag
                agent.full_route += [step[1] for step in frag]
                agent.waiting = False
                print(f"[FragmentPlanner] {agent.name} resumed with new fragment: {path}")
            except nx.NetworkXNoPath:
                print(f"[FragmentPlanner] {agent.name} still blocked at {agent.wait_node}")
                # Stay waiting

def build_temporal_occupancy_table(agents: List[Agent]) -> Dict[int, Dict[str, str]]:
    """
    This function is mostly relevant to time-aware planners. Here, it is just a stub.
    """
    return {}

# ---- END of fragment_planner_core.py ----
