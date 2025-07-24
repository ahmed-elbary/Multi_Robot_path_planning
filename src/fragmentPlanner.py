# Fragment Planner Animation with Correct Pause & Conflict Resolution
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import math
import networkx as nx
from itertools import combinations
from utils import build_graph_from_yaml, find_path
from matplotlib.patches import FancyArrowPatch

# === Utility Functions ===
def load_map(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def quaternion_to_yaw(z, w):
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * (z ** 2))

# === A* Replanner Stub ===
class DStarLite:
    def __init__(self, graph, positions):
        self.graph = graph
        self.positions = positions

    def heuristic(self, u, v):
        x1, y1 = self.positions[u]
        x2, y2 = self.positions[v]
        return math.hypot(x2 - x1, y2 - y1)

    def plan(self, start, goal):
        if start not in self.graph or goal not in self.graph:
            return []
        try:
            return nx.astar_path(
                self.graph, start, goal,
                heuristic=lambda u, v: self.heuristic(u, v),
                weight=lambda u, v, d: d.get('weight', 1)
            )
        except nx.NetworkXNoPath:
            return []

# === Motion & Delay Computation ===
def compute_arrival_times(path, positions, speed, delays):
    times = [0.0]
    for i in range(len(path) - 1):
        wait = delays.get(path[i], 0)
        t = times[-1] + wait
        x1, y1 = positions[path[i]]
        x2, y2 = positions[path[i + 1]]
        t += math.hypot(x2 - x1, y2 - y1) / speed
        times.append(t)
    return times

# === Path Interpolation ===
def interpolate_with_indices(path, positions, speed, fps, delays):
    pts, angs, idxs = [], [], []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(int(dist / speed * fps), 1)
        for s in range(steps):
            a = s / steps
            pts.append((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a))
            angs.append(math.atan2(y2 - y1, x2 - x1))
            idxs.append(i)
        wait = delays.get(v, 0)
        for _ in range(int(wait * fps)):
            pts.append((x2, y2))
            angs.append(math.atan2(y2 - y1, x2 - x1))
            idxs.append(i + 1)
    if path:
        pts.append(positions[path[-1]])
        angs.append(0.0)
        idxs.append(len(path) - 1)
    return pts, angs, idxs

# === Conflict Prediction ===
def detect_conflicts(arts, specs, lookahead):
    future_nodes = {}
    for i, art in enumerate(arts):
        idx = art['idxs'][art['frame']]
        future_nodes[i] = specs[i]['path'][idx + 1:idx + 1 + lookahead]
    conflicts = []
    for i, j in combinations(range(len(arts)), 2):
        shared = set(future_nodes[i]) & set(future_nodes[j])
        if shared:
            node = min(shared, key=lambda n: specs[i]['path'].index(n))
            conflicts.append((i, j, node))
    return conflicts

# === Resolution ===
def apply_delay(rid, node, specs, arts, speed, fps, delay):
    art = arts[rid]
    if art.get('last_pause_node') == node:
        return
    specs[rid]['delays'][node] = specs[rid]['delays'].get(node, 0) + delay
    pts, angs, idxs = interpolate_with_indices(
        specs[rid]['path'], positions, speed, fps, specs[rid]['delays']
    )
    art.update({'pts': pts, 'angs': angs, 'idxs': idxs})
    art['pause_until'] = art['frame'] + int(delay * fps)
    art['last_pause_node'] = node
    print(f"→ Robot {rid+1} pausing at {node} for {delay}s")

# === Animation ===
def animate(yaml_file, goals, colors, speed=1.0, fps=10, lookahead=3, delay=2.0):
    global topo_map, positions
    topo_map = load_map(yaml_file)
    graph, positions = build_graph_from_yaml(topo_map)

    specs, arts = [], []
    print("Initial paths:")
    for rid, g in enumerate(goals):
        path = []
        for si, (a, b) in enumerate(zip(g, g[1:])):
            seg = find_path(graph, a, b, positions)
            if si > 0:
                seg = seg[1:]
            path += seg
        specs.append({'path': path, 'delays': {}})
        print(f"  Robot {rid+1}: {path}")

    fig, ax = plt.subplots(figsize=(8,6))
    ax.axis('equal'); ax.grid(False)
    for item in topo_map:
        n = item['node']
        x, y = n['pose']['position']['x'], n['pose']['position']['y']
        ax.scatter(x, y, c='blue')
        ax.text(x+0.1, y-0.1, n['name'], fontsize=9)
        for e in n.get('edges', []):
            x2, y2 = positions[e['node']]
            ax.plot([x, x2], [y, y2], color='gray', lw=1)

    for i, s in enumerate(specs):
        pts, angs, idxs = interpolate_with_indices(
            s['path'], positions, speed, fps, s['delays']
        )
        dot, = ax.plot([], [], 'o', color=colors[i])
        arts.append({
            'pts': pts, 'angs': angs, 'idxs': idxs,
            'dot': dot, 'arrow': None,
            'frame': -1, 'pause_until': -1, 'last_pause_node': None
        })

    maxf = max(len(a['pts']) for a in arts)

    def update(_):
        artists = []
        for rid, art in enumerate(arts):
            # if paused and not expired, hold frame
            if 0 <= art['pause_until'] >= art['frame'] + 1:
                pass
            else:
                art['frame'] = min(art['frame'] + 1, len(art['pts']) - 1)
            idx = art['frame']

            x, y = art['pts'][idx]
            art['dot'].set_data([x], [y])
            if art['arrow']:
                art['arrow'].remove()
            dx = 0.3 * math.cos(art['angs'][idx])
            dy = 0.3 * math.sin(art['angs'][idx])
            art['arrow'] = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='-|>', mutation_scale=10)
            ax.add_patch(art['arrow'])
            artists += [art['dot'], art['arrow']]

        for i, j, node in detect_conflicts(arts, specs, lookahead):
            print(f"Conflict ahead at {node} between R{i+1}/R{j+1}")
            ti = compute_arrival_times(specs[i]['path'], positions, speed, specs[i]['delays'])[specs[i]['path'].index(node)]
            tj = compute_arrival_times(specs[j]['path'], positions, speed, specs[j]['delays'])[specs[j]['path'].index(node)]
            rid = j if ti <= tj else i
            apply_delay(rid, node, specs, arts, speed, fps, delay)

        return artists

    ani = animation.FuncAnimation(fig, update, frames=maxf, interval=1000/fps, blit=False)
    plt.show()

if __name__ == '__main__':
    yaml_path = os.path.join('.', 'data', 'map.yaml')
    goals = [
        ['Park1','Start1','Spare1','Park1'],
        ['Park2','Start1','Spare1','Park2'],
        ['Park3','Start2','Spare2','Park3']
    ]
    colors = ['red','green','blue']
    animate(yaml_path, goals, colors)
