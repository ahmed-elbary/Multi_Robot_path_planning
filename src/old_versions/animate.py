import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import math
import networkx as nx
from utils import build_graph_from_yaml, find_path
from matplotlib.patches import FancyArrowPatch

# === Multi-Robot Animation with Variable Robot Count & Multi-Waypoint Goals ===

def load_map(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def quaternion_to_yaw(z, w):
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * (z ** 2))


def is_fixed_orientation_node(node_name, topo_map):
    for item in topo_map:
        node = item['node']
        if node['name'] == node_name:
            if node.get('custom_type') == 'fixed_orientation':
                return True
            pose = node.get('pose', {})
            if pose.get('custom_type') == 'fixed_orientation':
                return True
    return False


def get_yaw(node_name, topo_map):
    for item in topo_map:
        node = item['node']
        if node['name'] == node_name:
            q = node['pose']['orientation']
            return quaternion_to_yaw(q['z'], q['w'])
    return 0.0


def interpolate_path(path, positions, speed_mps, fps, topo_map):
    points, angles = [], []
    for i in range(len(path) - 1):
        x1, y1 = positions[path[i]]
        x2, y2 = positions[path[i+1]]
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(int(dist / speed_mps * fps), 1)
        for t in range(steps):
            alpha = t / steps
            x = (1 - alpha) * x1 + alpha * x2
            y = (1 - alpha) * y1 + alpha * y2
            yaw = math.atan2(y2 - y1, x2 - x1)
            points.append((x, y))
            angles.append(yaw)
    final = path[-1]
    points.append(positions[final])
    if is_fixed_orientation_node(final, topo_map):
        angles.append(get_yaw(final, topo_map))
    else:
        x1, y1 = positions[path[-2]]
        x2, y2 = positions[final]
        angles.append(math.atan2(y2 - y1, x2 - x1))
    return points, angles


def animate_robots(robot_specs, positions, graph, speed_mps=1.0, fps=30, topo_map=None):
    fig, ax = plt.subplots()
    ax.set_title("Multi-Robot Path Animation")
    ax.axis('equal')
    ax.grid(False)

    # fraction parameters for edge arrows
    arrow_pos_frac = 0.25   # arrow placed at 25% of edge length from source
    arrow_head_frac = 0.05  # arrow head length as fraction of edge length

    # draw map nodes and edges
    for item in topo_map:
        node = item['node']
        x, y = node['pose']['position']['x'], node['pose']['position']['y']
        ax.scatter(x, y, c='blue')
        ax.text(x + 0.1, y, node['name'], fontsize=9)
        for edge in node.get('edges', []):
            u, v = node['name'], edge['node']
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            ax.plot([x1, x2], [y1, y2], color='gray', linewidth=1)
            # compute normalized direction for edge arrow
            dx, dy = x2 - x1, y2 - y1
            dist = math.hypot(dx, dy)
            if dist > 0:
                # arrow at fractional position
                base_x = x1 + arrow_pos_frac * dx
                base_y = y1 + arrow_pos_frac * dy
                tip_x = x1 + (arrow_pos_frac + arrow_head_frac) * dx
                tip_y = y1 + (arrow_pos_frac + arrow_head_frac) * dy
                arrow = FancyArrowPatch((base_x, base_y), (tip_x, tip_y), arrowstyle='-|>', mutation_scale=15, color='gray')
                ax.add_patch(arrow)

    # highlight each robot's full path
    for spec in robot_specs:
        path = spec['path']
        color = spec.get('color', 'red')
        xs = [positions[node][0] for node in path]
        ys = [positions[node][1] for node in path]
        ax.plot(xs, ys, linestyle='--', color=color, linewidth=2)

    # prepare robots for animation
    robots = []
    max_frames = 0
    for spec in robot_specs:
        color = spec.get('color', 'red')
        pts, angs = interpolate_path(spec['path'], positions, speed_mps, fps, topo_map)
        dot, = ax.plot([], [], 'o', color=color, markersize=8)
        robots.append({'points': pts, 'angles': angs, 'dot': dot, 'arrow': None, 'color': color})
        max_frames = max(max_frames, len(pts))

    def update(frame):
        patches = []
        for r in robots:
            idx = frame if frame < len(r['points']) else len(r['points']) - 1
            x, y = r['points'][idx]
            yaw = r['angles'][idx]
            dx = 0.5 * math.cos(yaw)
            dy = 0.5 * math.sin(yaw)
            r['dot'].set_data([x], [y])
            if r['arrow']:
                r['arrow'].remove()
            r['arrow'] = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='-|>', color=r['color'], mutation_scale=20)
            ax.add_patch(r['arrow'])
            patches += [r['dot'], r['arrow']]
        return patches

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=1000/fps, blit=True, repeat=False)
    plt.show()

if __name__ == '__main__':
    yaml_path = os.path.join('.', 'data', 'map.yaml')
    topo_map = load_map(yaml_path)
    graph, positions = build_graph_from_yaml(topo_map)

    # Configure each robot's waypoint sequence
    multi_robot_goals = [
        ['Park1', 'Start1', 'T00', 'Spare1', 'Park1'],
        ['Park2', 'Start1', 'T20', 'Spare2', 'Park2'],
        ['Park3', 'Spare2', 'Park3']
    ]
    colors = ['red', 'green', 'blue']

    robot_specs = []
    for idx, goals in enumerate(multi_robot_goals):
        full_path = []
        for i in range(len(goals) - 1):
            segment = find_path(graph, goals[i], goals[i+1], positions)
            if not segment:
                raise ValueError(f"No path from {goals[i]} to {goals[i+1]}")
            if i > 0:
                segment = segment[1:]
            full_path.extend(segment)
        robot_specs.append({'path': full_path, 'color': colors[idx % len(colors)]})

    animate_robots(robot_specs, positions, graph, speed_mps=1.0, fps=10, topo_map=topo_map)
