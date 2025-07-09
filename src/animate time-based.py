import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import math
import networkx as nx
from utils import build_graph_from_yaml, find_path
from matplotlib.patches import FancyArrowPatch

# === Multi-Robot Animation with Separate Timeline Figure ===

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


def compute_node_times(path, positions, speed_mps):
    times = [0.0]
    for i in range(len(path) - 1):
        x1, y1 = positions[path[i]]
        x2, y2 = positions[path[i+1]]
        dt = math.hypot(x2 - x1, y2 - y1) / speed_mps
        times.append(times[-1] + dt)
    return times


def interpolate_path(path, positions, speed_mps, fps, topo_map):
    points, angles = [], []
    for i in range(len(path) - 1):
        x1, y1 = positions[path[i]]
        x2, y2 = positions[path[i+1]]
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(int(dist / speed_mps * fps), 1)
        for t in range(steps):
            alpha = t / steps
            points.append(((1 - alpha) * x1 + alpha * x2,
                           (1 - alpha) * y1 + alpha * y2))
            angles.append(math.atan2(y2 - y1, x2 - x1))
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
    # Map figure
    fig_map, ax_map = plt.subplots(figsize=(8, 6))
    ax_map.set_title("Multi-Robot Path Animation")
    ax_map.axis('equal')
    ax_map.grid(False)

    # Timeline figure
    fig_time, ax_time = plt.subplots(figsize=(8, 3))
    ax_time.set_title("Robot Waypoint Timeline")
    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Waypoint Index')
    ax_time.grid(True)

    # Draw map nodes, edges, and highlight paths
    for item in topo_map:
        node = item['node']
        x, y = node['pose']['position']['x'], node['pose']['position']['y']
        ax_map.scatter(x, y, c='blue')
        ax_map.text(x + 0.1, y - 0.1, node['name'], fontsize=9, ha='left', va='top')
        for edge in node.get('edges', []):
            u, v = node['name'], edge['node']
            x1, y1 = positions[u]; x2, y2 = positions[v]
            ax_map.plot([x1, x2], [y1, y2], color='gray', linewidth=1)
            dx, dy = x2 - x1, y2 - y1
            ax_map.add_patch(
                FancyArrowPatch(
                    (x1 + 0.25*dx, y1 + 0.25*dy),
                    (x1 + 0.30*dx, y1 + 0.30*dy),
                    arrowstyle='-|>', mutation_scale=15, color='gray'
                )
            )

    # Prepare timeline static plot and data
    timeline_data = []
    for spec in robot_specs:
        times = compute_node_times(spec['path'], positions, speed_mps)
        nodes = spec['path']
        color = spec['color']
        ax_time.plot(times, list(range(len(nodes))), linestyle='--', marker='o', color=color)
        for t, idx in zip(times, range(len(nodes))):
            ax_time.text(t, idx, nodes[idx], color=color, va='bottom', ha='left')
        timeline_data.append((times, nodes, color))

    # Live time cursor
    time_cursor = ax_time.axvline(0, color='black')  # creates vertical line with two points

    # Highlight robot paths on map
    for spec in robot_specs:
        xs = [positions[n][0] for n in spec['path']]
        ys = [positions[n][1] for n in spec['path']]
        ax_map.plot(xs, ys, linestyle='--', color=spec['color'], linewidth=2)

    # Prepare robot markers on map
    robots = []
    max_frames = 0
    for spec in robot_specs:
        pts, angs = interpolate_path(spec['path'], positions, speed_mps, fps, topo_map)
        dot, = ax_map.plot([], [], 'o', color=spec['color'], markersize=8)
        robots.append({'points': pts, 'angles': angs, 'dot': dot, 'arrow': None})
        max_frames = max(max_frames, len(pts))

    # Update function animates both figures
    def update(frame):
        t = frame / fps
        time_cursor.set_xdata([t, t])  # update by providing sequence of x-values
        artists = [time_cursor]
        for r in robots:
            idx = min(frame, len(r['points']) - 1)
            x, y = r['points'][idx]; yaw = r['angles'][idx]
            dx, dy = 0.5 * math.cos(yaw), 0.5 * math.sin(yaw)
            r['dot'].set_data([x], [y])
            if r['arrow']:
                r['arrow'].remove()
            r['arrow'] = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='-|>',
                                         color=r['dot'].get_color(), mutation_scale=20)
            ax_map.add_patch(r['arrow'])
            artists += [r['dot'], r['arrow']]
        return artists

    # Run animation on map figure, timeline updates in callback
    ani = animation.FuncAnimation(fig_map, update, frames=max_frames,
                                  interval=1000/fps, blit=True, repeat=False)

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
            seg = find_path(graph, goals[i], goals[i+1], positions)
            if not seg:
                raise ValueError(f"No path from {goals[i]} to {goals[i+1]}")
            if i > 0: seg = seg[1:]
            full_path.extend(seg)
        robot_specs.append({'path': full_path, 'color': colors[idx % len(colors)]})

    animate_robots(robot_specs, positions, graph, speed_mps=1.0, fps=10, topo_map=topo_map)
