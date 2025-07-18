import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import math
import networkx as nx
from itertools import combinations
from utils import build_graph_from_yaml, find_path
from matplotlib.patches import FancyArrowPatch

# === Multi-Robot Animation with Enhanced Collision Detection & Timeline + Replanning ===

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

def compute_node_times(path, positions, speed_mps):
    times = [0.0]
    for i in range(len(path) - 1):
        x1, y1 = positions[path[i]]
        x2, y2 = positions[path[i+1]]
        dt = math.hypot(x2 - x1, y2 - y1) / speed_mps
        times.append(times[-1] + dt)
    return times

def detect_collisions(robot_specs, positions, speed_mps, delta_t):
    arrivals_list = [compute_node_times(spec['path'], positions, speed_mps) for spec in robot_specs]
    simulation_end = max(arr[-1] for arr in arrivals_list)
    specs_times = []
    for rid, spec in enumerate(robot_specs, start=1):
        arrivals = arrivals_list[rid-1]
        departures = arrivals[1:] + [simulation_end]
        specs_times.append((rid, spec['path'], arrivals, departures))

    node_occ = {}
    for rid, path, arr, dep in specs_times:
        for i, node in enumerate(path):
            node_occ.setdefault(node, []).append((rid, arr[i], dep[i]))

    collisions = []
    seen = set()
    for node, occs in node_occ.items():
        n = len(occs)
        for k in range(n, 1, -1):
            for combo in combinations(occs, k):
                present_ids = tuple(sorted(rid for rid, _, _ in combo))
                if len(present_ids) < 2:
                    continue
                starts = [a - delta_t for (_, a, _) in combo]
                ends   = [d + delta_t for (_, _, d) in combo]
                if max(starts) <= min(ends):
                    angles = []
                    for rid, _, _ in combo:
                        path = robot_specs[rid-1]['path']
                        idxs = [i for i, nm in enumerate(path) if nm == node]
                        if not idxs or idxs[0] == 0:
                            continue
                        prev_node = path[idxs[0] - 1]
                        x0, y0 = positions[prev_node]
                        x1, y1 = positions[node]
                        angles.append(math.atan2(y1 - y0, x1 - x0))
                    if len(angles) > 1:
                        base = angles[0]
                        if all(abs((a - base + math.pi) % (2*math.pi) - math.pi) < 0.2 for a in angles[1:]):
                            continue
                    collision_time = max(a for (_, a, _) in combo)
                    key = (present_ids, node, round(collision_time, 2))
                    if key not in seen:
                        seen.add(key)
                        collisions.append((list(present_ids), node, collision_time))
    return collisions

def interpolate_path(path, positions, speed_mps, fps, topo_map):
    points, angles = [] , []
    for i in range(len(path) - 1):
        x1, y1 = positions[path[i]]
        x2, y2 = positions[path[i+1]]
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(int(dist / speed_mps * fps), 1)
        for t in range(steps):
            alpha = t / steps
            x = (1 - alpha) * x1 + alpha * x2
            y = (1 - alpha) * y1 + alpha * y2
            points.append((x, y))
            angles.append(math.atan2(y2 - y1, x2 - x1))
    final = path[-1]
    points.append(positions[final])
    if is_fixed_orientation_node(final, topo_map):
        for item in topo_map:
            node = item['node']
            if node['name'] == final:
                q = node['pose']['orientation']
                angles.append(quaternion_to_yaw(q['z'], q['w']))
                break
    else:
        x1, y1 = positions[path[-2]]
        x2, y2 = positions[final]
        angles.append(math.atan2(y2 - y1, x2 - x1))
    return points, angles

def animate_robots(robot_specs, positions, graph, speed_mps=1.0, fps=30, topo_map=None, collision_delta_t=2.0):
    # Print paths
    for rid, spec in enumerate(robot_specs, start=1):
        print(f"Robot {rid} path: {spec['path']}")
        # Detect collisions
    collisions = detect_collisions(robot_specs, positions, speed_mps, collision_delta_t)
    # Apply wait-based policy: slower (longer-path) robot waits at the previous node of collision
    for robots, node, t in collisions:
        # pick robot with longer remaining path length
        rid_wait = max(robots, key=lambda rid: len(robot_specs[rid-1]['path']))
        spec = robot_specs[rid_wait-1]
        # find where collision occurs in its path
        path = spec['path']
        try:
            idx = path.index(node)
        except ValueError:
            wait_node = node
        else:
            # wait at previous node before collision
            wait_node = path[idx-1] if idx > 0 else node
        # record delay at wait_node
        spec.setdefault('delay_at', {})[wait_node] = collision_delta_t
        print(f"Robot {rid_wait} will wait {collision_delta_t}s at node '{wait_node}' before proceeding")

    # Setup map figure
    fig_map, ax_map = plt.subplots(figsize=(8, 6))
    ax_map.set_title("Multi-Robot Path Animation")
    ax_map.axis('equal')
    ax_map.grid(False)
    # Draw static map
    for item in topo_map:
        node = item['node']
        x, y = node['pose']['position']['x'], node['pose']['position']['y']
        ax_map.scatter(x, y, c='blue')
        ax_map.text(x+0.1, y-0.1, node['name'], fontsize=9, ha='left', va='top')
        for edge in node.get('edges', []):
            u, v = node['name'], edge['node']
            x1, y1 = positions[u]; x2, y2 = positions[v]
            ax_map.plot([x1, x2], [y1, y2], color='gray', linewidth=1)
            dx, dy = x2 - x1, y2 - y1
            ax_map.add_patch(FancyArrowPatch((x1+0.25*dx, y1+0.25*dy),
                                             (x1+0.30*dx, y1+0.30*dy),
                                             arrowstyle='-|>', mutation_scale=15, color='gray'))
    # Prepare robots for animation
    robots_art = []
    max_frames = 0
    for spec in robot_specs:
        # Interpolate raw path
        pts, angs = interpolate_path(spec['path'], positions, speed_mps, fps, topo_map)
        # Insert wait frames if needed
        delay_map = spec.get('delay_at', {})
        for node, delay in delay_map.items():
            pos = positions[node]
            # find first index where robot at this node
            try:
                idx = next(i for i, p in enumerate(pts) if math.isclose(p[0], pos[0], abs_tol=1e-6) and math.isclose(p[1], pos[1], abs_tol=1e-6))
            except StopIteration:
                continue
            wait_frames = int(delay * fps)
            # yaw at that position
            yaw = angs[idx]
            # insert duplicates
            pts[idx:idx] = [pos] * wait_frames
            angs[idx:idx] = [yaw] * wait_frames
        # plot full path trace
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax_map.plot(xs, ys, '--', color=spec['color'], linewidth=2)
        # create dot and arrow placeholders
        dot, = ax_map.plot([], [], 'o', color=spec['color'], markersize=8)
        robots_art.append({'points': pts, 'angles': angs, 'dot': dot, 'arrow': None})
        max_frames = max(max_frames, len(pts))



    # # Timeline plot
    # fig_time, ax_time = plt.subplots(figsize=(8, 3))
    # ax_time.set_title("Robot Timeline (Robot Index vs Time)")
    # ax_time.set_xlabel('Time (s)')
    # ax_time.set_ylabel('Robot')
    # ax_time.grid(True)
    # end_times = [compute_node_times(spec['path'], positions, speed_mps)[-1] for spec in robot_specs]
    # global_end = max(end_times)
    # for idx, spec in enumerate(robot_specs, start=1):
    #     times = compute_node_times(spec['path'], positions, speed_mps)
    #     ax_time.hlines(idx, times[0], global_end, colors=spec['color'], linewidth=2)
    #     ax_time.plot(times, [idx]*len(times), 'o', color=spec['color'])
    #     for t, node in zip(times, spec['path']):
    #         ax_time.text(t, idx+0.1, node, color=spec['color'], ha='left', va='bottom')
    # ax_time.set_yticks(range(1, len(robot_specs)+1))
    # ax_time.set_yticklabels([f'Robot{i}' for i in range(1, len(robot_specs)+1)])
    # for robots, node, t in collisions:
    #     for rid in robots:
    #         ax_time.plot(t, rid, 'X', color='red', markersize=12)
    #     ax_time.text(t, sum(robots)/len(robots) + 0.2, node, color='red', ha='center', va='bottom')




    # Animate update function
    def update(frame):
        artists = []
        for r in robots_art:
            idx = min(frame, len(r['points']) - 1)
            x, y = r['points'][idx]; yaw = r['angles'][idx]
            dx, dy = 0.5 * math.cos(yaw), 0.5 * math.sin(yaw)
            r['dot'].set_data([x], [y])
            if r['arrow']: r['arrow'].remove()
            r['arrow'] = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='-|>', mutation_scale=20, color=r['dot'].get_color())
            ax_map.add_patch(r['arrow'])
            artists += [r['dot'], r['arrow']]
        return artists

    ani = animation.FuncAnimation(fig_map, update, frames=max_frames, interval=1000/fps, blit=True, repeat=False)
    plt.show()

if __name__ == '__main__':
    yaml_path = os.path.join('.', 'data', 'map.yaml')
    topo_map = load_map(yaml_path)
    graph, positions = build_graph_from_yaml(topo_map)
    multi_robot_goals = [
        ['Park1', 'Start1', 'T00', 'Spare1', 'Park1'],
        ['Park2', 'Start1', 'T10', 'Spare1', 'Park2'],
        ['Park3', 'Start2', 'T40', 'Spare2', 'Park3']
    ]
    colors = ['red', 'green', 'blue']
    robot_specs = []
    for idx, goals in enumerate(multi_robot_goals, start=1):
        full = []
        for i in range(len(goals)-1):
            seg = find_path(graph, goals[i], goals[i+1], positions)
            if not seg:
                raise ValueError(f"No path from {goals[i]} to {goals[i+1]}")
            if i>0: seg=seg[1:]
            full.extend(seg)
        robot_specs.append({'path': full, 'color': colors[(idx-1)%len(colors)]})
    animate_robots(robot_specs, positions, graph, speed_mps=1.0, fps=10, topo_map=topo_map, collision_delta_t=2.0)
    
    