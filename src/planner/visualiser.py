import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import math
from planner.planner_core import replan_waiting_agents

def animate_paths(agents, positions, topo_map, get_orientation_from_map, fps=10, wait_time=3.0, show_legend=False, graph=None, save=False):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Multi-Robot Path Animation")
    ax.axis('equal')
    ax.grid(False)

    for entry in topo_map:
        node = entry['node']
        name = node['name']
        x, y = positions[name]
        ax.scatter(x, y, c='blue')
        ax.text(x + 0.1, y - 0.1, name, fontsize=9)
        for edge in node.get('edges', []):
            to_node = edge['node']
            if to_node in positions:
                x2, y2 = positions[to_node]
                ax.plot([x, x2], [y, y2], color='gray')
                dx, dy = x2 - x, y2 - y
                ax.add_patch(FancyArrowPatch((x + 0.25 * dx, y + 0.25 * dy),
                                             (x + 0.30 * dx, y + 0.30 * dy),
                                             arrowstyle='-|>', mutation_scale=15, color='gray'))

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    trails = []
    max_frames = 0

    for idx, agent in enumerate(agents):
        # Highlight start and goal with unfilled shapes
        if agent.start in positions:
            sx, sy = positions[agent.start]
            ax.plot(sx, sy, marker=(3, 0, 0), markerfacecolor='none', markeredgecolor=colors[idx % len(colors)], markersize=20)  # triangle
        if agent.goal and agent.goal in positions:
            gx, gy = positions[agent.goal]
            ax.plot(gx, gy, marker='s', markerfacecolor='none', markeredgecolor=colors[idx % len(colors)], markersize=13)  # square

        path = agent.full_route if agent.full_route else agent.route
        if not path:
            if agent.start in positions:
                x, y = positions[agent.start]
                ax.plot(x, y, 'o', color=colors[idx % len(colors)], markersize=8)
                ax.text(x + 0.2, y + 0.2, agent.name, fontsize=8, color=colors[idx % len(colors)])
            else:
                print(f"[!] Warning: Start node '{agent.start}' not found in positions for {agent.name}")
            continue
        if len(path) < 2:
            continue

        coords, angles = [], []
        for i in range(len(path) - 1):
            x1, y1 = positions[path[i]]
            x2, y2 = positions[path[i + 1]]
            steps = max(int(math.hypot(x2 - x1, y2 - y1) * fps), 1)

            for t in range(steps):
                alpha = t / steps
                x = (1 - alpha) * x1 + alpha * x2
                y = (1 - alpha) * y1 + alpha * y2
                coords.append((x, y))
                angles.append(math.atan2(y2 - y1, x2 - x1))
        coords.append(positions[path[-1]])
        final_angle = get_orientation_from_map(path[-1]) or angles[-1]
        while len(angles) < len(coords):
            angles.append(final_angle)

        if agent.wait_node:
            wait_pos = positions[agent.wait_node]
            try:
                wait_idx = next(i for i, p in enumerate(coords)
                                if math.isclose(p[0], wait_pos[0], abs_tol=1e-3)
                                and math.isclose(p[1], wait_pos[1], abs_tol=1e-3))
                
                # Estimate wait duration from timestamps in route
                if hasattr(agent, "route") and len(agent.route) >= 1:
                    last_leg = agent.route[-1]
                    wait_start = last_leg[3]  # arrival time at wait_node
                    next_start = agent.fragments[0][0][2] if agent.fragments and len(agent.fragments[0]) > 0 else wait_start + 1.0
                    wait_duration = max(0.0, next_start - wait_start)
                else:
                    wait_duration = wait_time  # fallback
                
                wait_frames = int(wait_duration * fps)
                coords[wait_idx:wait_idx] = [wait_pos] * wait_frames
                angles[wait_idx:wait_idx] = [angles[wait_idx]] * wait_frames
                print(f"[Visualizer] {agent.name} is set to wait at {agent.wait_node} for {wait_frames} frames (≈{wait_duration:.2f}s).")

            except StopIteration:
                print(f"[!] {agent.name}: Wait node {agent.wait_node} not found in coordinates.")


        xs, ys = zip(*[positions[n] for n in path if n in positions])
        ax.plot(xs, ys, '--', color=colors[idx % len(colors)], linewidth=2)
        dot, = ax.plot([], [], 'o', color=colors[idx % len(colors)], markersize=8)
        arrow = None
        trails.append({'coords': coords, 'angles': angles, 'dot': dot, 'arrow': arrow,
                       'color': colors[idx % len(colors)], 'agent': agent})
        agent.dynamic_coords = coords.copy()
        agent.dynamic_angles = angles.copy()
        max_frames = max(max_frames, len(coords))

    def update(frame):
        for trail in trails:
            print(f"Frame: {frame}")
            agent = trail['agent']
            idx = min(frame, len(agent.dynamic_coords) - 1)
            x, y = agent.dynamic_coords[idx]
            print(f"{agent.name} position at frame {frame}: {x}, {y}")
            coords = agent.dynamic_coords
            angles = agent.dynamic_angles

            if graph and agent.waiting and agent.wait_node and not agent.replanned:
                replan_waiting_agents([agent], graph, frame=frame, fps=fps)
                coords = agent.dynamic_coords
                angles = agent.dynamic_angles

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

    for idx, agent in enumerate(agents):
        if agent.goal is None and agent.start in positions:
            x, y = positions[agent.start]
            dot, = ax.plot([x], [y], 'o', color=colors[idx % len(colors)], markersize=8)
            ax.text(x + 0.2, y + 0.2, agent.name, fontsize=8, color=colors[idx % len(colors)])

    if show_legend:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=agent.name,
                   markerfacecolor=colors[idx % len(colors)], markersize=8)
            for idx, agent in enumerate(agents)
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=100, blit=True, repeat=False)

    if save:
        import os
        os.makedirs("output", exist_ok=True)
        ani.save("output/animation.gif", writer="pillow", fps=fps)
        print("[✓] Animation saved to output/animation.gif")
    else:
        plt.show()

