# planner/visualiser.py
import math, os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

# Use the new planner with frame/positions-based replan
try:
    from .fragment_planner import replan_waiting_agents
except ImportError:
    from planner.fragment_planner import replan_waiting_agents


def animate_paths(
    agents,
    positions,
    topo_map,
    get_orientation_from_map,
    fps=10,
    wait_time=3.0,       # also used as a small release buffer (in seconds)
    show_legend=False,
    graph=None,
    save=False,
    _second_run=False,   # internal to prevent infinite restarts
    freeze_agents_not_in=None,
):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Multi-Robot Path Animation")
    ax.axis("equal")
    ax.grid(False)

    # --- draw map
    for entry in topo_map:
        node = entry["node"]; name = node["name"]
        x, y = positions[name]
        ax.scatter(x, y, c="blue")
        ax.text(x + 0.1, y - 0.1, name, fontsize=9)
        for edge in node.get("edges", []):
            to_node = edge["node"]
            if to_node in positions:
                x2, y2 = positions[to_node]
                ax.plot([x, x2], [y, y2], color="gray", linewidth=1)
                dx, dy = x2 - x, y2 - y
                ax.add_patch(FancyArrowPatch(
                    (x + 0.25 * dx, y + 0.25 * dy),
                    (x + 0.30 * dx, y + 0.30 * dy),
                    arrowstyle="-|>", mutation_scale=15, color="gray"
                ))

    colors = ["red", "blue", "green", "orange", "purple"]
    trails, max_frames = [], 0

    # --- build per-frame coords from full_route
    def rebuild_dynamic_traj(agent, positions, get_orientation_from_map, fps=10):
        path = agent.full_route if getattr(agent, "full_route", None) else []
        if not path or len(path) < 2:
            agent.dynamic_coords = [positions[path[0]]] if path else []
            agent.dynamic_angles = [0.0] if path else []
            return
        coords, angles = [], []
        for i in range(len(path) - 1):
            x1, y1 = positions[path[i]]; x2, y2 = positions[path[i + 1]]
            steps = max(int(math.hypot(x2 - x1, y2 - y1) * fps), 1)
            for t in range(steps):
                a = t / steps
                x = (1 - a) * x1 + a * x2
                y = (1 - a) * y1 + a * y2
                coords.append((x, y))
                angles.append(math.atan2(y2 - y1, x2 - x1))
        coords.append(positions[path[-1]])
        final_angle = get_orientation_from_map(path[-1]) or (angles[-1] if angles else 0.0)
        while len(angles) < len(coords):
            angles.append(final_angle)
        agent.dynamic_coords, agent.dynamic_angles = coords, angles
    
    def rebuild_from_node_list(agent, node_path, positions, get_orientation_from_map, fps=10):
        if not node_path or len(node_path) < 2:
            agent.dynamic_coords = [positions[node_path[0]]] if node_path else []
            agent.dynamic_angles = [0.0] if node_path else []
            return
        coords, angles = [], []
        for i in range(len(node_path) - 1):
            x1, y1 = positions[node_path[i]]; x2, y2 = positions[node_path[i + 1]]
            steps = max(int(math.hypot(x2 - x1, y2 - y1) * fps), 1)
            for t in range(steps):
                a = t / steps
                x = (1 - a) * x1 + a * x2
                y = (1 - a) * y1 + a * y2
                coords.append((x, y))
                angles.append(math.atan2(y2 - y1, x2 - x1))
        coords.append(positions[node_path[-1]])
        final_angle = get_orientation_from_map(node_path[-1]) or (angles[-1] if angles else 0.0)
        while len(angles) < len(coords):
            angles.append(final_angle)
        agent.dynamic_coords, agent.dynamic_angles = coords, angles


    # initial trails (dash intended routes)
    for idx, agent in enumerate(agents):
        if agent.start in positions:
            sx, sy = positions[agent.start]
            ax.plot(sx, sy, marker=(3, 0, 0), markerfacecolor="none",
                    markeredgecolor=colors[idx % len(colors)], markersize=20)
        if getattr(agent, "goal", None) and agent.goal in positions:
            gx, gy = positions[agent.goal]
            ax.plot(gx, gy, marker="s", markerfacecolor="none",
                    markeredgecolor=colors[idx % len(colors)], markersize=13)

        path = agent.full_route if getattr(agent, "full_route", None) else agent.route
        if path and len(path) >= 2:
            xs, ys = zip(*[positions[n] for n in (path if isinstance(path[0], str)
                       else [path[0][0]] + [v for _, v in path]) if n in positions])
            ax.plot(xs, ys, "--", color=colors[idx % len(colors)], linewidth=2)

        rebuild_dynamic_traj(agent, positions, get_orientation_from_map, fps=fps)

        # In second window: freeze everyone not in the movers set,
        # and animate only the last fragment of the movers.
        if _second_run and freeze_agents_not_in is not None:
            if agents[idx].name not in freeze_agents_not_in:
                # Freeze at goal: show a single frame at final position
                if agents[idx].full_route:
                    goal_node = agents[idx].full_route[-1]
                    agent.dynamic_coords = [positions[goal_node]]
                    agent.dynamic_angles = [get_orientation_from_map(goal_node) or 0.0]
            else:
                # Animate only the last fragment (the resumed bit), not the whole path
                last_frag = agents[idx].fragments[-1] if getattr(agents[idx], "fragments", None) else None
                if last_frag:
                    node_path = [last_frag[0][0]] + [v for (_, v) in last_frag]
                    rebuild_from_node_list(agents[idx], node_path, positions, get_orientation_from_map, fps=fps)
                else:
                    # fallback: animate from wait_node to goal if present
                    if getattr(agent, "wait_node", None) and agent.full_route:
                        try:
                            start_idx = agent.full_route.index(agent.wait_node)
                            tail = agent.full_route[start_idx:]
                            rebuild_from_node_list(agent, tail, positions, get_orientation_from_map, fps=fps)
                        except ValueError:
                            pass


        dot, = ax.plot([], [], marker="o", linestyle="None",
               markerfacecolor=colors[idx % len(colors)],
               markeredgecolor="k", markersize=12, zorder=6)
        trails.append({"coords": agent.dynamic_coords,
                       "angles": agent.dynamic_angles,
                       "dot": dot, "arrow": None,
                       "color": colors[idx % len(colors)], "agent": agent})
        max_frames = max(max_frames, len(agent.dynamic_coords))

    # Debug: how many frames each agent has initially
    for t in trails:
        print(f"[Viz:init] {t['agent'].name} dynamic frames: {len(t['coords'])}")
    agent_to_trail = {t["agent"]: t for t in trails}
    second_window_spawned = False  # prevent endless spawns


    trigger_agent_name = "Robot1"  # the agent whose arrival triggers the replan
    replan_banner_printed = False  # "[→] Replan loop 1..."
    replan_success_printed = False # "[✓] One or more waiting agents..."



    

    def make_after_animation_summary(agents):
        lines = ["==== After Animation ====", ""]
        for a in agents:
            flags = []
            if getattr(a, "active", True): flags.append("active")
            if getattr(a, "finished", False): flags.append("finished")
            if getattr(a, "replanned", False): flags.append("replanned")
            status = ", ".join(flags) if flags else "idle"
            frag = None
            if getattr(a, "fragments", None):
                for f in a.fragments:
                    if f:
                        frag = f; break
            if not frag:
                frag = a.route if a.route else [a.start]
            lines.append(f"Agent: {a.name} [{status}]")
            lines.append("  Fragment: " + " -> ".join([frag[0][0]] + [v for _, v in frag]) if frag and isinstance(frag[0], tuple)
                         else "  Fragment: " + " -> ".join(frag))
            full = getattr(a, "full_route", []) or [a.start]
            lines.append(f"  Full route: {full}")
            lines.append("")
        return "\n".join(lines)

    def update(frame):
        nonlocal replan_banner_printed, replan_success_printed, second_window_spawned
        artists = []
        # --- draw agents
        for trail in trails:
            if not trail["coords"]:
                continue
            i = min(frame, len(trail["coords"]) - 1)
            x, y = trail["coords"][i]
            yaw = trail["angles"][i]
            dx, dy = 0.5 * math.cos(yaw), 0.5 * math.sin(yaw)

            trail["dot"].set_data([x], [y])
            if trail["arrow"]:
                trail["arrow"].remove()
            trail["arrow"] = FancyArrowPatch((x, y), (x + dx, y + dy),
                                            arrowstyle="-|>", mutation_scale=20,
                                            color=trail["color"])
            ax.add_patch(trail["arrow"])
            artists.extend([trail["dot"], trail["arrow"]])

        # --- detect when Robot1 (trigger) reaches end of its animated path
        trig = next((a for a in agents if a.name == trigger_agent_name), None)
        at_end = False
        if trig:
            tr = agent_to_trail.get(trig)
            if tr and tr["coords"]:
                at_end = frame >= len(tr["coords"]) - 1
                if at_end:
                    trig.finished = True

        # --- Replan ONCE and print the three lines you want (during animation)
        if graph and at_end and not replan_success_printed:
            if not replan_banner_printed:
                print("\n[→] Replan loop 1: attempting to resume all waiting agents...")
                replan_banner_printed = True

            resumed = replan_waiting_agents(
                agents, graph,
                frame=frame,
                positions=positions,                    # <-- IMPORTANT
                release_delay_frames=int(wait_time*fps) # small buffer
            )

            if resumed:
                # capture movers BEFORE clearing the flag
                movers = [a.name for a in agents if getattr(a, "replanned", False)]

                # print per-agent resumed line (no rebuild here; keep first window “as-is”)
                for a in agents:
                    if getattr(a, "replanned", False):
                        print(f"[Replan] {a.name} resumed: {a.full_route}")

                print("[✓] One or more waiting agents successfully replanned.")
                replan_success_printed = True

                # keep first window open; spawn ONE second window where only movers animate
                if not _second_run and not second_window_spawned:
                    second_window_spawned = True
                    tm = fig.canvas.new_timer(interval=400)
                    tm.single_shot = True           # <-- important: fire once
                    def _spawn_second():
                        # clear replanned flags now that we've captured movers
                        for a in agents:
                            if getattr(a, "replanned", False):
                                a.replanned = False
                        animate_paths(
                            agents, positions, topo_map, get_orientation_from_map,
                            fps=fps, wait_time=wait_time, show_legend=show_legend,
                            graph=None,                         # <-- optional: disable replan in 2nd window
                            save=save, _second_run=True,
                            freeze_agents_not_in=set(movers),
                        )
                    tm.add_callback(_spawn_second)
                    tm.start()
                    fig._spawn_timer = tm            # <-- keep a strong reference



                    return artists


    if show_legend:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=a.name,
                   markerfacecolor=colors[i % len(colors)], markersize=8)
            for i, a in enumerate(agents)
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    def init():
        artists = []
        for t in trails:
            t["dot"].set_data([], [])
            if t["arrow"]:
                t["arrow"].remove()
                t["arrow"] = None
            artists.append(t["dot"])
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=max_frames if max_frames > 0 else 1,
        interval=100, blit=False, repeat=False, init_func=init,
    ) 
    fig.ani = ani

    if save:
        os.makedirs("output", exist_ok=True)
        ani.save("output/animation.gif", writer="pillow", fps=fps)
        print("[✓] Animation saved to output/animation.gif")
    else:
        plt.show()
