# planner/visualiser.py
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import os

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
    wait_time=0.0,       # small release buffer (seconds) after arrival
    show_legend=False,
    graph=None,
    save=False,
    verbose_snapshots=False,
):
    # Arrival tolerance in map units — lets us treat "close enough" as arrived
    reach_tol = 0.25

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Multi-Robot Path Animation")
    ax.axis("equal")
    ax.grid(False)

    # --- draw map
    for entry in topo_map:
        node = entry["node"]
        name = node["name"]
        x, y = positions[name]
        ax.scatter(x, y, c="blue", zorder=1)
        ax.text(x + 0.1, y - 0.1, name, fontsize=9)
        for edge in node.get("edges", []):
            to_node = edge["node"]
            if to_node in positions:
                x2, y2 = positions[to_node]
                ax.plot([x, x2], [y, y2], color="gray", linewidth=1, zorder=1)
                dx, dy = x2 - x, y2 - y
                ax.add_patch(FancyArrowPatch(
                    (x + 0.25 * dx, y + 0.25 * dy),
                    (x + 0.30 * dx, y + 0.30 * dy),
                    arrowstyle="-|>", mutation_scale=15, color="gray", zorder=1
                ))

    colors = ["red", "blue", "green", "orange", "purple", "brown", "magenta", "teal"]
    trails, max_frames = [], 0

    # --- build per-frame coords from full_route
    def rebuild_dynamic_traj(agent, positions, get_orientation_from_map, fps=10, start_index=0, resume_xy=None):
        """
        Build per-frame coords that exactly follow agent.full_route[start_index:].
        If resume_xy is provided, the first sample of the new trajectory starts
        at that XY (visual continuity), then heads to the next waypoint.
        """
        path = agent.full_route if getattr(agent, "full_route", None) else []
        agent._node_frame_cache = {}

        # clamp defensive
        if not path:
            agent.dynamic_coords = []
            agent.dynamic_angles = []
            return

        if start_index >= len(path) - 1:
            # no edges left; pin at last node
            last = path[-1]
            xy = resume_xy if resume_xy is not None else positions[last]
            agent.dynamic_coords = [xy]
            agent.dynamic_angles = [get_orientation_from_map(last) or 0.0]
            agent._node_frame_cache[last] = 0
            return

        path = path[start_index:]  # suffix we’ll animate

        coords, angles = [], []

        # starting sample: current XY if supplied, otherwise node coord
        x0, y0 = (resume_xy if resume_xy is not None else positions[path[0]])
        coords.append((x0, y0))
        # set a reasonable initial heading (toward the next waypoint)
        x1, y1 = positions[path[0]]
        x2, y2 = positions[path[1]]
        init_yaw = math.atan2(y2 - y1, x2 - x1)
        angles.append(init_yaw)
        agent._node_frame_cache[path[0]] = 0

                # interpolate each edge in the suffix
        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]

            # FIRST segment after a replan: go from resume_xy → n2
            if i == 0 and resume_xy is not None:
                sx, sy = x0, y0                     # current on-screen XY
                ex, ey = positions[n2]
            else:
                sx, sy = positions[n1]
                ex, ey = positions[n2]

            steps = max(int(math.hypot(ex - sx, ey - sy) * fps), 1)

            # interpolate smoothly from (sx,sy) to (ex,ey)
            for t in range(1, steps):
                a = t / steps
                x = (1 - a) * sx + a * ex
                y = (1 - a) * sy + a * ey
                coords.append((x, y))
                angles.append(math.atan2(ey - sy, ex - sx))

            coords.append((ex, ey))
            angles.append(math.atan2(ey - sy, ex - sx))
            agent._node_frame_cache[n2] = len(coords) - 1


        # orient final node nicely if map has orientation
        final_angle = get_orientation_from_map(path[-1]) or (angles[-1] if angles else 0.0)
        if angles:
            angles[-1] = final_angle

        agent.dynamic_coords, agent.dynamic_angles = coords, angles



    # --- initial trails (dash intended routes)
    for idx, agent in enumerate(agents):
        if agent.start in positions:
            sx, sy = positions[agent.start]
            ax.plot(sx, sy, marker=(3, 0, 0), markerfacecolor="none",
                    markeredgecolor=colors[idx % len(colors)], markersize=20, zorder=3)
        if getattr(agent, "goal", None) and agent.goal in positions:
            gx, gy = positions[agent.goal]
            ax.plot(gx, gy, marker="s", markerfacecolor="none",
                    markeredgecolor=colors[idx % len(colors)], markersize=13, zorder=3)

        # dashed intended path (store handle so we can update it after replans)
        path = agent.full_route if getattr(agent, "full_route", None) else agent.route
        if path and len(path) >= 2:
            if isinstance(path[0], str):
                node_list = path
            else:
                node_list = [path[0][0]] + [v for _, v in path]
            xs, ys = zip(*[positions[n] for n in node_list if n in positions])
        else:
            xs, ys = [], []
        path_line, = ax.plot(xs, ys, "--",
                             color=colors[idx % len(colors)],
                             linewidth=2, alpha=0.9, zorder=2)

        rebuild_dynamic_traj(agent, positions, get_orientation_from_map, fps=fps)

        dot, = ax.plot([], [], marker="o", linestyle="None",
                       markerfacecolor=colors[idx % len(colors)],
                       markeredgecolor="k", markersize=12, zorder=6)

        # Create ONE arrow patch per agent and keep it; start hidden
        arrow = FancyArrowPatch((0, 0), (0, 0),
                                arrowstyle="-|>", mutation_scale=20,
                                color=colors[idx % len(colors)], zorder=6)
        arrow.set_visible(False)
        ax.add_patch(arrow)

        trails.append({
            "coords": agent.dynamic_coords,
            "angles": agent.dynamic_angles,
            "dot": dot,
            "arrow": arrow,                    # persistent arrow (no duplicates)
            "color": colors[idx % len(colors)],
            "agent": agent,
            "start_frame": 0,
            "path_line": path_line,            # dashed intended path (kept in sync)
        })

        max_frames = max(max_frames, len(agent.dynamic_coords))

    # for t in trails:
    #     print(f"[Viz:init] {t['agent'].name} dynamic frames: {len(t['coords'])}")
    agent_to_trail = {t["agent"]: t for t in trails}
    for a in agents:
        a._last_route_index = 0

    replan_banner_printed = False
    first_replan_done = False

    # periodic polling controls
    replan_poll_every = max(1, int(0.5 * fps))  # ~ every 0.5s
    last_poll_frame = -10**9

    def _refresh_dashed_path(tr, agent):
        """Update dashed intended path line to match agent.full_route after a replan."""
        new_path = agent.full_route if getattr(agent, "full_route", None) else []
        if new_path and len(new_path) >= 2:
            xs, ys = zip(*[positions[n] for n in new_path if n in positions])
        else:
            xs, ys = [], []
        if tr and tr.get("path_line"):
            tr["path_line"].set_data(xs, ys)

    def update(frame):
        nonlocal replan_banner_printed, first_replan_done, last_poll_frame

        artists = []

        # accumulate waiting time per agent (in frames)
        for a in agents:
            if getattr(a, "waiting", False):
                a.wait_frames = getattr(a, "wait_frames", 0) + 1
        

        # ---- draw agents (reuse one arrow per agent)
        for trail in trails:
            if not trail["coords"]:
                continue
            i = max(0, min(frame - trail["start_frame"], len(trail["coords"]) - 1))
            x, y = trail["coords"][i]
            yaw = trail["angles"][i]
            dx, dy = 0.5 * math.cos(yaw), 0.5 * math.sin(yaw)

            trail["dot"].set_data([x], [y])
            arrow = trail["arrow"]
            arrow.set_visible(True)
            arrow.set_positions((x, y), (x + dx, y + dy))
            artists.extend([trail["dot"], arrow])

        # ---- progress tracker: advance _last_route_index as we cross nodes
        for a in agents:
            fr = getattr(a, "full_route", None)
            tr = agent_to_trail.get(a)
            if not fr or not tr or not tr["coords"]:
                continue
            local_i = max(0, min(frame - tr.get("start_frame", 0), len(tr["coords"]) - 1))
            x, y = tr["coords"][local_i]
            last_idx = getattr(a, "_last_route_index", 0)
            j = last_idx
            while j < len(fr):
                nx, ny = positions[fr[j]]
                if (x - nx) ** 2 + (y - ny) ** 2 <= reach_tol ** 2:
                    j += 1
                else:
                    break
            if j > 0:
                a._last_route_index = max(last_idx, j - 1)

        # ---- detect goal reached (for nice bookkeeping)
        for a in agents:
            if getattr(a, "goal", None) and not getattr(a, "finished", False):
                tr = agent_to_trail.get(a)
                if tr and tr["coords"]:
                    gx, gy = positions[a.goal]
                    local_i = max(0, min(frame - tr.get("start_frame", 0), len(tr["coords"]) - 1))
                    x, y = tr["coords"][local_i]
                    if (x - gx) ** 2 + (y - gy) ** 2 <= reach_tol ** 2:
                        a.finished = True
                        if not hasattr(a, "finished_frame"):
                            a.finished_frame = frame

        # ---- trigger replan when ANY agent completes its current fragment
        fragment_completed = False
        for a in agents:
            endn = getattr(a, "fragment_end_node", None)
            if not endn:
                continue
            tr = agent_to_trail.get(a)
            if not tr or not tr["coords"]:
                continue
            local_i = max(0, min(frame - tr.get("start_frame", 0), len(tr["coords"]) - 1))
            x, y = tr["coords"][local_i]
            ex, ey = positions[endn]
            if (x - ex) ** 2 + (y - ey) ** 2 <= reach_tol ** 2:
                a.fragment_end_node = None  # consume once
                fragment_completed = True

        # Helper: apply a resumed plan with no jumps
        def _apply_resume(a):
            # Current XY on screen
            old_tr = agent_to_trail.get(a)
            resume_xy = None
            if old_tr and old_tr["coords"]:
                idx_old = max(0, min(frame - old_tr.get("start_frame", 0), len(old_tr["coords"]) - 1))
                resume_xy = old_tr["coords"][idx_old]

            fr = getattr(a, "full_route", None) or []
            if not fr:
                a.replanned = False
                return

            # Choose start index by nearest route node to current XY, but never go backwards
            if resume_xy is not None:
                def d2(i):
                    px, py = positions[fr[i]]
                    return (px - resume_xy[0]) ** 2 + (py - resume_xy[1]) ** 2
                nearest_idx = min(range(len(fr)), key=d2)
            else:
                nearest_idx = getattr(a, "_last_route_index", 0)

            start_idx = max(getattr(a, "_last_route_index", 0), nearest_idx)

            # Rebuild from CURRENT XY toward fr[start_idx:]
            rebuild_dynamic_traj(
                a, positions, get_orientation_from_map,
                fps=fps, start_index=start_idx, resume_xy=resume_xy
            )

            # Install new trajectory & refresh dashed intended path
            tr = agent_to_trail.get(a)
            if tr:
                tr["coords"] = a.dynamic_coords
                tr["angles"] = a.dynamic_angles
                tr["start_frame"] = frame
                if tr.get("path_line"):
                    xs, ys = (zip(*[positions[n] for n in fr if n in positions]) if fr else ([], []))
                    tr["path_line"].set_data(xs, ys)

            a.replanned = False  # clear flag

        # ---- run replan immediately after a fragment completes
        if graph and fragment_completed:
            if not replan_banner_printed:
                print("\n[→] Replan: fragment completed — resuming waiters if possible...\n")
                replan_banner_printed = True

            start_frames = {t["agent"].name: t.get("start_frame", 0) for t in trails}
            resumed = replan_waiting_agents(
                agents, graph,
                frame=frame,
                positions=positions,
                release_delay_frames=int(wait_time * fps),
                start_frames=start_frames,
                reach_tol=reach_tol,
                verbose_snapshots=False,
            )
            if resumed:
                for a in agents:
                    if getattr(a, "replanned", False):
                        print(f"[Replan] {a.name} resumed: {a.full_route}")
                        _apply_resume(a)
                print("[✓] One or more waiting agents successfully replanned.\n")
                last_poll_frame = frame

        # ---- periodic poll: owners may free a gate node without a fragment finishing
        if graph and any(getattr(a, "waiting", False) for a in agents):
            if frame - last_poll_frame >= replan_poll_every:
                last_poll_frame = frame
                start_frames = {t["agent"].name: t.get("start_frame", 0) for t in trails}
                resumed = replan_waiting_agents(
                    agents, graph,
                    frame=frame,
                    positions=positions,
                    release_delay_frames=int(wait_time * fps),
                    start_frames=start_frames,
                    reach_tol=reach_tol,
                )
                if resumed:
                    for a in agents:
                        if getattr(a, "replanned", False):
                            print(f"[Replan] {a.name} resumed: {a.full_route}")
                            _apply_resume(a)
                    print("[✓] One or more waiting agents successfully replanned.\n")

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
                # keep the arrow instance; just hide it initially
                t["arrow"].set_visible(False)
                artists.append(t["arrow"])
            artists.append(t["dot"])
        return artists

    ani = animation.FuncAnimation(
        fig, update,
        frames=max(1, int(180 * fps)),   # ~3 minutes of animation time
        interval=100, blit=False, repeat=False, init_func=init,
    )

    fig.ani = ani  # keep a strong ref

    
    if save:
        try:
            os.makedirs("output", exist_ok=True)
            out_path = os.path.abspath(os.path.join("output", "animation.gif"))
            print(f"[i] Saving animation to {out_path} ...")
            ani.save(out_path, writer="pillow", fps=fps)
            print(f"[✓] Animation saved to {out_path}")
        except Exception as e:
            print(f"[x] Failed to save animation: {type(e).__name__}: {e}")
    else:
        plt.show()

