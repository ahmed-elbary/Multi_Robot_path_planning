# src/space_time_planner/visualiser_st.py
import math
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch


def _iter_node_entries(topo_map):
    """Return an iterable of node entries for both schemas:
       - small maps: a list of {node:{...}} or { ... } per entry
       - big maps : a dict with key 'nodes' holding that list
    """
    return topo_map.get("nodes", topo_map) if isinstance(topo_map, dict) else topo_map

def _node_dict(entry):
    """Return the node dict from an entry that may be {node:{...}} or already {...}."""
    return entry.get("node", entry)

def animate_paths(
    agents,
    positions,
    topo_map,
    get_orientation_from_map,
    fps=10,
    show_legend=False,
    graph=None,
    save=False,
    planner=None,   # SpaceTimePlanner instance (REQUIRED)
    wait_time=0,    # seconds of optional release delay
    live_timeline=True,    # print second-by-second while animating
):
    assert planner is not None, "planner (SpaceTimePlanner) is required"

    last_printed_second = -1
    timeline_done = False  # stop printing once horizon reached

    def _horizon_sec():
        """Latest arrival time among all booked edges, rounded up to seconds."""
        m = 0.0
        for a in agents:
            for _, _, _, ta in (getattr(a, "route", []) or []):
                if ta > m:
                    m = ta
        return int(math.ceil(m))

    def _all_finished():
        return all(getattr(a, "finished", False) for a in agents)

    def _state_at(a, t):
        """('moving', u, v, prog, dur) or ('waiting', node)."""
        r = getattr(a, "route", []) or []
        if not r:
            return ('waiting', getattr(a, 'start', None))
        for (u, v, td, ta) in r:
            if td <= t < ta:
                return ('moving', u, v, t - td, ta - td)
        if t < r[0][2]:
            return ('waiting', r[0][0])
        last = r[0][0]
        for (u, v, td, ta) in r:
            if ta <= t:
                last = v
            else:
                break
        return ('waiting', last)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Multi-Robot (Space–Time) — Animation")
    ax.axis("equal")
    ax.grid(False)

    # ---- draw map (keep baseline look)
    for entry in _iter_node_entries(topo_map):
        node = _node_dict(entry)
        name = node.get("name")
        if name not in positions:
            continue  # skip nodes not present in the graph/positions

        x, y = positions[name]
        ax.scatter(x, y, c="blue", zorder=1)
        ax.text(x + 0.1, y - 0.1, name, fontsize=9)

        for edge in (node.get("edges") or []):
            to_node = edge.get("node")
            if not to_node or to_node not in positions:
                continue
            x2, y2 = positions[to_node]
            ax.plot([x, x2], [y, y2], color="gray", linewidth=1, zorder=1)
            dx, dy = x2 - x, y2 - y
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.25 * dx, y + 0.25 * dy),
                    (x + 0.30 * dx, y + 0.30 * dy),
                    arrowstyle="-|>", mutation_scale=15, color="gray", zorder=1
                )
            )


    colors = ["red", "blue", "green", "orange", "purple", "brown", "magenta", "teal"]
    trails = []

    # ---------- helpers ----------
    def build_time_synced_traj(agent):
        """
        Build per-frame coords/angles from agent.route using *true* depart/arrive times.
        Any time gap between two edges becomes a visible pause at the node.
        """
        r = getattr(agent, "route", None) or []
        if not r:
            if getattr(agent, "start", None) in positions:
                sx, sy = positions[agent.start]
                agent.dynamic_coords = [(sx, sy)]
                agent.dynamic_angles = [get_orientation_from_map(agent.start) or 0.0]
            else:
                agent.dynamic_coords, agent.dynamic_angles = [], []
            agent._first_depart_frame = 10**9
            return

        coords, angles = [], []
        t_cursor = 0.0  # begin timeline at t=0 so pre-first depart shows as a pause
        first_depart = int(round(r[0][2] * fps))

        for (u, v, td, ta) in r:
            # dwell (pause) before this edge, if any (td > t_cursor)
            dwell_frames = max(0, int(round((td - t_cursor) * fps)))
            ux, uy = positions[u]
            heading_uv = 0.0
            if v in positions:
                vx, vy = positions[v]
                heading_uv = math.atan2(vy - uy, vx - ux)

            if dwell_frames > 0:
                if coords:
                    coords.extend([coords[-1]] * dwell_frames)
                    angles.extend([angles[-1]] * dwell_frames)
                else:
                    coords.extend([(ux, uy)] * dwell_frames)
                    angles.extend([heading_uv] * dwell_frames)

            # traverse u->v over its true duration
            dur = max(ta - td, 1.0 / fps)
            steps = max(int(round(dur * fps)), 1)
            for k in range(1, steps + 1):
                a = k / steps
                x = ux + a * (vx - ux)
                y = uy + a * (vy - uy)
                coords.append((x, y))
                angles.append(heading_uv)

            t_cursor = ta  # next dwell measured from this arrival

        agent.dynamic_coords = coords
        agent.dynamic_angles = angles
        agent._first_depart_frame = first_depart

    def _route_nodes(agent):
        """
        Convert agent.route (timed edges) into a node list for highlighting.
        If no timed edges yet, fall back to planned_path / full_route.
        """
        r = getattr(agent, "route", None) or []
        if r:
            nodes = [r[0][0]] + [v for (_, v, _, _) in r]
            return [n for n in nodes if n in positions]
        # fallback: initial plan (only until first edge is booked)
        dashed = getattr(agent, "planned_path", None) or getattr(agent, "full_route", None) or []
        return [n for n in dashed if n in positions]

    def _refresh_highlight(tr, agent):
        """
        Update the dashed 'active path' to reflect the path ACTUALLY being taken:
        i.e., the currently booked route (agent.route). This will change whenever
        the planner commits a new edge or the robot detours.
        """
        nodes = _route_nodes(agent)
        if nodes and len(nodes) >= 2:
            xs, ys = zip(*[positions[n] for n in nodes])
        else:
            xs, ys = [], []
        if tr and tr.get("path_line"):
            tr["path_line"].set_data(xs, ys)

    def _apply_resume(a):
        """Rebuild on-screen trajectory after a new release, time-synced and jump-free."""
        build_time_synced_traj(a)
        tr = agent_to_trail.get(a)
        if tr:
            tr["coords"] = a.dynamic_coords
            tr["angles"] = a.dynamic_angles
            # Start at 0 because we encoded pre-first depart wait into the frames themselves
            tr["start_frame"] = 0
            _refresh_highlight(tr, a)  # <— update dashed highlight to reflect new route
        a.replanned = False

    # ---------- initial trails (markers, dashed paths, dynamic arrays) ----------
    for idx, agent in enumerate(agents):
        if agent.start in positions:
            sx, sy = positions[agent.start]
            ax.plot(
                sx, sy, marker=(3, 0, 0), markerfacecolor="none",
                markeredgecolor=colors[idx % len(colors)], markersize=20, zorder=3
            )
        if getattr(agent, "goal", None) and agent.goal in positions:
            gx, gy = positions[agent.goal]
            ax.plot(
                gx, gy, marker="s", markerfacecolor="none",
                markeredgecolor=colors[idx % len(colors)], markersize=13, zorder=3
            )

        # dashed polyline that will FOLLOW the actually booked route (agent.route)
        nodes = _route_nodes(agent)
        if nodes and len(nodes) >= 2:
            xs, ys = zip(*[positions[n] for n in nodes])
        else:
            xs, ys = [], []
        path_line, = ax.plot(xs, ys, "--",
                             color=colors[idx % len(colors)],
                             linewidth=2, alpha=0.9, zorder=2)

        build_time_synced_traj(agent)

        dot, = ax.plot([], [], marker="o", linestyle="None",
                       markerfacecolor=colors[idx % len(colors)],
                       markeredgecolor="k", markersize=12, zorder=6)

        arrow = FancyArrowPatch((0, 0), (0, 0),
                                arrowstyle="-|>", mutation_scale=20,
                                color=colors[idx % len(colors)], zorder=6)
        arrow.set_visible(False)
        ax.add_patch(arrow)

        trails.append({
            "coords": agent.dynamic_coords,
            "angles": agent.dynamic_angles,
            "dot": dot,
            "arrow": arrow,
            "color": colors[idx % len(colors)],
            "agent": agent,
            "path_line": path_line,   # actively-tracked route highlight
            "start_frame": 0,         # we include pre-depart waits in coords, so start at frame 0
        })

    agent_to_trail = {t["agent"]: t for t in trails}

    # cadence for time-aware releases (periodic poll)
    replan_poll_every = max(1, int(0.5 * fps))  # ~ every 0.5s
    last_poll_frame = -10**9

    def update(frame):
        nonlocal last_poll_frame, last_printed_second, timeline_done

        artists = []

        # ---- draw agents
        for tr in trails:
            coords = tr["coords"]
            if not coords:
                continue
            i = max(0, min(frame - tr["start_frame"], len(coords) - 1))
            x, y = coords[i]
            yaw = tr["angles"][i]
            dx, dy = 0.5 * math.cos(yaw), 0.5 * math.sin(yaw)

            tr["dot"].set_data([x], [y])
            arrow = tr["arrow"]
            arrow.set_visible(True)
            arrow.set_positions((x, y), (x + dx, y + dy))
            artists.extend([tr["dot"], arrow])

        # ---- periodic poll: let planner release new edges if possible
        if graph and (frame - last_poll_frame >= replan_poll_every):
            last_poll_frame = frame
            resumed = planner.replan_waiting_agents(
                agents=agents,
                G=graph,
                frame=frame,
                release_delay_frames=int(wait_time * fps),
            )
            if resumed:
                for a in agents:
                    if getattr(a, "replanned", False):
                        # planner already printed the release line; rebuild trajectory for this agent
                        _apply_resume(a)

        # keep route highlight synced to the *booked* path (changes as edges are released)
        for tr in trails:
            _refresh_highlight(tr, tr["agent"])

        # ----- live per-second timeline that STOPS at the true horizon -----
        if live_timeline and not timeline_done:
            current_second = int(frame / fps)
            horizon_sec = _horizon_sec()  # grows as new edges are released

            if horizon_sec > 0 and current_second >= horizon_sec:
                # Everyone's plan ends by 'horizon_sec' → stop logging further seconds
                timeline_done = True
            elif current_second != last_printed_second:
                print(f"[t={current_second:2d}–{current_second+1:d}s]", flush=True)
                for a in agents:
                    st = _state_at(a, current_second)
                    if st[0] == 'moving':
                        _, u, v, prog, dur = st
                        print(f"  - {a.name:6}: moving  {u:>6} → {v:<6}  ( {prog:4.1f}s / {dur:4.1f}s )", flush=True)
                    else:
                        _, node = st
                        node = node if node is not None else "<unknown>"
                        print(f"  - {a.name:6}: waiting at {node}", flush=True)
                print("------------------------------------------", flush=True)
                last_printed_second = current_second

        return artists

    # ---------- set up the plot ----------
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
                t["arrow"].set_visible(False)
                artists.append(t["arrow"])
            artists.append(t["dot"])
        return artists

    ani = animation.FuncAnimation(
        fig, update,
        frames=max(1, int(40 * fps)),  # up to ~3 minutes
        interval=100, blit=False, repeat=False, init_func=init,
    )

    fig.ani = ani  # keep a strong ref

    if save:
        try:
            os.makedirs("output", exist_ok=True)
            out_path = os.path.abspath(os.path.join("output", "space_time_planner-0.gif"))
            print(f"[i] Saving animation to {out_path} ...")
            ani.save(out_path, writer="pillow", fps=fps)
            print(f"[✓] Animation saved to {out_path}")
        except Exception as e:
            print(f"[x] Failed to save animation: {type(e).__name__}: {e}")
    else:
        plt.show()
