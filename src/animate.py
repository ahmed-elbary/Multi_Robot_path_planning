import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import math
import networkx as nx
from utils import build_graph_from_yaml, find_path


# === Load and Parse Map ===
def load_map(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def quaternion_to_yaw(z, w):
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * (z ** 2))

def get_node_data(top_map):
    positions = {}
    edges = []
    for item in top_map:
        node = item['node']
        name = node['name']
        x = node['pose']['position']['x']
        y = node['pose']['position']['y']
        positions[name] = (x, y)

        for edge in node.get('edges', []):
            edges.append((name, edge['node']))
    return positions, edges

# === Animate Agent ===
def animate_robot(path, positions, speed_mps=1.0, fps=30, topo_map=None):
    fig, ax = plt.subplots()
    ax.set_title("Robot Path Animation with Orientation")
    ax.axis("equal")
    ax.grid(False)

    # Plot all nodes
    for name, (x, y) in positions.items():
        ax.scatter(x, y, c='blue')
        ax.text(x + 0.1, y + 0.1, name, fontsize=9)

    # Draw all available edges (gray, thin)
    for u, v in graph.edges():
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2], color='gray', linestyle='-', linewidth=1)

    # Highlight the path
    for i in range(len(path) - 1):
        x1, y1 = positions[path[i]]
        x2, y2 = positions[path[i + 1]]
        ax.plot([x1, x2], [y1, y2], 'r--', linewidth=2)

    # Agent marker and orientation arrow
    agent_dot, = ax.plot([], [], 'ro', markersize=12)
    heading_arrow = ax.arrow(0, 0, 0, 0, head_width=0.15, head_length=0.2, fc='red', ec='red')

    # Helper: get yaw angle from node name
    def get_yaw(node_name):
        for item in topo_map:
            node = item['node']
            if node['name'] == node_name:
                q = node['pose']['orientation']
                return quaternion_to_yaw(q['z'], q['w'])
        return 0.0  # fallback

    # Build movement + orientation frames
    points = []
    angles = []
    for i in range(len(path) - 1):
        x1, y1 = positions[path[i]]
        x2, y2 = positions[path[i + 1]]
        yaw1 = get_yaw(path[i])
        yaw2 = get_yaw(path[i + 1])

        distance = math.hypot(x2 - x1, y2 - y1)
        travel_time = distance / speed_mps
        steps = max(int(travel_time * fps), 1)

        for t in range(steps):
            alpha = t / steps
            x = (1 - alpha) * x1 + alpha * x2
            y = (1 - alpha) * y1 + alpha * y2
            yaw = math.atan2(y2 - y1, x2 - x1)
            points.append((x, y))
            angles.append(yaw)

    points.append(positions[path[-1]])
    angles.append(get_yaw(path[-1]))

    # Update animation
    def update(frame):
        x, y = points[frame]
        yaw = angles[frame]
        dx = 0.5 * math.cos(yaw)
        dy = 0.5 * math.sin(yaw)

        agent_dot.set_data([x], [y])

        # Remove previous arrow and draw new one
        for patch in reversed(ax.patches):
            patch.remove()

        ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.2, fc='red', ec='red')
        return agent_dot,

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(points),
        interval=1000 / fps,
        repeat=False
    )
    plt.show()



# === Main Entry ===
if __name__ == "__main__":
    # Load map.yaml
    yaml_path = os.path.join(".", "data", "map.yaml")
    topo_map = load_map(yaml_path)
    
    from utils import build_graph_from_yaml, find_path
    graph, positions = build_graph_from_yaml(topo_map)

    # Define multiple goals (start and waypoints)
    multi_goal_path = ["Node1",  "Node15", "Node1"]

    # Build full A* path across all segments
    full_path = []
    for i in range(len(multi_goal_path) - 1):
        start = multi_goal_path[i]
        goal = multi_goal_path[i + 1]
        segment = find_path(graph, start, goal, positions)

        if not segment:
            print(f"No path found between {start} and {goal}")
            continue

        if i > 0:
            segment = segment[1:]  # Avoid repeating shared node
        full_path.extend(segment)

    print("Multi-goal path:", full_path)

    # Animate the full path
    animate_robot(full_path, positions, speed_mps=1.0, fps=10, topo_map=topo_map)


