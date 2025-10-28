import yaml
import matplotlib.pyplot as plt
import os
import math
from matplotlib.patches import FancyArrowPatch


def load_topological_map(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def quaternion_to_angle(z, w):
    """Convert a 2D quaternion to a yaw angle in radians."""
    return math.atan2(2.0 * (w * z), 1.0 - 2.0 * (z ** 2))

def plot_topological_map(top_map):
    positions = {}
    edges = []

    for item in top_map:
        node = item['node']
        name = node['name']
        pos = node['pose']['position']
        ori = node['pose']['orientation']

        positions[name] = (pos['x'], pos['y'])

        angle = quaternion_to_angle(ori['z'], ori['w'])
        node['angle'] = angle

        for edge in node.get('edges', []):
            edges.append((name, edge['node']))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(False)

    # Plot nodes
    for name, (x, y) in positions.items():
        ax.scatter(x, y, c='blue')
        ax.text(x + 0.1, y + 0.1, name, fontsize=10, color='black')

    # Plot orientation arrows (red)
    for item in top_map:
        node = item['node']
        name = node['name']
        x, y = positions[name]
        angle = node['angle']
        dx = 0.3 * math.cos(angle)
        dy = 0.3 * math.sin(angle)
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')

    # Plot edges and direction markers
    for start, end in edges:
        if start in positions and end in positions:
            x1, y1 = positions[start]
            x2, y2 = positions[end]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1)

            # Draw direction marker at edge midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            norm = math.hypot(dx, dy)
            if norm == 0:
                continue
            dx /= norm
            dy /= norm

            arrow = FancyArrowPatch(
                (mx - 0.1 * dx, my - 0.1 * dy),
                (mx + 0.1 * dx, my + 0.1 * dy),
                arrowstyle='-|>',
                mutation_scale=15,
                color='gray',
                linewidth=1
            )
            ax.add_patch(arrow)

    plt.title("Topological Map with Direction Markers")
    plt.show()


if __name__ == '__main__':
    yaml_file = os.path.join(".", "data", "map.yaml")
    top_map = load_topological_map(yaml_file)
    plot_topological_map(top_map)
