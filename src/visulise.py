import yaml
import matplotlib.pyplot as plt
import os
import math

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

        # Save position
        positions[name] = (pos['x'], pos['y'])

        # Convert orientation to angle
        angle = quaternion_to_angle(ori['z'], ori['w'])

        # Save orientation
        node['angle'] = angle  # Store for later use in drawing arrows

        # Save edges
        for edge in node.get('edges', []):
            edges.append((name, edge['node']))

    # Plot nodes
    for name, (x, y) in positions.items():
        plt.scatter(x, y, c='blue')
        plt.text(x + 0.1, y + 0.1, name, fontsize=10, color='black')

    # Plot orientation arrows
    for item in top_map:
        node = item['node']
        name = node['name']
        x, y = positions[name]
        angle = node['angle']
        dx = 0.3 * math.cos(angle)
        dy = 0.3 * math.sin(angle)
        plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')

    # Plot edges
    for start, end in edges:
        if start in positions and end in positions:
            x1, y1 = positions[start]
            x2, y2 = positions[end]
            plt.plot([x1, x2], [y1, y2], 'k--')

    plt.title("Topological Map")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    yaml_file = os.path.join(".", "data", "map.yaml")
    top_map = load_topological_map(yaml_file)
    plot_topological_map(top_map)
