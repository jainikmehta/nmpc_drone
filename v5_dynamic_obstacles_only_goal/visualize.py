import numpy as np
import matplotlib.pyplot as plt

# Map parameters (edit as needed)
xlimit = 1
neg_x_limit = -1
ylimit = 2
neg_ylimit = -1
wall_thickness = 0.05
middle_y = 0.5
middle_gap = 0.5

walls = np.array([
    [neg_x_limit - wall_thickness, ylimit + wall_thickness, xlimit + wall_thickness, ylimit],         # Top wall
    [neg_x_limit - wall_thickness, neg_ylimit - wall_thickness, neg_x_limit, ylimit + wall_thickness], # Left wall
    [neg_x_limit - wall_thickness, neg_ylimit - wall_thickness, xlimit + wall_thickness, neg_ylimit], # Bottom wall
    [xlimit, neg_ylimit - wall_thickness, xlimit + wall_thickness, ylimit + wall_thickness],          # Right wall
    # Left middle wall: from left edge to start of gap
    [neg_x_limit, middle_y + wall_thickness / 2, -middle_gap / 2, middle_y - wall_thickness / 2],
    # Right middle wall: from end of gap to right edge
    [middle_gap / 2, middle_y + wall_thickness / 2, xlimit, middle_y - wall_thickness / 2],
])

def plot_map(walls):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    for wall in walls:
        x1, y1, x2, y2 = wall
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        lower_left = (min(x1, x2), min(y1, y2))
        rect = plt.Rectangle(lower_left, width, height, color='gray', alpha=0.8)
        ax.add_patch(rect)
    plt.xlim(neg_x_limit - 0.2, xlimit + 0.2)
    plt.ylim(neg_ylimit - 0.2, ylimit + 0.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Map with Walls')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_map(walls)
