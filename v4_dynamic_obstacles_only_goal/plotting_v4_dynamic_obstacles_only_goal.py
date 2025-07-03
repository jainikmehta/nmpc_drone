# Plotting for v4_dynamic_obstacles_only_goal
# Plots robot, static walls, and goal
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle

def plot_states_controls(opt_states, opt_controls, start, goal, walls, obstacle_radius, min_safe_dist, step=None, save_path=None, actual_path=None):
    plt.close('all')
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    # Plot predicted path (blue dotted line)
    plt.plot(opt_states[0, :], opt_states[1, :], 'b--', marker='o', label='Predicted Path')
    # Plot actual path (black solid line)
    if actual_path is not None and len(actual_path) > 1:
        actual_path_np = np.array(actual_path)
        plt.plot(actual_path_np[:, 0], actual_path_np[:, 1], 'k-', linewidth=2, label='Actual Path')
    plt.scatter(start[0], start[1], c='lime', marker='o', s=120, label='Start', edgecolors='black', zorder=10)
    plt.scatter(goal[0], goal[1], c='red', marker='*', s=180, label='Goal', edgecolors='black', zorder=10)
    # Plot static walls
    for wall in walls:
        rect = Rectangle((wall['x'], wall['y']), wall['w'], wall['h'], color='gray', alpha=0.8, zorder=5)
        ax.add_patch(rect)
    # Plot robot safety circle at each step in prediction
    for i in range(opt_states.shape[1]):
        circ = Circle((opt_states[0, i], opt_states[1, i]), min_safe_dist, color='orange', fill=False, linestyle=':', alpha=0.5)
        ax.add_patch(circ)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('NMPC Path Planning with Static Walls')
    plt.legend()
    plt.axis('equal')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 2.5)
    if save_path is not None and step is not None:
        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'nmpc_step_{step:04d}.png'), bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')
