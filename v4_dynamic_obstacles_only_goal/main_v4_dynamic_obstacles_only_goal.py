# Main script for v4_dynamic_obstacles_only_goal
import numpy as np
from utils_v4_dynamic_obstacles_only_goal import NMPCNode
from plotting_v4_dynamic_obstacles_only_goal import plot_states_controls

# Map and wall parameters
dt = 0.1
pred_horizn = 20  # Increased horizon
ctrl_horizn = 20
num_states = 3
num_controls = 2
v_max = 1.0
v_min = 0.1
omega_max = np.pi
obstacle_radius = 0.05
buffer_distance = 0.05 # From the obstacle boundary
min_safe_dist = obstacle_radius + buffer_distance

# Define static walls (3 walls in a room-like map)
walls = [
    {'x': 0.5, 'y': 0.0, 'w': 0.1, 'h': 1.5},  # vertical wall
    {'x': 1.0, 'y': 1.0, 'w': 1.0, 'h': 0.1},  # horizontal wall
    {'x': 1.5, 'y': 0.0, 'w': 0.1, 'h': 1.0},  # vertical wall
]

# Start and goal (no orientation for goal)
start = np.array([0.2, 0.2, 0.0])
goal = np.array([2.0, 1.8])

Q_running = np.diag([5.0, 5.0, 0.0])  # Increased state error cost
R_running = np.diag([0.1, 0.1])         # Lower control cost
Q_terminal = np.diag([200.0, 200.0, 0.0])  # Increased terminal cost

nmpc_node = NMPCNode(
    num_states=num_states,
    num_controls=num_controls,
    pred_horizn=pred_horizn,
    ctrl_horizn=ctrl_horizn,
    start=start,
    max_velocity=v_max,
    min_velocity=v_min,
    max_angular_velocity=omega_max,
    sampling_time=dt,
    Q_running=Q_running,
    R_running=R_running,
    Q_terminal=Q_terminal,
    walls=walls,
    min_safe_dist=min_safe_dist
)

current_state = start.copy()
store_opt_states = []
store_opt_controls = []
actual_path = [current_state[:2].copy()]
step = 0
max_steps = 100
goal_tol = 0.2  # Slightly larger goal tolerance

while step < max_steps:
    opt_controls, opt_states = nmpc_node.solve_nmpc(current_state, goal)
    store_opt_states.append(opt_states)
    store_opt_controls.append(opt_controls)
    plot_states_controls(
        opt_states=opt_states,
        opt_controls=opt_controls,
        start=start,
        goal=goal,
        walls=walls,
        obstacle_radius=obstacle_radius,
        min_safe_dist=min_safe_dist,
        step=step,
        save_path="plots_v4_dynamic_obstacles_only_goal",
        actual_path=actual_path
    )
    # Apply first control
    current_state = opt_states[:, 1]
    actual_path.append(current_state[:2].copy())
    if np.linalg.norm(current_state[:2] - goal) < goal_tol:
        print(f"Goal reached at step {step}")
        break
    step += 1
