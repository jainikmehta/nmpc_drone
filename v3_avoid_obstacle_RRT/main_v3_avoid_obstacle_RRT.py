import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle  # Import Circle patch
import time  # To time the solver
from utils_v3_avoid_obstacle_RRT import nmpc_node, RRTReferenceGenerator 
from plotting_v3_avoid_obstacle_RRT import plot_states_controls

# Map parameters
xlimit = 2
neg_x_limit = -1
ylimit = 2
neg_ylimit = -1
map_limits = [neg_x_limit, xlimit, neg_ylimit, ylimit] # Added for RRT

# --- Obstacle Parameters ---
random_obstacles = False
num_obstacles = 1  # number of random obstacles if using
obstacle_radius = 0.05  # Radius of the physical obstacle
safe_distance = 0.05  # Minimum distance robot center should maintain from obstacle EDGE
min_dist_from_center = obstacle_radius + safe_distance

if random_obstacles == False:
    obstacle_centers = np.array([[0.7, 0.75]])
    num_obstacles = len(obstacle_centers)
else:
    obstacle_centers =  np.random.rand(num_obstacles, 2) * 1.0

# Simulation parameters
pred_horizn = 5
ctrl_horizn = 5
dt = 0.1  # Sampling time (seconds)

# Robot 0 parameters
num_states = 3
num_control = 2
v_max = 1.0  # Maximum linear velocity (m/s)
v_min = 0.8
omega_max = np.pi   # Maximum angular velocity (rad/s) (~45 deg/s)
large_number = cas.inf  # Use CasADi infinity for bounds
goal_reached = False
Q_running = cas.diag([10.0, 10.0, 0.05])       # Weights for tracking reference state [x, y, θ]
R_running = cas.diag([10.0, 10.0])              #  Weights for control effort [v, ω] - Keep this!
Q_terminal = cas.diag([500.0, 500.0, 500.0])   # Weights for final state deviation from goal


# Goal Robot 0
goal_0 = [1.0, 1.0, 3.141] # Goal state [x_g0, y_g0, theta_g0]
# Define starting state Robot 0
start_0 = [0.0, 0.0, 0.0]  # Starting state [x0, y0, theta0]
store_opt_states = []
store_opt_controls = []

# --- MODIFIED: RRT-based Reference Trajectory Generation ---
# The RRT planner is called once to generate a global path.
print("Initializing RRT Reference Generator...")
ref_generator_0 = RRTReferenceGenerator(start=start_0, goal=goal_0, pred_horizn=pred_horizn,
                                        dt=dt, v_max=v_max,
                                        obstacle_centers=obstacle_centers,
                                        min_dist_from_center=min_dist_from_center,
                                        map_limits=map_limits)
print("RRT Reference Generator Initialized.")
# ---

# Handles for Robot 0
nmpc_node_robot_0 = nmpc_node(num_states=num_states, num_controls=num_control,
                                pred_horizn=pred_horizn, ctrl_horizn=ctrl_horizn,
                                start=start_0, max_velocity=v_max, min_velocity=v_min, max_angular_velocity=omega_max,
                                sampling_time=dt, Q_running=Q_running, R_running=R_running, Q_terminal=Q_terminal,
                                num_obstacles=num_obstacles, obstacle_centers=obstacle_centers, safe_distance=safe_distance, min_dist_from_center=min_dist_from_center)
current_state_0=start_0

# # NMPC loop
while goal_reached == False:
    # MODIFIED: Generate waypoints by following the global RRT path
    ref_waypoints_0 = ref_generator_0.generate_waypoints(current_state=current_state_0)
    
    print("Generated Waypoints:")
    print(ref_waypoints_0)
    
    opt_control_0, opt_states_0, min_h_values = nmpc_node_robot_0.solve_nmpc(ref_waypoints=ref_waypoints_0, current_state=current_state_0)
    
    # Apply the first control input.
    current_state_0 = opt_states_0[:, 0]
    
    # Check if goal is reached
    dist_to_goal = (current_state_0[0] - goal_0[0])**2 + (current_state_0[1] - goal_0[1])**2
    angle_to_goal = current_state_0[2] - goal_0[2]
    
    if dist_to_goal <= 0.01 and abs(angle_to_goal) <= 0.1:
        goal_reached = True
        print("Goal Reached!")
    
    plot_states_controls(pred_horizn=pred_horizn, ctrl_horizn=ctrl_horizn,
                         opt_states_0=opt_states_0, opt_control_0=opt_control_0,
                         start=start_0, goal=goal_0, ref_waypoints=ref_waypoints_0,
                         sampling_time=dt, v_max=v_max, omega_max=omega_max,
                         num_obstacles=num_obstacles, obstacle_centers=obstacle_centers,
                         safe_distance=safe_distance, obstacle_radius=obstacle_radius,
                         min_dist_from_center=min_dist_from_center, min_h_values=min_h_values)