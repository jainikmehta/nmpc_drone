# The problem of using NMPC for quadcopter motion planning can be done in two ways:
# 1) Provide general reference trajectory (non-optimal) to goal as input to NMPC solver as guess.
# 2) Provide optimal reference trajectory as way points to NMPC solver


import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle  # Import Circle patch
import time  # To time the solver
from utils import nmpc_node, ref_generator_2d
from plotting import plot_states_controls
# from dynamics import dynamics_unicycle


# Map parameters
xlimit = 2
neg_x_limit = -1
ylimit = 2
neg_ylimit = -1
# --- Obstacle Parameters ---
random_obstacles = False
num_obstacles = 1  # number of random obstacles if using
obstacle_radius = 0.05  # Radius of the physical obstacle
safe_distance = 0.4  # Minimum distance robot center should maintain from obstacle EDGE
min_dist_from_center = obstacle_radius + safe_distance

if random_obstacles == False:
    obstacle_centers = np.array([[0.75, 0.5]])
    num_obstacles = len(obstacle_centers)
else:
    obstacle_centers =  np.random.rand(num_obstacles, 2) * 1.0

# Simulation parameters
pred_horizn = 40
ctrl_horizn = 40
dt = 0.1  # Sampling time (seconds)

# Robot 0 parameters
num_states = 3
num_control = 2
v_max = 1.0  # Maximum linear velocity (m/s)
omega_max = np.pi   # Maximum angular velocity (rad/s) (~45 deg/s)
large_number = cas.inf  # Use CasADi infinity for bounds
goal_reached = False
Q_running = cas.diag([100.0, 100.0, 0.0])       # Weights for tracking reference state [x, y, θ]
R_running = cas.diag([10.0, 10.0])            # Weights for control effort [v, ω] - Keep this!
Q_terminal = cas.diag([5.0, 5.0, 5.0])   # Weights for final state deviation from goal


# Goal Robot 0
goal_0 = [1.0, 1.0, 3.141] # Goal state [x_g0, y_g0, theta_g0]
# Define starting state Robot 0
start_0 = [0.0, 0.0, 0.0]  # Starting state [x0, y0, theta0]
store_opt_states = []
store_opt_controls = []
# Handles for Robot 0
ref_generator_0 = ref_generator_2d(start=start_0, goal=goal_0, max_velocity_step=v_max * dt, pred_horizn=pred_horizn)
nmpc_node_robot_0 = nmpc_node(num_states=num_states, num_controls=num_control, 
                              pred_horizn=pred_horizn, ctrl_horizn=ctrl_horizn, 
                              start=start_0, max_velocity=v_max, max_angular_velocity=omega_max,
                              sampling_time=dt, Q_running=Q_running, R_running=R_running, Q_terminal=Q_terminal,
                              num_obstacles=num_obstacles, obstacle_centers=obstacle_centers, safe_distance=safe_distance, min_dist_from_center=min_dist_from_center)
current_state_0=start_0

# # NMPC loop
while goal_reached == False:
    ref_waypoints_0 = ref_generator_0.generate_waypoints(current_state=current_state_0)
    print("Generated Waypoints:")
    print(ref_waypoints_0)
    opt_control_0, opt_states_0 = nmpc_node_robot_0.solve_nmpc(ref_waypoints=ref_waypoints_0, current_state=current_state_0)
    # Do simple plot with static obstacle
    # Apply the first control input. Basically store first input and state as well as all the predicted states.
    current_state_0 = opt_states_0[:, 0]
    dist_to_goal = (current_state_0[0] - goal_0[0])**2 + (current_state_0[1] - goal_0[1])**2
    if dist_to_goal <= 0.01:
        goal_reached = True
    
    plot_states_controls(pred_horizn=pred_horizn, ctrl_horizn=ctrl_horizn, 
                        opt_states_0=opt_states_0, opt_control_0=opt_control_0, 
                        start=start_0, goal=goal_0, ref_waypoints=ref_waypoints_0, 
                        sampling_time=dt, v_max=v_max, omega_max=omega_max,
                        num_obstacles=num_obstacles, obstacle_centers=obstacle_centers, 
                        safe_distance=safe_distance, obstacle_radius=obstacle_radius,
                        min_dist_from_center=min_dist_from_center)
    
