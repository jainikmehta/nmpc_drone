

import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle  # Import Circle patch
import time  # To time the solver
from utils_v2_avoid_obstacle_traj import nmpc_node, ref_generator_2d
from plotting_v2_avoid_obstacle_traj import plot_states_controls
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
buffer_distance_from_obstacle_boundary = 0.05  # Minimum distance robot center should maintain from obstacle EDGE
min_safe_dist_from_obstacle_center = obstacle_radius + buffer_distance_from_obstacle_boundary

if random_obstacles == False:
    obstacle_centers = np.array([[0.85, 0.75], [0.3,0.3], [0.5, 0.8], [0.75, 0.25], [0.25, 0.75]])  # Fixed obstacle centers
    num_obstacles = len(obstacle_centers)
else:
    obstacle_centers =  np.random.rand(num_obstacles, 2) * 1.0

# Simulation parameters
pred_horizn = 5
ctrl_horizn = 5
dt = 0.1  # Sampling time (seconds)

# Robot 0 parameters
num_states = 3
num_controls = 2
v_max = 1.0  # Maximum linear velocity (m/s)
v_min = 0.1  # Minimum linear velocity (m/s)
omega_max = np.pi   # Maximum angular velocity (rad/s) (~45 deg/s)
large_number = cas.inf  # Use CasADi infinity for bounds
goal_reached = False
Q_running = cas.diag([10.0, 10.0, 0.005])       # Weights for tracking reference state [x, y, θ]
R_running = cas.diag([10.0, 10.0])            #  Weights for control effort [v, ω] - Keep this!
Q_terminal = cas.diag([50.0, 50.0, 50.0])   # Weights for final state deviation from goal


# Goal Robot 0
goal = [0.6, 1.0, 3.141] # Goal state [x_g0, y_g0, theta_g0]
# Define starting state Robot 0
start = [0.0, 0.0, 0.0]  # Starting state [x0, y0, theta0]
store_opt_states = []
store_opt_controls = []
# Handles for Robot 0
ref_generator = ref_generator_2d(start=start, goal=goal, max_velocity_step=v_max * dt, 
                                 pred_horizn=pred_horizn, obstacle_radius=obstacle_radius)
nmpc_node_robot = nmpc_node(num_states=num_states, num_controls=num_controls, 
                              pred_horizn=pred_horizn, ctrl_horizn=ctrl_horizn, 
                              start=start, max_velocity=v_max, min_velocity=v_min, max_angular_velocity=omega_max,
                              sampling_time=dt, Q_running=Q_running, R_running=R_running, Q_terminal=Q_terminal,
                              num_obstacles=num_obstacles, obstacle_centers=obstacle_centers, 
                              buffer_distance_from_obstacle_boundary=buffer_distance_from_obstacle_boundary, 
                              min_safe_dist_from_obstacle_center=min_safe_dist_from_obstacle_center)

current_state = start
previous_waypoints = [current_state]*pred_horizn
# NMPC loop
while goal_reached == False:
    ref_waypoints = ref_generator.generate_waypoints(
        previous_waypoints=np.array(previous_waypoints),
        current_state=current_state,
        obstacle_centers=obstacle_centers)
    previous_waypoints = ref_waypoints
    print("Generated Waypoints:")
    print(ref_waypoints)
    opt_control, opt_states, min_h_values = nmpc_node_robot.solve_nmpc(ref_waypoints=ref_waypoints, current_state=current_state)
    # Do simple plot with static obstacle
    # Apply the first control input. Basically store first input and state as well as all the predicted states.
    current_state = opt_states[:, 0]
    dist_to_goal = np.linalg.norm(current_state[:2] - goal[:2])
    print("current_angle:", current_state[2])
    print("goal_angle:", goal[2])
    angle_to_goal = current_state[2] - goal[2]
    print("angle_to_goal", abs(angle_to_goal))
    if dist_to_goal <= 0.35 and abs(angle_to_goal) <= 0.3:
        goal_reached = True
    
    plot_states_controls(pred_horizn=pred_horizn, ctrl_horizn=ctrl_horizn, 
                        opt_states_0=opt_states, opt_control_0=opt_control, 
                        start=start, goal=goal, ref_waypoints=ref_waypoints, 
                        sampling_time=dt, v_max=v_max, omega_max=omega_max,
                        num_obstacles=num_obstacles, obstacle_centers=obstacle_centers, 
                        safe_distance=buffer_distance_from_obstacle_boundary, obstacle_radius=obstacle_radius,
                        min_dist_from_center=min_safe_dist_from_obstacle_center, min_h_values=min_h_values)

