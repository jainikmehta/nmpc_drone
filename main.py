# The problem of using NMPC for quadcopter motion planning can be done in two ways:
# 1) Provide general reference trajectory (non-optimal) to goal as input to NMPC solver as guess.
# 2) Provide optimal reference trajectory as way points to NMPC solver


import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle  # Import Circle patch
import time  # To time the solver
from utils import main_node, ref_generator_2d  # obstacles
from dynamics import dynamics_unicycle


# Map parameters
xlimit = 2
neg_x_limit = -1
ylimit = 2
neg_ylimit = -1
dt = 0.1  # Sampling time (seconds)

# Robot 0 parameters
v_max = 1.0  # Maximum linear velocity (m/s)
ω_max = np.pi / 4  # Maximum angular velocity (rad/s) (~45 deg/s)
large_number = cas.inf  # Use CasADi infinity for bounds

# Goal Robot 0
goal_0 = np.array([1.0, 1.0, 0.0])  # Goal state [x_g0, y_g0, theta_g0]
# Define starting state Robot 0
start_0 = np.array([0.0, 0.0, 0.0])  # Starting state [x0, y0, theta0]

# Reference trajectory Robot 0
ref_generator_0 = ref_generator_2d(start=start_0, goal=goal_0, max_velocity=v_max * dt)
waypoints_ref_0, waypoint_ref_0_length = ref_generator_0.generate_waypoints()
print("Generated Waypoints:")
print(waypoints_ref_0)

# Parameters Robot 0
# Should depend on the waypoints and should be always less than waypoints
if len(waypoint_ref_0_length) < 20:
    pred_horizn = waypoint_ref_0_length  # Prediction horizon
    ctrl_horizn = pred_horizn  # Control horizon

# --- Obstacle Parameters ---
num_obstacles = 1  # Increased number
obstacle_radius = 0.05  # Radius of the physical obstacle
safe_distance = 0.05  # Minimum distance robot center should maintain from obstacle EDGE
min_dist_from_center = obstacle_radius + safe_distance

# Define symbolic variables
nu = 2  # Control dimension

