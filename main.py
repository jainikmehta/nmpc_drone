import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle # Import Circle patch
import time # To time the solver
from utils import reference_generator, obstacles
from dynamics import dynamics_unicycle

# Parameters Robot 0
N = 200  # Prediction horizon (MAKE SURE N IS LARGE ENOUGH FOR REFERENCE)
dt = 0.1  # Sampling time (seconds)
v_max = 1.0  # Maximum linear velocity (m/s)
ω_max = np.pi / 4  # Maximum angular velocity (rad/s) (~45 deg/s)
large_number = cas.inf # Use CasADi infinity for bounds

# Goal Robot 0
x_goal = np.array([1.0, 1.0, 0.0]) # Goal state [x_g0, y_g0, theta_g0]

# Define initial state Robot 0
x_current = np.array([0.0, 0.0, 0.0])  # [x0, y0, theta0]

# Trajectory for Robot 0
wp1 = np.array([1.2, 0.0, 0.0])      # Waypoint 1 (end of first segment)
wp2 = np.array([1.2, 1.0, np.pi/2])  # Waypoint 2 (end of second segment)
waypoints_robot0 = np.array([wp1, wp2])

# Reference trajectory Robot 0
reference_generator_0 = reference_generator(nx, initial_pos = x_current, final_pos = x_goal, waypoints = waypoints_robot0)
traj_ref_0 = reference_generator_0.generate_trajectory()

# --- Obstacle Parameters ---
num_obstacles = 1 # Increased number
obstacle_radius = 0.05 # Radius of the physical obstacle
safe_distance = 0.05   # Minimum distance robot center should maintain from obstacle EDGE
min_dist_from_center = obstacle_radius + safe_distance

# Define symbolic variables
nu = 2  # Control dimension

