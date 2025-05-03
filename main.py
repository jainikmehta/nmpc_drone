import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle # Import Circle patch
import time # To time the solver
from utils import main_node, reference_generator_2d # obstacles
from dynamics import dynamics_unicycle

# Map parameters
xlimit = 2
neg_x_limit = -1
ylimit = 2
neg_ylimit = -1

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

# Obstacles centers


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

