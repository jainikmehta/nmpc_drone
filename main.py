import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle # Import Circle patch
import time # To time the solver

# Parameters
N = 200  # Prediction horizon (MAKE SURE N IS LARGE ENOUGH FOR REFERENCE)
dt = 0.1  # Sampling time (seconds)
v_max = 1.0  # Maximum linear velocity (m/s)
ω_max = np.pi / 4  # Maximum angular velocity (rad/s) (~45 deg/s)
large_number = cas.inf # Use CasADi infinity for bounds

# Goal
x_goal = np.array([1.0, 1.0, 0.0]) # Goal state [x, y, theta]

# Define initial state
x_current = np.array([0.0, 0.0, 0.0])  # [x0, y0, θ0]

# --- Obstacle Parameters ---
num_obstacles = 1 # Increased number
obstacle_radius = 0.05 # Radius of the physical obstacle
safe_distance = 0.05   # Minimum distance robot center should maintain from obstacle EDGE
min_dist_from_center = obstacle_radius + safe_distance