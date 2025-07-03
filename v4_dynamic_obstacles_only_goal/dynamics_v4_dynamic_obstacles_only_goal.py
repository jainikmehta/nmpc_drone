# Dynamics for v4_dynamic_obstacles_only_goal
# This file should define the robot dynamics for the NMPC solver.
# You can adapt from your previous dynamics file, but ensure it is compatible with the new main script (no reference generator, only goal).

import numpy as np

def unicycle_dynamics(state, control, dt):
    x, y, theta = state
    v, omega = control
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt
    return np.array([x_next, y_next, theta_next])
