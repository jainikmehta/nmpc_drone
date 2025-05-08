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
num_obstacles = 1  # Increased number
obstacle_radius = 0.05  # Radius of the physical obstacle
safe_distance = 0.05  # Minimum distance robot center should maintain from obstacle EDGE
min_dist_from_center = obstacle_radius + safe_distance

# Simulation parameters
pred_horizn = 20
ctrl_horizn = 20
dt = 0.1  # Sampling time (seconds)

# Robot 0 parameters
num_states = 3
num_control = 2
v_max = 1.0  # Maximum linear velocity (m/s)
omega_max = np.pi / 4  # Maximum angular velocity (rad/s) (~45 deg/s)
large_number = cas.inf  # Use CasADi infinity for bounds
goal_reached = False

# Goal Robot 0
goal_0 = [1.0, 1.0, 0.0] # Goal state [x_g0, y_g0, theta_g0]
# Define starting state Robot 0
start_0 = [0.0, 0.0, 0.0]  # Starting state [x0, y0, theta0]

# Handles for Robot 0
ref_generator_0 = ref_generator_2d(start=start_0, goal=goal_0, max_velocity=v_max * dt, pred_horizn=pred_horizn)
nmpc_node_robot_0 = nmpc_node(num_states=num_states, num_controls=num_control, pred_horizn=pred_horizn, ctrl_horizn=ctrl_horizn, start=start_0, max_velocity=v_max, max_angular_velocity=omega_max, sampling_time=dt)
current_state_0=start_0

# # NMPC loop
# while goal_reached == False:
    
waypoints_ref_0 = ref_generator_0.generate_waypoints(current_state=current_state_0)
print("Generated Waypoints:")
print(waypoints_ref_0)
opt_control_0, opt_states_0 = nmpc_node_robot_0.solve_nmpc(ref_waypoints=waypoints_ref_0)





# def plot_optimal_trajectories(optimal_controls, optimal_states, dt_controls=1.0, dt_states=1.0,
#                               control_names=None, state_names=None,
#                               plot_xy_trajectory=True):
#     """
#     Plots the optimal control inputs and state trajectories.

#     Args:
#         optimal_controls (np.ndarray): Array of optimal control inputs.
#                                        Expected shape: (num_controls, ctrl_horizon)
#         optimal_states (np.ndarray):   Array of optimal states.
#                                        Expected shape: (num_states, pred_horizon)
#         dt_controls (float):           Time step for the control horizon (for x-axis scaling).
#         dt_states (float):             Time step for the state prediction horizon (for x-axis scaling).
#         control_names (list of str, optional): Names for each control input.
#                                                Defaults to ['Control 0', 'Control 1', ...].
#         state_names (list of str, optional):   Names for each state variable.
#                                                Defaults to ['State 0', 'State 1', ...].
#         plot_xy_trajectory (bool):     If True and num_states >= 2, plots an X-Y trajectory
#                                        assuming the first two states are X and Y.
#     """

#     if not isinstance(optimal_controls, np.ndarray) or not isinstance(optimal_states, np.ndarray):
#         print("Error: Inputs must be NumPy arrays.")
#         return

#     # --- Plot Optimal Controls ---
#     if optimal_controls.ndim == 2:
#         num_controls, ctrl_horizon = optimal_controls.shape
#         time_controls = np.arange(ctrl_horizon) * dt_controls

#         if control_names is None or len(control_names) != num_controls:
#             control_names = [f"Control {i}" for i in range(num_controls)]

#         plt.figure(figsize=(12, num_controls * 3))
#         plt.suptitle("Optimal Control Trajectories", fontsize=16)
#         for i in range(num_controls):
#             plt.subplot(num_controls, 1, i + 1)
#             plt.plot(time_controls, optimal_controls[i, :], marker='o', linestyle='-')
#             plt.title(control_names[i])
#             plt.xlabel("Time (s)" if dt_controls > 0 else "Control Step")
#             plt.ylabel("Control Value")
#             plt.grid(True)
#         plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
#         plt.show()

#     elif optimal_controls.ndim == 1 and optimal_controls.size > 0: # Single control input or single step
#         num_controls = 1
#         ctrl_horizon = optimal_controls.size
#         time_controls = np.arange(ctrl_horizon) * dt_controls
#         if control_names is None:
#             control_names = ["Control 0"]

#         plt.figure(figsize=(10, 4))
#         plt.plot(time_controls, optimal_controls, marker='o', linestyle='-')
#         plt.title(f"Optimal {control_names[0]} Trajectory")
#         plt.xlabel("Time (s)" if dt_controls > 0 else "Control Step")
#         plt.ylabel("Control Value")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
#     else:
#         print(f"Could not plot optimal_controls. Unexpected shape or empty: {optimal_controls.shape}")


#     # --- Plot Optimal States ---
#     if optimal_states.ndim == 2:
#         num_states, pred_horizon = optimal_states.shape
#         time_states = np.arange(pred_horizon) * dt_states

#         if state_names is None or len(state_names) != num_states:
#             state_names = [f"State {i}" for i in range(num_states)]

#         plt.figure(figsize=(12, num_states * 3))
#         plt.suptitle("Optimal State Trajectories", fontsize=16)
#         for i in range(num_states):
#             plt.subplot(num_states, 1, i + 1)
#             plt.plot(time_states, optimal_states[i, :], marker='.', linestyle='-')
#             plt.title(state_names[i])
#             plt.xlabel("Time (s)" if dt_states > 0 else "Prediction Step")
#             plt.ylabel("State Value")
#             plt.grid(True)
#         plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout
#         plt.show()

#         # Plot X-Y trajectory if applicable
#         if plot_xy_trajectory and num_states >= 2:
#             x_coord = optimal_states[0, :]
#             y_coord = optimal_states[1, :]
#             x_name = state_names[0] if state_names else "State 0 (X)"
#             y_name = state_names[1] if state_names else "State 1 (Y)"

#             plt.figure(figsize=(8, 7))
#             plt.plot(x_coord, y_coord, marker='o', linestyle='-', label="Trajectory")
#             plt.plot(x_coord[0], y_coord[0], 'go', markersize=10, label="Start of Prediction")
#             plt.plot(x_coord[-1], y_coord[-1], 'ro', markersize=10, label="End of Prediction")
#             plt.title("Predicted X-Y Trajectory")
#             plt.xlabel(x_name)
#             plt.ylabel(y_name)
#             plt.legend()
#             plt.grid(True)
#             plt.axis('equal') # Important for a proper aspect ratio in XY plots
#             plt.tight_layout()
#             plt.show()

#     elif optimal_states.ndim == 1 and optimal_states.size > 0: # Single state variable or single step
#         num_states = 1
#         pred_horizon = optimal_states.size
#         time_states = np.arange(pred_horizon) * dt_states
#         if state_names is None:
#             state_names = ["State 0"]

#         plt.figure(figsize=(10, 4))
#         plt.plot(time_states, optimal_states, marker='.', linestyle='-')
#         plt.title(f"Optimal {state_names[0]} Trajectory")
#         plt.xlabel("Time (s)" if dt_states > 0 else "Prediction Step")
#         plt.ylabel("State Value")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
#     else:
#         print(f"Could not plot optimal_states. Unexpected shape or empty: {optimal_states.shape}")

