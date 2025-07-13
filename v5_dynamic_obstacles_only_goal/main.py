import casadi as cas
import numpy as np
from utils import nmpc_node, pred_obs_traj
from plotting import plot_states_controls


# Map parameters
xlimit = 2
neg_x_limit = -2
ylimit = 2
neg_ylimit = -2
# --- Obstacle Parameters ---
rand_obs = False # Set to True to use random obstacles, False for predefined ones
num_obs = 1  # number of random obstacles if using
obs_radius = 0.05  # Radius of the physical obstacle
# constant_obstacle_speed = 0.1
buffer_dist = 0.05  # Minimum distance robot center should maintain from obstacle EDGE
min_safe_dist = (
    obs_radius + buffer_dist
)


# Simulation parameters
pred_horizn = 1 # Prediction horizon (number of steps to predict)
ctrl_horizn = 2 # Control horizon (number of steps to apply control)
dt = 0.1  # Sampling time (seconds)


# Robot 0 parameters
num_states = 3 # [x, y, theta]
num_ctrls = 2 # [v, omega]
v_max = 1.0  # Maximum linear velocity (m/s)
v_min = 0.1  # Minimum linear velocity (m/s)
omega_max = np.pi   # Maximum angular veloctiy (rad/s) (~45 deg/s)
large_number = cas.inf  # Use CasADi infinity for bounds
goal_reached = False
state_running_cost = [10.0, 10.0, 0.005] # Weights for tracking reference state [x, y, θ], Shape: (3, 1)
control_running_cost = [10.0, 10.0]  # Weights for control effort [v, ω], Shape: (2, 1)
state_terminal_cost = [50.0, 50.0, 50.0] # Weights for final state deviation from goal, Shape: (3, 1)
# Goal Robot 0
rb_goal = [0.6, 1.0, 3.141]  # Goal state [x_g0, y_g0, theta_g0]
# Define starting state Robot 0
rb_start = [0.0, 0.0, 0.0]  # Starting state [x0, y0, theta0]
store_opt_states = [] # Store optimal states for plotting
store_opt_ctrls = [] # Store optimal states and controls for plotting
rb_curr_state = rb_start # Initial state of the robot
rb_state_hist = [rb_curr_state] # Initialize state history for plotting and storing


# Obstacle state initialization
x_var = 0.01  # Small initial variance in position
y_var = 0.01  # Small initial variance in position
# Each obstacle: [x, y, heading, speed, angular_velocity]
if not rand_obs:
    # shape: (num_obs, 5)
    obs_curr_state = np.array([
        [0.1, x_var, 0.75, y_var, 0.0, 0.035, 0.0],    # Obstacle 1: x, x_var, y, y_var, heading (rad), speed (m/s), angular velocity (rad/s)
        [0.9, x_var, 0.5, y_var, 3.14, 0.12, 0.0],     # Obstacle 2: x, x_var, y, y_var, heading (rad), speed (m/s), angular velocity (rad/s)
        [0.1, x_var, 0.25, y_var, 1.57, 0.035, 0.0],   # Obstacle 3: x, x_var, y, y_var, heading (rad), speed (m/s), angular velocity (rad/s)
    ])
    num_obs = len(obs_curr_state)
else:
    # Random obstacles: assign random heading, speed, and angular velocity as well
    obs_curr_state = np.zeros(num_obs)
    for i in range(num_obs):
        x = np.random.uniform(neg_x_limit + obs_radius, xlimit - obs_radius)
        y = np.random.uniform(neg_ylimit + obs_radius, ylimit - obs_radius)
        heading = np.random.uniform(-np.pi, np.pi)
        speed = np.random.uniform(0.05, 0.15)  # Random speed between 0.05 and 0.15 m/s
        angular_velocity = np.random.uniform(-0.1, 0.1)  # Random angular velocity between -0.1 and 0.1 rad/s
        obs_curr_state[i] = [x, x_var, y, y_var, heading, speed, angular_velocity]


# Static obstacle represntated as walls
# Each wall: [x_start, y_start, x_end, y_end]
# Thickness of the wall for plotting is not required as it is implied from the x, y pairs: x_top_left, y_top_left, x_bottom_right, y_bottom_right
wall_thickness = 0.1
# Walls defined as rectangles using [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
# Two horizontal walls in the middle with adjustable gap
middle_gap = 0.4  # Adjustable gap between the two middle walls
middle_y = 0.0    # y-coordinate for the middle walls
walls = np.array([
    [neg_x_limit - wall_thickness, ylimit + wall_thickness, xlimit + wall_thickness, ylimit],         # Top wall
    [neg_x_limit - wall_thickness, neg_ylimit - wall_thickness, neg_x_limit, ylimit + wall_thickness], # Left wall
    [neg_x_limit - wall_thickness, neg_ylimit - wall_thickness, xlimit + wall_thickness, neg_ylimit], # Bottom wall
    [xlimit, neg_ylimit - wall_thickness, xlimit + wall_thickness, ylimit + wall_thickness],          # Right wall
    # Left middle wall: from left edge to start of gap
    [neg_x_limit, middle_y + wall_thickness / 2, -middle_gap / 2, middle_y - wall_thickness / 2],
    # Right middle wall: from end of gap to right edge
    [middle_gap / 2, middle_y + wall_thickness / 2, xlimit, middle_y - wall_thickness / 2],
])


nmpc_node_robot = nmpc_node(
    num_states=num_states,
    num_ctrls=num_ctrls,
    pred_horizn=pred_horizn,
    ctrl_horizn=ctrl_horizn,
    start=rb_start,
    goal = rb_goal,
    v_max=v_max,
    v_min=v_min,
    omega_max=omega_max,
    sampling_time=dt,
    state_running_cost=state_running_cost,
    control_running_cost=control_running_cost,
    state_terminal_cost=state_terminal_cost,
    num_obs=num_obs,
    buffer_dist=buffer_dist,
    min_safe_dist=min_safe_dist,
)



obs_state_hist = [obs_curr_state]  # Initialize obstacle history for plotting and storing

# NMPC loop
while not goal_reached:

    # Predict obstacle states based on their current speed and heading as well as past histories for next pred_horizn - 1 steps
    obs_state_pred = pred_obs_traj(
        obs_curr_state, dt, pred_horizn
    ) # Numpy array of obstacle state for each step in the prediction horizon
    
    # Solve NMPC
    opt_control, opt_states, min_h_values = nmpc_node_robot.solve_nmpc(
        rb_curr_state=rb_curr_state,
        walls=walls, # Static obstacles as walls in the environment
        obs_state_pred=obs_state_pred, # Current and predicted dynamic obstacle positions
    )

    # Apply the first control input. Basically store first input and state as well as all the predicted states.
    rb_curr_state = opt_states[:, 0]
    # Update obstacle states based on their dynamics (constant velocity model here)
    obs_curr_state = nmpc_node_robot.obstacle_dynamics(obs_curr_state, dt).full()

    # Store states for plotting and storing
    rb_state_hist.append(rb_curr_state)  # Store current state for plotting
    obs_state_hist.append(obs_curr_state)  # Store current obstacle state for plotting
    dist_to_goal = np.linalg.norm(rb_curr_state[:2] - rb_goal[:2])
    
    if dist_to_goal <= 0.35:
        print("Distance to goal:", dist_to_goal)
        print("Goal Reached!")
        goal_reached = True

    # Plotting and storing
    plot_states_controls(
        pred_horizn=pred_horizn,
        ctrl_horizn=ctrl_horizn,
        opt_states_0=opt_states,
        opt_control_0=opt_control,
        start=rb_start,
        goal=rb_goal,
        state_hist_rb=rb_state_hist,
        state_hist_obs=obs_state_hist,
        sampling_time=dt,
        v_max=v_max,
        omega_max=omega_max,
        num_obstacles=num_obs,
        obs_centers_pred=obs_state_pred,  # current and predicted positions
        buffer_dist=buffer_dist,
        obs_radius=obs_radius,
        min_safe_dist=min_safe_dist,
        min_h_values=min_h_values,
        save_path="plots",
        step=len(store_opt_states) if store_opt_states is not None else 0,
    )

    store_opt_states.append(opt_states)
    store_opt_ctrls.append(opt_control)
