import numpy as np
import casadi as cas
from dynamics import unicycle_dynamics
import time
import math as m


def pred_obs_traj(obs_curr_state, dt, pred_horizn):
    """
    Predict the trajectory of an obstacle assuming it moves in a straight line with constant velocity.
    obs_curr_state: Current state of the obstacle [x, x_var, y, y_var, theta, v, omega]
    dt: Time step for prediction
    pred_horizn: Number of prediction steps

    Returns:
        obs_state_pred: Numpy array of shape (pred_horizn, 7) containing (x, x_var, y, y_var, thetha).
    """
    obs_state_pred = [obs_curr_state.copy()]  # Initialize with current position (x, y)
    # obs_next_state = obs_curr_state.copy()
    obs_next_state = np.zeros_like(obs_curr_state)
    for _ in range(1, pred_horizn):
        # Update state: x = x + v*cos(theta)*dt, y = y + v*sin(theta)*dt, theta = theta + omega*dt
        obs_next_state[:, 0] = obs_curr_state[:, 0] + obs_curr_state[:, 5] * np.cos(obs_curr_state[:, 4]) * dt
        obs_next_state[:, 2] = obs_curr_state[:, 2] + obs_next_state[:, 5] * np.sin(obs_next_state[:, 4]) * dt
        obs_next_state[:, 4] = obs_curr_state[:, 4] + obs_curr_state[:, 6] * dt
        # Keep variances and velocities constant for simplicity
        obs_next_state[:, 1] = obs_curr_state[:, 1]  # Constant x variance
        obs_next_state[:, 3] = obs_curr_state[:, 3]  # Constant y variance
        obs_next_state[:, 5] = obs_curr_state[:, 5]  # Constant speed
        obs_next_state[:, 6] = obs_curr_state[:, 6]  # Constant angular velocity

        obs_state_pred.append(obs_next_state.copy())
        obs_curr_state = obs_next_state.copy()

    return np.array(obs_state_pred)  # Shape: (pred_horizn, num_obs, 7)


class nmpc_node:
    def __init__(
        self, num_states, num_ctrls, pred_horizn, ctrl_horizn, start, goal,
        v_max, v_min, omega_max, sampling_time,
        state_running_cost, control_running_cost, state_terminal_cost, num_obs,
        buffer_dist, min_safe_dist
    ):
        self.num_states = num_states              # Number of states
        self.num_ctrls = num_ctrls                  # Control inputs
        self.pred_horizn = pred_horizn                # Prediction horizon
        self.ctrl_horizn = ctrl_horizn                # Control horizon
        self.start = start
        self.goal = goal            # Goal location
        self.max_velocity = v_max
        self.min_velocity = v_min
        self.max_angular_velocity = omega_max
        self.dt = sampling_time
        # Obstacle parameters
        self.num_obs = num_obs
        self.buffer_dist = buffer_dist
        self.min_safe_dist = min_safe_dist
        # Constants defined
        self.state_running_cost = state_running_cost      # Weights for tracking reference state [x, y, θ]
        self.control_running_cost = control_running_cost      # Weights for control effort [v, ω] - Keep this!
        self.state_terminal_cost = state_terminal_cost    # Weights for final state deviation from goal


    def solve_nmpc(self, rb_curr_state, obs_state_pred):
        
        self.rb_curr_state = rb_curr_state
        self.obs_state_pred = obs_state_pred

        # Parameter vector 'p' now contains initial state
        # self.rb_params = cas.SX.sym(
        #     'p'
        # ) 
        
        # Decision variables declared and 
        self.num_decision_vars = self.num_ctrls * self.ctrl_horizn + self.num_states * self.pred_horizn
        
        self.Z = cas.SX.sym('Z', self.num_decision_vars)

        # Controls and predicted states reshaped
        self.pred_controls = cas.reshape(
            self.Z[0:self.num_ctrls * self.ctrl_horizn], # First num_ctrls * ctrl_horizn Z variables are controls
            self.num_ctrls,
            self.ctrl_horizn,
        )

        self.pred_states = cas.reshape(
            self.Z[self.num_ctrls * self.ctrl_horizn :], # Remaining num_states * pred_horizn Z variables are states
            self.num_states,
            self.pred_horizn,
        )
        
        # Define cost objective
        self.cost_objective()
        # print(self.horizn_cost)
        
        # Define constraints
        self.get_constraints()
        # print(self.constraints)
        
        # Solve NMPC problem
        self.opt_controls, self.opt_states = self.solver()
        # print("optimal control:")
        # print(self.opt_controls)
        # print("optimal states:")
        # print(self.opt_states)
        
        
        # --- Calculate CBF (h) values along the optimal trajectory ---

        # h_values = np.zeros((self.num_obs, self.pred_horizn))
        # for k in range(self.pred_horizn):  # Iterate over each time step of the optimal trajectory X_opt
        #     pos_k_opt = self.opt_states[0:2, k]  # Optimal position at time step k
        #     for obs_idx in range(self.num_obs):
        #         obs_center = self.obs_state_pred[obs_idx][k]  # Obstacle center at time step k
        #         # Calculate h = ||pos_opt - obs_center||^2 - min_dist_sq
        #         min_dist_sq = self.min_safe_dist**2 # Obstacle_radius + safe_distance
        #         dist_sq = cas.sumsqr(pos_k_opt - obs_center) # Squared distance from obstacle center
        #         h_k_obs = dist_sq - min_dist_sq          # Barrier function value for this obstacle
        #         h_values[obs_idx, k] = h_k_obs

        # # Find the minimum h value across all obstacles at each time step
        # min_h_values = np.min(h_values, axis=0)
        # min_overall_h = np.min(min_h_values) # Smallest h value achieved over the entire trajectory
        # print(f"Minimum CBF value (h_min) reached during trajectory: {min_overall_h:.4f}")

        # if min_overall_h < -1e-4: # Allow small numerical tolerance
        #     print("WARNING: Safety constraint (h >= 0) appears to be violated!")
        # else:
        #     print("Safety constraint (h >= 0) appears satisfied.")
        
        # return self.opt_controls, self.opt_states, min_h_values

    def cost_objective(self):
        # --- Define the NEW cost function (Trajectory Tracking) ---
        self.horizn_cost = cas.SX(0)

        # Control cost
        for k in range(self.ctrl_horizn):
            # Control Cost (penalize effort)
            control = self.pred_controls[:, k] # Shape: (num_ctrls, 1)
            # Shape: const = (1, num_ctrls) x (2, 1) x (num_ctrls, 1)
            self.horizn_cost += control.T @ self.control_running_cost @ control

        # State error cost
        for k in range(self.pred_horizn): # Iterate through the prediction horizon for states
            # State Cost (penalize deviation from REFERENCE trajectory X_ref)
            state_error_xy_sq = (self.pred_states[0:2, k] - self.goal[0:2])** 2
            
            state_cost = (
                self.state_running_cost[0] * state_error_xy_sq[0] 
                + self.state_running_cost[1] * state_error_xy_sq[1]
            )
            self.horizn_cost += state_cost


    def get_constraints(self):
        large_number = cas.inf # Use CasADi infinity for bounds
        # Constraints on decision variables (Z)
        # Initialize large upper and lower bounds for decision variables (Z) 
        self.lbz = np.full((self.num_decision_vars), -large_number)  # Lower bounds for Z
        self.ubz = np.full((self.num_decision_vars), large_number)   # Upper bounds for Z

        # For num
        # General input constraints: 0 <= v_k <= v_max, -ω_max <= ω_k <= ω_max 
        for k in range(self.ctrl_horizn):
            # Linear velocity
            self.lbz[k * self.num_ctrls] = self.min_velocity        # v_k lower bound (non-negative velocity)
            self.ubz[k * self.num_ctrls] = self.max_velocity    # v_k upper bound
            # Angular velocity
            self.lbz[k * self.num_ctrls + 1] = -self.max_angular_velocity    # ω_k lower bound
            self.ubz[k * self.num_ctrls + 1] = self.max_angular_velocity     # ω_k upper bound
        
        # Define other additional nonlinear constraints 
        self.g = [] # Constraint vector. This defines constraints on function values g(x) as lbg <= g(x) <= ubg. Meaning constraints are on function values, not decision variables directly.
        self.lbg = [] # Lower bounds for additional constraints
        self.ubg = [] # Upper bounds for additional constraints

        # 1. Dynamics Constraints (x_{k+1} = f(x_k, u_k)) - Uses symbolic x0 from p
        for k in range(self.pred_horizn):

            x_curr_step = (
                self.pred_states[:, k - 1] if k > 0 else self.rb_curr_state
            ) # State at start of interval k
            u_curr_step = self.pred_controls[:, k]                   # Control during interval k

            x_next_pred = unicycle_dynamics(x_curr_step, u_curr_step, self.dt) # Predicted state at end of interval
            self.g.append(self.pred_states[:, k] - x_next_pred) # Constraint: x_{k+1} - f(x_k, u_k) = 0

            self.lbg.extend([0.0] * self.num_states) # Equality constraint: lower bound = 0
            self.ubg.extend([0.0] * self.num_states) # Equality constraint: upper bound = 0


        # # 2. Control Barrier Function (CBF) Constraints - NO CHANGE
        for k in range(1): # For each time step in the prediction horizon
            pos_k = self.pred_states[0:2, k] # Predicted position [x_{k+1}, y_{k+1}] at the END of step k
            for obs_idx in range(self.num_obs): # For each obstacle
                obs_center = self.obs_state_pred[obs_idx][k] # Obstacle center at time step k
                min_dist_sq = self.min_safe_dist**2
                dist_sq = cas.sumsqr(pos_k - obs_center) # Squared distance from obstacle center
                h_k_obs = dist_sq - min_dist_sq          # Barrier function value for this obstacle
                self.g.append(h_k_obs)
                self.lbg.append(0.0)
                self.ubg.append(large_number)
            

        # 3. Chance Constraints for dynamic obstacles 
        for k in range(1, self.pred_horizn):
            rb_pos = self.pred_states[0:2, k]
            obs_pos = self.obs_state_pred[:, 0:2, k] # Shape: (num_obs, 2)
            obs_var = self.obs_state_pred[:, 3, k]   # Shape: (num_obs,)
            for obs_idx in range(self.num_obs):
                obs_center = obs_pos[obs_idx]
                variance = obs_var[obs_idx]
                # Calculate the squared distance between robot and obstacle center
                dist_sq = cas.sumsqr(rb_pos - obs_center)
                # Define the chance constraint: P(||rb_pos - obs_center|| >= min_safe_dist) >= 0.95
                # Using the Gaussian assumption, this translates to:
                # (||rb_pos - obs_center||^2) >= (min_safe_dist + z_alpha * sqrt(variance))^2
                z_alpha = 1.645 # z-score for 95% confidence
                adjusted_safe_dist = self.min_safe_dist + z_alpha * cas.sqrt(variance)

        self.g = cas.vertcat(*self.g) # Stack constraints into a single vector
        self.lbg = np.array(self.lbg)
        self.ubg = np.array(self.ubg)    
            

    def solver(self):

        # --- Define the nonlinear programming problem (NLP) ---
        nlp = {
            'x': self.Z,      # Decision variables
            'p': self.P,     # Parameters (includes current and predicted mean and variance of obstacles)
            'f': self.horizn_cost,   # Objective function (now tracks X_ref)
            'g': self.constraints       # Constraints
        }

        # --- Create the solver ---
        solver_opts = {'ipopt': {'print_level': 3, 'sb': 'yes', 'max_iter': 3000, 'tol': 1e-4}} # Inc max_iter
        solver = cas.nlpsol('solver', 'ipopt', nlp, solver_opts)

        # Solve the optimization problem
        # Initial guess (zeros is usually okay, but reference might be better)
        guess = np.zeros(
            self.num_ctrls * self.ctrl_horizn + self.num_states * self.pred_horizn
        )


        print("\n--- Starting Solver ---")
        start_time = time.time()
        
        # Pass the p_value containing x_current and X_ref_traj
        sol = solver(
            x0=guess,
            p=self.ref_waypoints_params_value,
            lbx=self.lbz,
            ubx=self.ubz,
            lbg=self.lbg,
            ubg=self.ubg,
        )
        solve_time = time.time() - start_time
        Z_opt = sol['x'].full().flatten()
        stats = solver.stats()

        if stats['success']:
            print(f"Solver found optimal solution in {solve_time:.2f} seconds.")
        else:
            print(
                f"Solver finished with status: {stats['return_status']} in {solve_time:.2f} seconds."
            )

        # --- Extract optimal solution ---
        opt_controls = Z_opt[0 : self.num_ctrls * self.ctrl_horizn].reshape(
            self.ctrl_horizn, self.num_ctrls
        ).T
        opt_states = Z_opt[self.num_ctrls * self.ctrl_horizn :].reshape(
            self.pred_horizn, self.num_states
        ).T
        # opt_states = np.hstack((x_current.reshape(nx, 1), X_opt)) # Prepend initial state

        return opt_controls, opt_states


    
        

