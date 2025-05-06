import numpy as np
import casadi as cas

class obstacles_2d:

    def create_obstacles_in_map():
    
        obstacle_centers = np.random.rand(num_obstacles, 2) * 1.0 # Spread in [0, 1] box more broadly

        # Refilter based on start/goal
        min_start_goal_dist = 0.15 # Allow slightly closer obstacles
        valid_centers = []
        for center in obstacle_centers:
             if np.linalg.norm(center - x_current[:2]) > min_start_goal_dist and \
                np.linalg.norm(center - x_goal[:2]) > min_dist_from_center + 0.05: # Ensure goal isn't inside safety zone
                 valid_centers.append(center)
        obstacle_centers = np.array(valid_centers)
        num_obstacles = obstacle_centers.shape[0]

        print(f"Using {num_obstacles} obstacles.")

        # Pre-calculate minimum squared distances from centers
        min_dist_sq_array = np.full(num_obstacles, min_dist_from_center**2)


class ref_generator_2d: 

    def __init__(self, start, goal, max_velocity):
        self.start = start  # Initial position of robot [x_0, y_0, theta_0]
        self.goal = goal # Final position of robot [x_g, y_g, theta_g]
        self.max_velocity = max_velocity

    def generate_waypoints(self):
        waypoints = []
        current_pose = self.start[:2]
        goal_pose = self.goal[:2]
        max_vel = self.max_velocity

        direction = goal_pose - current_pose
        distance_to_goal = np.linalg.norm(direction)
        if distance_to_goal <= max_vel:
            return np.array(waypoints) # No waypoint can be created

        distance_to_goal_units = np.floor(distance_to_goal/max_vel)
        if distance_to_goal_units % 2 == 0:
            num_waypoints = int(distance_to_goal_units - 1)
        else:
            num_waypoints = int(distance_to_goal_units)

        step = max_vel * direction / distance_to_goal
        for i in range(num_waypoints):
            current_pose += step
            waypoints.append(current_pose.copy())

        return np.array(waypoints)# [[x_1, y_1][x_2, y_2],...,[x_g-1, y_g-1]]


# class constraint:

#     # --- Define constraints ---
#     g = [] # Constraint vector
#     lbg = [] # Lower bounds for constraints
#     ubg = [] # Upper bounds for constraints

#     # 1. Dynamics Constraints (x_{k+1} = f(x_k, u_k)) - Uses symbolic x0 from p
#     for k in range(N):
#         x_curr_step = X[:, k-1] if k > 0 else x0 # State at start of interval k
#         u_curr_step = U[:, k]                   # Control during interval k
#         x_next_pred = dynamics(x_curr_step, u_curr_step) # Predicted state at end of interval
#         g.append(X[:, k] - x_next_pred) # Constraint: x_{k+1} - f(x_k, u_k) = 0
#         lbg.extend([0.0] * nx) # Equality constraint: lower bound = 0
#         ubg.extend([0.0] * nx) # Equality constraint: upper bound = 0

#     # 2. Control Barrier Function (CBF) Constraints - NO CHANGE
#     for k in range(N): # For each time step in the prediction horizon
#         pos_k = X[0:2, k] # Predicted position [x_{k+1}, y_{k+1}] at the END of step k
#         for obs_idx in range(num_obstacles): # For each obstacle
#             obs_center = obstacle_centers[obs_idx, :]
#             min_dist_sq = min_dist_sq_array[obs_idx]
#             dist_sq = cas.sumsqr(pos_k - obs_center) # Squared distance from obstacle center
#             h_k_obs = dist_sq - min_dist_sq          # Barrier function value for this obstacle
#             g.append(h_k_obs)
#             lbg.append(0.1)
#             ubg.append(large_number)

#     # --- Combine constraints and bounds ---
#     # REMOVE the explicit final state constraint if using terminal cost
#     g.append(X[:, N-1] - x_goal) # Goal constraint X_N = x_goal
#     lbg.extend([-1e-3] * nx) # Allow small tolerance
#     ubg.extend([1e-3] * nx)

#     g = cas.vertcat(*g) # Stack constraints into a single vector
#     lbg = np.array(lbg)
#     ubg = np.array(ubg)


class main_node:


    def __init__(self, num_states, num_control, pred_horizn, ctrl_horizn, start, goal, ref_waypoints):
        self.num_states = num_states              # Number of states
        self.num_ctrl = num_ctrl            # Control inputs
        self.pred_horizn = pred_horizn                # Prediction horizon
        self.ctrl_horizn = ctrl_horizn                # Control horizon
        self.start = start      # Initial location
        self.goal = goal            # Goal location
        self.ref_waypoints = ref_waypoints     # Reference waypoints

        # --- IMPORTANT: Parameters ---
        # Parameter vector 'p' now contains initial state AND reference waypoints
        self.param = cas.SX.sym('p', self.num_state, self.pred_horizn)  # [number of states; current_position + prediction horizon]

        # Decision variables: control U and predicted states X flattened
        self.Z = cas.SX.sym('Z', self.nu * self.Np + self.nx * self.N)  # [vec(U); vec(X)]

        # Create the parameter vector value including the reference trajectory
        self.param_value = np.insert(self.X_ref_traj, 0, self.x_initial, axis=1)

        # --- Extract variables from Z and p ---
        self.x0 = self.p[0:nx, 0]                        # Symbolic initial state
        self.X_ref = self.p[:, 1:]                       # Symbolic reference trajectory (vectorized)

        self.U = cas.reshape(Z[0:nu * N], nu, N)         # Symbolic Controls u0 to u_{N-1}
        self.X = cas.reshape(Z[nu * N:], nx, N)          # Symbolic States x1 to xN


    def cost_objective(self):
        # --- Define the NEW cost function (Trajectory Tracking) ---
        self.cost = cas.SX(0)
        self.Q_running = cas.diag([5.0, 5.0, 0.5])       # Weights for tracking reference state [x, y, θ]
        self.R_running = cas.diag([0.5, 0.5])            # Weights for control effort [v, ω] - Keep this!
        self.Q_terminal = cas.diag([50.0, 50.0, 50.0])   # Weights for final state deviation from goal

        for k in range(self.N):
            # Control Cost (penalize effort)
            self.control = self.U[:, k]
            self.cost += self.control.T @ self.R_running @ self.control

            # State Cost (penalize deviation from REFERENCE trajectory X_ref)
            self.state_error = self.X[:, k] - self.X_ref[:, k]
            # # Handle angle wrapping for theta error (more robust)
            # theta_error = state_error[2] # X[2, k] - X_ref[2, k]
            # # Use atan2 for robust wrapping: atan2(sin(angle), cos(angle)) = angle in [-pi, pi]
            # theta_error_wrapped = cas.atan2(cas.sin(theta_error), cas.cos(theta_error))

            # Combine state costs (using weights from Q_running)
            # Note: We use the wrapped theta error directly here.
            state_cost = Q_running[0,0]*state_error[0]**2 + \
                            Q_running[1,1]*state_error[1]**2 
                        #  Q_running[2,2]*theta_error_wrapped**2
            cost += state_cost

            # Add terminal cost (deviation from final goal x_goal)
            # This helps ensure the robot reaches the actual desired goal state,
            # even if the reference trajectory doesn't end exactly there or isn't perfect.
            # terminal_state_error = X[:, N-1] - x_goal
            # terminal_theta_error = terminal_state_error[2]
            # terminal_theta_error_wrapped = cas.atan2(cas.sin(terminal_theta_error), cas.cos(terminal_theta_error))
            # terminal_cost = Q_terminal[0,0]*terminal_state_error[0]**2 + \
            #                 Q_terminal[1,1]*terminal_state_error[1]**2
            #                 # Q_terminal[2,2]*terminal_theta_error_wrapped**2
            # cost += 100*terminal_cost
            
    def solver(self):
        # --- Define the nonlinear programming problem (NLP) ---
        nlp = {
            'x': Z,      # Decision variables
            'p': p,      # Parameters (now includes x0 and X_ref)
            'f': cost,   # Objective function (now tracks X_ref)
            'g': g       # Constraints
        }

        # --- Define bounds for decision variables (Z) ---
        self.lbz = np.full((self.nu * self.N + self.nx * self.N,), -large_number)  # Lower bounds for Z
        self.ubz = np.full((self.nu * self.N + self.nx * self.N,), large_number)   # Upper bounds for Z

        # General Input constraints: 0 <= v_k <= v_max, -ω_max <= ω_k <= ω_max - NO CHANGE
        for k in range(N):
            self.lbz[k*self.nu + 0] = 0         # v_k lower bound (non-negative velocity)
            self.ubz[k*self.nu + 0] = v_max     # v_k upper bound
            self.lbz[k*self.nu + 1] = -ω_max    # ω_k lower bound
            self.ubz[k*self.nu + 1] = ω_max     # ω_k upper bound

        # --- Add Specific Velocity Constraints ---
        # Initial Velocity Constraint: U[:, 0] = [0, 0] - NO CHANGE
        self.lbz[0] = 0.0 # v_0 lower bound
        self.ubz[0] = 0.0 # v_0 upper bound
        self.lbz[1] = 0.0 # ω_0 lower bound
        self.ubz[1] = 0.0 # ω_0 upper bound
        print(f"Constraining initial control U[:,0] (indices 0, 1) to zero.")

        # --- Create the solver ---
        solver_opts = {'ipopt': {'print_level': 3, 'sb': 'yes', 'max_iter': 3000, 'tol': 1e-4}} # Inc max_iter
        solver = cas.nlpsol('solver', 'ipopt', nlp, solver_opts)

        # --- Solve the optimization problem ---
        # Initial guess (zeros is usually okay, but reference might be better)
        Z0 = np.zeros(self.nu * self.N + self.nx * self.N)
        # Provide a guess based on reference trajectory (can sometimes help)
        # Z0[nu*N:] = X_ref_traj.ravel() # Guess states follow reference
        # Guess controls (e.g., small forward velocity) - harder to guess well
        # U_guess = np.zeros((nu, N))
        # U_guess[0, :] = 0.1 # Small forward velocity
        # Z0[:nu*N] = U_guess.ravel()

        print("\n--- Starting Solver ---")
        start_time = time.time()
        
        # Pass the p_value containing x_current and X_ref_traj
        sol = solver(x0=Z0, p=p_value, lbx=lbz, ubx=ubz, lbg=lbg, ubg=ubg)
        solve_time = time.time() - start_time
        Z_opt = sol['x'].full().flatten()
        stats = solver.stats()
        if stats['success']:
            print(f"Solver found optimal solution in {solve_time:.2f} seconds.")
        else:
            print(f"Solver finished with status: {stats['return_status']} in {solve_time:.2f} seconds.")

        # --- Extract optimal solution ---
        U_opt = Z_opt[0:nu * N].reshape(N, nu).T
        X_opt = Z_opt[nu * N:].reshape(N, nx).T
        X_plot = np.hstack((x_current.reshape(nx, 1), X_opt)) # Prepend initial state

        # --- Calculate CBF (h) values along the optimal trajectory ---
        h_values = np.zeros((num_obstacles, N))
        for k in range(N):  # Iterate over each time step of the optimal trajectory X_opt
            pos_k_opt = X_opt[0:2, k]  # Optimal position at time step k
            for obs_idx in range(num_obstacles):
                obs_center = obstacle_centers[obs_idx, :]
                min_dist_sq = min_dist_sq_array[obs_idx]
                # Calculate h = ||pos_opt - obs_center||^2 - min_dist_sq
                dist_sq_opt = np.sum((pos_k_opt - obs_center)**2)
                h_values[obs_idx, k] = dist_sq_opt - min_dist_sq

        # Find the minimum h value across all obstacles at each time step
        min_h_values = np.min(h_values, axis=0)
        min_overall_h = np.min(min_h_values) # Smallest h value achieved over the entire trajectory
        print(f"Minimum CBF value (h_min) reached during trajectory: {min_overall_h:.4f}")
        if min_overall_h < -1e-4: # Allow small numerical tolerance
            print("WARNING: Safety constraint (h >= 0) appears to be violated!")
        else:
            print("Safety constraint (h >= 0) appears satisfied.")
