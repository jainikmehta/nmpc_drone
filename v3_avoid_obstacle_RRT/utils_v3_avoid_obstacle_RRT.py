import numpy as np
import casadi as cas
from dynamics_v3_avoid_obstacle_RRT import unicycle_dynamics
import matplotlib.pyplot as plt
import time

# --- NEW: RRT Planner Implementation ---
class RRTPlanner:
    """
    A simple RRT planner.
    """
    class Node:
        """
        RRT Node class.
        """
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.parent = None

    def __init__(self, start, goal, obstacle_centers, min_dist_from_center, map_limits, step_size=0.3, max_iter=1000, goal_sample_rate=0.1):
        self.start = self.Node(start[0], start[1])
        self.goal = self.Node(goal[0], goal[1])
        self.obstacle_centers = obstacle_centers
        self.min_dist_from_center = min_dist_from_center
        self.map_x_min, self.map_x_max, self.map_y_min, self.map_y_max = map_limits
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.node_list = []

    def plan(self):
        """
        Plans the path from start to goal.
        """
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self._get_random_node()
            nearest_ind = self._get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self._steer(nearest_node, rnd_node, self.step_size)

            if self._is_collision_free(new_node.x, new_node.y, nearest_node.x, nearest_node.y):
                new_node.parent = nearest_node
                self.node_list.append(new_node)

                if self._calc_dist_to_goal(new_node.x, new_node.y) <= self.step_size:
                    final_node = self._steer(new_node, self.goal, self.step_size)
                    if self._is_collision_free(final_node.x, final_node.y, new_node.x, new_node.y):
                        final_node.parent = new_node
                        return self._generate_final_path(final_node)
        
        print("RRT failed to find a path within the iteration limit.")
        return None  # Path not found

    def _generate_final_path(self, goal_node):
        path = [[self.goal.x, self.goal.y]]
        node = goal_node
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path[::-1]

    def _get_random_node(self):
        if np.random.rand() > self.goal_sample_rate:
            rnd = self.Node(np.random.uniform(self.map_x_min, self.map_x_max),
                            np.random.uniform(self.map_y_min, self.map_y_max))
        else:  # goal-biasing
            rnd = self.Node(self.goal.x, self.goal.y)
        return rnd

    def _get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        return dlist.index(min(dlist))

    def _steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self._calc_distance_and_angle(new_node, to_node)

        if extend_length > d:
            extend_length = d

        new_node.x += extend_length * np.cos(theta)
        new_node.y += extend_length * np.sin(theta)
        new_node.parent = from_node
        return new_node

    def _calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta

    def _calc_dist_to_goal(self, x, y):
        return np.hypot(x - self.goal.x, y - self.goal.y)

    def _is_collision_free(self, new_x, new_y, prev_x, prev_y):
        points = np.linspace([prev_x, prev_y], [new_x, new_y], num=15)
        for p in points:
            for obs_center in self.obstacle_centers:
                if np.hypot(p[0] - obs_center[0], p[1] - obs_center[1]) <= self.min_dist_from_center:
                    return False
        return True

# --- NEW: RRT-based Reference Generator ---
class RRTReferenceGenerator:
    """
    Generates a reference trajectory using a one-time RRT plan.
    """
    def __init__(self, start, goal, pred_horizn, dt, v_max,
                 obstacle_centers, min_dist_from_center, map_limits):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.pred_horizn = pred_horizn
        self.dt = dt
        self.v_max = v_max
        self.obstacle_centers = obstacle_centers
        self.min_dist_from_center = min_dist_from_center
        self.map_limits = map_limits
        
        self.global_path = self._plan_global_path()
        if self.global_path is None:
            raise Exception("RRT planner failed to find a valid path.")
            
        self.path_index = 0

    def _plan_global_path(self):
        """Calls the RRT planner and processes the path."""
        print("Planning global path with RRT...")
        rrt_planner = RRTPlanner(
            start=self.start,
            goal=self.goal,
            obstacle_centers=self.obstacle_centers,
            min_dist_from_center=self.min_dist_from_center,
            map_limits=self.map_limits
        )
        path = rrt_planner.plan()
        
        if path:
            print("RRT path found. Smoothing and interpolating...")
            smoothed_path = self._smooth_path(path)
            interpolated_path = self._interpolate_path(np.array(smoothed_path))
            print(f"Global path has {len(interpolated_path)} points.")
            return interpolated_path
        return None

    def _smooth_path(self, path, iterations=100):
        """Simple path shortcutting."""
        spath = list(path)
        for _ in range(iterations):
            if len(spath) <= 2: break
            i = np.random.randint(0, len(spath) - 1)
            j = np.random.randint(i + 1, len(spath))
            if abs(i-j) <=1: continue

            start_node, end_node = spath[i], spath[j]
            if self._is_direct_path_collision_free(start_node, end_node):
                new_spath = []
                for k in range(len(spath)):
                    if k <= i or k >=j:
                         new_spath.append(spath[k])
                spath = new_spath
        return spath

    def _is_direct_path_collision_free(self, p1, p2):
        points = np.linspace(p1, p2, num=20)
        for p in points:
            for obs_center in self.obstacle_centers:
                if np.hypot(p[0] - obs_center[0], p[1] - obs_center[1]) <= self.min_dist_from_center:
                    return False
        return True
        
    def _interpolate_path(self, path):
        """Interpolate path points to be a certain distance apart."""
        new_path = []
        step_dist = self.v_max * self.dt * 0.8  # Target distance between points
        for i in range(len(path) - 1):
            start_p, end_p = path[i], path[i+1]
            dist = np.linalg.norm(end_p - start_p)
            num_points = int(np.ceil(dist / step_dist))
            if num_points > 0:
                x_interp = np.linspace(start_p[0], end_p[0], num_points, endpoint=False)
                y_interp = np.linspace(start_p[1], end_p[1], num_points, endpoint=False)
                for j in range(len(x_interp)):
                    new_path.append([x_interp[j], y_interp[j]])
        new_path.append(path[-1].tolist())
        return np.array(new_path)

    def generate_waypoints(self, current_state):
        """
        Provides the next N waypoints from the global path based on the robot's current position.
        """
        current_pos = current_state[:2]
        distances = np.linalg.norm(self.global_path - current_pos, axis=1)
        self.path_index = np.argmin(distances)

        waypoints = []
        for i in range(self.pred_horizn):
            target_idx = min(self.path_index + i, len(self.global_path) - 1)
            waypoint_pos = self.global_path[target_idx]
            
            # Determine orientation by looking ahead
            lookahead_idx = min(target_idx + 1, len(self.global_path) - 1)
            next_pos = self.global_path[lookahead_idx]
            direction_vector = next_pos - waypoint_pos
            
            # Avoid zero vector for orientation at the goal
            if np.linalg.norm(direction_vector) < 1e-6:
                theta = self.goal[2]
            else:
                theta = np.arctan2(direction_vector[1], direction_vector[0])
            
            waypoints.append([waypoint_pos[0], waypoint_pos[1], theta])
        
        return np.array(waypoints)


class nmpc_node:
    # --- This class remains unchanged ---
    def __init__(self, num_states, num_controls,
                 pred_horizn, ctrl_horizn, start,
                 max_velocity, min_velocity, max_angular_velocity,
                 sampling_time, Q_running, R_running, Q_terminal,
                 num_obstacles, obstacle_centers, safe_distance, min_dist_from_center):
        # ... (original code)
        self.num_states = num_states
        self.num_ctrls = num_controls
        self.pred_horizn = pred_horizn
        self.ctrl_horizn = ctrl_horizn
        self.start = start
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.max_angular_velocity = max_angular_velocity
        self.dt = sampling_time
        self.num_obstacles = num_obstacles
        self.obstacle_centers = obstacle_centers
        self.safe_distance = safe_distance
        self.min_dist_from_center = min_dist_from_center
        self.Q_running = Q_running
        self.R_running = R_running
        self.Q_terminal = Q_terminal

    def cost_objective(self):
        # --- This method remains unchanged ---
        self.horizn_cost = cas.SX(0)
        for k in range(self.pred_horizn):
            state_error_xy = self.pred_states[0:2, k] - self.ref_waypoints_params[0:2, k]
            theta_error = self.pred_states[2, k] - self.ref_waypoints_params[2, k]
            theta_error_wrapped = cas.atan2(cas.sin(theta_error), cas.cos(theta_error))
            state_cost = self.Q_running[0,0]*state_error_xy[0]**2 + \
                         self.Q_running[1,1]*state_error_xy[1]**2 + \
                         self.Q_running[2,2]*theta_error_wrapped**2
            self.horizn_cost += state_cost

    def get_constraints(self,):
        # --- This method remains unchanged ---
        large_number = cas.inf
        self.lbz = np.full((self.num_ctrls * self.ctrl_horizn + self.num_states * self.pred_horizn,), -large_number)
        self.ubz = np.full((self.num_ctrls * self.ctrl_horizn + self.num_states * self.pred_horizn,), large_number)
        for k in range(self.ctrl_horizn):
            self.lbz[k*self.num_ctrls + 0] = self.min_velocity
            self.ubz[k*self.num_ctrls + 0] = self.max_velocity
            self.lbz[k*self.num_ctrls + 1] = -self.max_angular_velocity
            self.ubz[k*self.num_ctrls + 1] = self.max_angular_velocity
        self.constraints = []
        self.lb_constraints = []
        self.ub_constraints = []
        for k in range(self.pred_horizn):
            x_curr_step = self.pred_states[:, k-1] if k > 0 else self.current_state
            u_curr_step = self.pred_controls[:, k]
            x_next_pred = unicycle_dynamics(x_curr_step, u_curr_step, self.dt)
            self.constraints.append(self.pred_states[:, k] - x_next_pred)
            self.lb_constraints.extend([0.0] * self.num_states)
            self.ub_constraints.extend([0.0] * self.num_states)
        for k in range(self.pred_horizn):
            pos_k = self.pred_states[0:2, k]
            for obs_idx in range(self.num_obstacles):
                obs_center = self.obstacle_centers[obs_idx, :]
                min_dist_sq = self.min_dist_from_center**2
                dist_sq = cas.sumsqr(pos_k - obs_center)
                h_k_obs = dist_sq - min_dist_sq
                self.constraints.append(h_k_obs)
                self.lb_constraints.append(0.0)
                self.ub_constraints.append(large_number)
        self.constraints = cas.vertcat(*self.constraints)
        self.lb_constraints = np.array(self.lb_constraints)
        self.ub_constraints= np.array(self.ub_constraints)

    def solver(self):
        # --- This method remains unchanged ---
        nlp = {'x': self.Z, 'p': self.ref_waypoints_params, 'f': self.horizn_cost, 'g': self.constraints}
        solver_opts = {'ipopt': {'print_level': 0, 'sb': 'yes', 'max_iter': 3000, 'tol': 1e-4}} # Quieter solver
        solver = cas.nlpsol('solver', 'ipopt', nlp, solver_opts)
        guess = np.zeros(self.num_ctrls * self.ctrl_horizn + self.num_states * self.pred_horizn)
        start_time = time.time()
        sol = solver(x0=guess, p=self.ref_waypoints_params_value, lbx=self.lbz, ubx=self.ubz, lbg=self.lb_constraints, ubg=self.ub_constraints)
        solve_time = time.time() - start_time
        Z_opt = sol['x'].full().flatten()
        stats = solver.stats()
        if stats['success']:
            # print(f"Solver found optimal solution in {solve_time:.3f} seconds.")
            pass
        else:
            print(f"Solver finished with status: {stats['return_status']} in {solve_time:.3f} seconds.")
        opt_controls = Z_opt[0:self.num_ctrls * self.ctrl_horizn].reshape(self.ctrl_horizn, self.num_ctrls).T
        opt_states = Z_opt[self.num_ctrls * self.ctrl_horizn:].reshape(self.pred_horizn, self.num_states).T
        return opt_controls, opt_states

    def solve_nmpc(self, ref_waypoints, current_state):
        # --- This method remains unchanged ---
        self.ref_waypoints_params = cas.SX.sym('p', self.num_states, self.pred_horizn)
        self.ref_waypoints_params_value = ref_waypoints.T
        self.current_state = current_state
        self.Z = cas.SX.sym('Z', self.num_ctrls * self.ctrl_horizn + self.num_states * self.pred_horizn)
        self.pred_controls = cas.reshape(self.Z[0:self.num_ctrls * self.ctrl_horizn], self.num_ctrls, self.ctrl_horizn)
        self.pred_states = cas.reshape(self.Z[self.num_ctrls * self.ctrl_horizn:], self.num_states, self.pred_horizn)
        self.cost_objective()
        self.get_constraints()
        self.opt_controls, self.opt_states = self.solver()
        h_values = np.zeros((self.num_obstacles, self.pred_horizn))
        for k in range(self.pred_horizn):
            pos_k_opt = self.opt_states[0:2, k]
            for obs_idx in range(self.num_obstacles):
                obs_center = self.obstacle_centers[obs_idx, :]
                min_dist_sq = self.min_dist_from_center**2
                dist_sq = cas.sumsqr(pos_k_opt - obs_center)
                h_k_obs = dist_sq - min_dist_sq
                h_values[obs_idx, k] = h_k_obs
        min_h_values = np.min(h_values, axis=0)
        min_overall_h = np.min(min_h_values)
        if min_overall_h < -1e-4:
            print(f"WARNING: Safety constraint (h >= 0) may be violated! h_min = {min_overall_h:.4f}")
        return self.opt_controls, self.opt_states, min_h_values