# Utilities for v4_dynamic_obstacles_only_goal
# Contains NMPC node and helper functions for static wall constraints
import numpy as np
from casadi import SX, vertcat, Function, nlpsol, inf, logic_and, if_else

class NMPCNode:
    def __init__(self, num_states, num_controls, pred_horizn, ctrl_horizn, start, max_velocity, min_velocity, max_angular_velocity, sampling_time, Q_running, R_running, Q_terminal, walls, min_safe_dist):
        self.num_states = num_states
        self.num_controls = num_controls
        self.pred_horizn = pred_horizn
        self.ctrl_horizn = ctrl_horizn
        self.start = start
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.max_angular_velocity = max_angular_velocity
        self.dt = sampling_time
        self.Q_running = Q_running
        self.R_running = R_running
        self.Q_terminal = Q_terminal
        self.walls = walls
        self.min_safe_dist = min_safe_dist
        self._setup_solver()

    def _inside_wall_penalty(self, x, y):
        penalty = 0
        for wall in self.walls:
            x_min = wall['x'] - self.min_safe_dist
            x_max = wall['x'] + wall['w'] + self.min_safe_dist
            y_min = wall['y'] - self.min_safe_dist
            y_max = wall['y'] + wall['h'] + self.min_safe_dist
            # Use CasADi logic_and and if_else for symbolic logic
            inside_x = logic_and(x > x_min, x < x_max)
            inside_y = logic_and(y > y_min, y < y_max)
            inside = logic_and(inside_x, inside_y)
            penalty += 1e3 * if_else(inside, 1, 0)  # Lower penalty
        return penalty

    def _setup_solver(self):
        # Define symbolic variables
        X = SX.sym('X', self.num_states, self.pred_horizn+1) # +1 for initial state
        # Note: X[0, k] = x, X[1, k] = y, X[2, k] = theta
        U = SX.sym('U', self.num_controls, self.ctrl_horizn)
        P = SX.sym('P', self.num_states + 2) # initial state + goal (x, y)
        cost = 0
        g = []
        for k in range(self.ctrl_horizn):
            # Dynamics
            x_next = X[0, k] + U[0, k]*SX.cos(X[2, k])*self.dt
            y_next = X[1, k] + U[0, k]*SX.sin(X[2, k])*self.dt
            theta_next = X[2, k] + U[1, k]*self.dt
            g += [X[0, k+1] - x_next, X[1, k+1] - y_next, X[2, k+1] - theta_next]
            # Running cost (to goal)
            goal = P[self.num_states:self.num_states+2]
            state_err = vertcat(X[0, k] - goal[0], X[1, k] - goal[1], 0)
            cost += state_err.T @ self.Q_running @ state_err + U[:, k].T @ self.R_running @ U[:, k]
            # Soft penalty for being inside any wall
            cost += self._inside_wall_penalty(X[0, k+1], X[1, k+1])
        # Terminal cost
        goal = P[self.num_states:self.num_states+2]
        terminal_err = vertcat(X[0, self.pred_horizn] - goal[0], X[1, self.pred_horizn] - goal[1], 0)
        cost += terminal_err.T @ self.Q_terminal @ terminal_err
        # NLP
        OPT_variables = vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
        nlp_prob = {'f': cost, 'x': OPT_variables, 'g': vertcat(*g), 'p': P}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.solver = nlpsol('solver', 'ipopt', nlp_prob, opts)
        self.X = X
        self.U = U
        self.P = P
        self.num_g = len(g)

    def solve_nmpc(self, current_state, goal):
        # Initial guess
        x0 = np.tile(current_state.reshape(-1, 1), (1, self.pred_horizn+1))
        u0 = np.zeros((self.num_controls, self.ctrl_horizn))
        # Pack variables
        vars_init = np.concatenate([x0.flatten(), u0.flatten()])
        p = np.concatenate([current_state, goal])
        # Bounds
        lbx = []
        ubx = []
        for k in range(self.pred_horizn+1):
            lbx += [-inf, -inf, -inf]
            ubx += [inf, inf, inf]
        for k in range(self.ctrl_horizn):
            lbx += [self.min_velocity, -self.max_angular_velocity]
            ubx += [self.max_velocity, self.max_angular_velocity]
        lbg = [0]*self.num_g
        ubg = [0]*self.num_g
        # Solve
        sol = self.solver(x0=vars_init, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        sol_x = sol['x'].full().flatten()
        opt_states = sol_x[:self.num_states*(self.pred_horizn+1)].reshape(self.num_states, self.pred_horizn+1)
        opt_controls = sol_x[self.num_states*(self.pred_horizn+1):].reshape(self.num_controls, self.ctrl_horizn)
        return opt_controls, opt_states
