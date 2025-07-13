import casadi as cas

# Define the dynamics function (discrete time) - NO CHANGE
def unicycle_dynamics(x, u, dt): # First order unicycle dynamics
    return cas.vertcat(
        x[0] + dt * u[0] * cas.cos(x[2]),  # x_{k+1}
        x[1] + dt * u[0] * cas.sin(x[2]),  # y_{k+1}
        x[2] + dt * u[1]                    # θ_{k+1}
    )


def obstacle_dynamics(obs_state, dt):
    """
    Predict the next state of obstacles based on their current state and speed.
    obs_state: numpy array of shape (num_obs, 4) where each row is [x, y, heading, speed]
    dt: time step for prediction
    """
    num_obs = obs_state.shape[0]
    obs_next = obs_state.copy()
    
    # Update each obstacle's position based on its speed and heading
    for i in range(num_obs):
        obs_next[i, 0] += dt * obs_state[i, 3] * cas.cos(obs_state[i, 2])  # x_{k+1}
        obs_next[i, 1] += dt * obs_state[i, 3] * cas.sin(obs_state[i, 2])  # y_{k+1}
    
    return obs_next

# # Define the 2nd order unicycle dynamics (discrete time)
# # x = [x, y, theta, v, omega]
# # u = [a_v, a_omega] (linear and angular acceleration)
# def unicycle_2nd_order_dynamics(x, u, dt):
#     x_next = x[0] + dt * x[3] * cas.cos(x[2])
#     y_next = x[1] + dt * x[3] * cas.sin(x[2])
#     theta_next = x[2] + dt * x[4]
#     v_next = x[3] + dt * u[0]
#     omega_next = x[4] + dt * u[1]
#     return cas.vertcat(x_next, y_next, theta_next, v_next, omega_next)
