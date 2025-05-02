import casadi as cas

# Define the dynamics function (discrete time) - NO CHANGE
def dynamics_unicycle(x, u, dt):
    return cas.vertcat(
        x[0] + dt * u[0] * cas.cos(x[2]),  # x_{k+1}
        x[1] + dt * u[0] * cas.sin(x[2]),  # y_{k+1}
        x[2] + dt * u[1]                    # θ_{k+1}
    )
