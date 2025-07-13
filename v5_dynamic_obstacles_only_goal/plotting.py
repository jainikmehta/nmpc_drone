import matplotlib
matplotlib.use('TkAgg') # Or 'Qt5Agg', 'QtAgg', 'wxAgg', 'MacOSX', etc.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle # Import Circle patch

# import seaborn as sb

def plot_states_controls(
    pred_horizn, ctrl_horizn, opt_states_0, opt_control_0, 
    start, goal, ref_waypoints, sampling_time, v_max, omega_max,
    num_obstacles, obs_state, obs_centers_pred,  # <-- added
    obs_radius, buffer_dist, min_safe_dist,
    min_h_values,
    save_path=None, step=None
):
    # --- Plotting ---
    plt.close("all")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(28, 20))

    # 1. Plot the (x, y) trajectory
    ax1 = plt.subplot(2, 1, 1)
    # Plot Optimized Trajectory
    ax1.plot(opt_states_0[0, :], opt_states_0[1, :], 'b-', marker='.', markersize=4, linewidth=1.5, label='Optimized Trajectory')
    # Plot Reference Trajectory
    ax1.plot(ref_waypoints[:, 0], ref_waypoints[:, 1], 'g-o', alpha=0.7, linewidth=1.5, label='Reference Trajectory')
    ax1.scatter(start[0], start[1], c='lime', marker='o', s=150, label='Start', zorder=10, edgecolors='black')
    ax1.scatter(goal[0], goal[1], c='red', marker='X', s=150, label='Goal', zorder=10, edgecolors='black')

    # Draw obstacles and safety boundaries (current positions)
    for i in range(num_obstacles):
        obs_center_plot = obs_state[i][:2]  # Only x, y
        obstacle_patch = Circle(obs_center_plot, obs_radius, color='black', alpha=0.6, zorder=5)
        ax1.add_patch(obstacle_patch)
        safety_circle = Circle(obs_center_plot, min_safe_dist, color='darkorange', fill=False, linestyle=':', linewidth=1.0, zorder=4)
        ax1.add_patch(safety_circle)

    # Plot predicted obstacle trajectories as dotted lines, safety circles, and black dots
    for i in range(num_obstacles):
        pred_traj = np.array([obs_centers_pred[i, j] for j in range(pred_horizn)])
        ax1.plot(pred_traj[:, 0], pred_traj[:, 1], 'k--', linewidth=1.5, alpha=0.7, label='Predicted Obstacle' if i == 0 else None)
        for j in range(1, pred_horizn):
            # Faded predicted obstacle body
            pred_patch = Circle(pred_traj[j], obs_radius, color='gray', alpha=0.3, zorder=3)
            ax1.add_patch(pred_patch)
            # Safety circle for predicted state
            pred_safety = Circle(pred_traj[j], min_safe_dist, color='darkorange', fill=False, linestyle=':', linewidth=1.0, zorder=2)
            ax1.add_patch(pred_safety)
            # Black dot at predicted center
            ax1.plot(pred_traj[j, 0], pred_traj[j, 1], 'ko', markersize=6, zorder=7)
        # Also plot for the first predicted state (j=0)
        ax1.plot(pred_traj[0, 0], pred_traj[0, 1], 'ko', markersize=6, zorder=7)
        pred_safety0 = Circle(pred_traj[0], min_safe_dist, color='darkorange', fill=False, linestyle=':', linewidth=1.0, zorder=2)
        ax1.add_patch(pred_safety0)

    # Direction Arrows (Optional)
    # ... (keep arrow plotting code if desired) ...
    arrow_skip = max(1, pred_horizn // 20) # Show fewer arrows
    head_width = 0.025
    head_length = 0.04
    arrow_color = 'darkcyan'
    arrow_length = 0.05 # Fixed length

    for i in range(0, pred_horizn, arrow_skip):
        x_pos = opt_states_0[0, i]
        y_pos = opt_states_0[1, i]
        theta = opt_states_0[2, i]
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        plt.arrow(x_pos, y_pos, dx, dy,
                    head_width=head_width, head_length=head_length,
                    fc=arrow_color, ec=arrow_color,
                    alpha=0.9, zorder=6)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('NMPC Safe Trajectory Planning')
    plt.legend(fontsize='small', loc='upper left')
    #plt.grid(True)
    plt.axis('equal') # Crucial for seeing circles correctly
    plt.xlim(-0.5, 1.5) # Adjust limits if needed
    plt.ylim(-0.5, 1.5)

    # 2. Plot control inputs
    ax2 = plt.subplot(2, 1, 2)
    time_steps = np.arange(ctrl_horizn) * sampling_time
    ax2.plot(time_steps, opt_control_0[0, :], 'b-o', alpha=0.7, linewidth=1.5, label='Optimized linear velocity control')
    ax2.plot(time_steps, opt_control_0[1, :], 'r-o', alpha=0.7, linewidth=1.5, label='Optimized angular velocity control')
    ax2.axhline(v_max, color='royalblue', linestyle=':', alpha=0.5, label='$v_{max}$')
    ax2.axhline(-v_max, color='royalblue', linestyle=':', alpha=0.5)
    ax2.axhline(omega_max, color='firebrick', linestyle=':', alpha=0.5, label='$\\omega_{max}$')
    ax2.axhline(-omega_max, color='firebrick', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Inputs')
    ax2.set_title('Optimal Control Inputs')
    ax2.legend(fontsize='small')
    ax2.set_ylim([-max(v_max, omega_max)*1.1, max(v_max, omega_max)*1.1])
    ax2.set_xlim(0, ctrl_horizn*sampling_time)


    # # # # --- Plot 3: Minimum CBF Value (h_min) ---
    # # plt.figure(figsize=(10, 5))
    # ax3 = plt.subplot(3, 1, 3)
    # time_steps_h = np.arange(pred_horizn) * sampling_time # Time steps for X_opt states (0 to N-1)
    # ax3.plot(time_steps_h, min_h_values, 'm-', linewidth=1.5, label='$h_{min}(t) = min_i(||x(t)-c_i||^2 - d_{min}^2)$')
    # # Add the safety boundary line h=0
    # ax3.axhline(0, color='r', linestyle='--', linewidth=1.5, label='Safety Boundary ($h=0$)')
    # ax3.set_xlabel('Time (s)')
    # ax3.set_ylabel('Minimum CBF Value (h_min)')
    # ax3.set_title('Control Barrier Function Satisfaction Check')
    # ax3.legend(fontsize='small')
    # ax3.grid(True)
    # # Adjust y-limits to see if it dips below zero
    # min_y = min(np.min(min_h_values), -0.01) # Ensure 0 is visible even if h_min is positive
    # max_y = max(np.max(min_h_values) * 1.1, 0.1) # Add some space above
    # ax3.set_ylim([min_y, max_y])
    # ax3.set_xlim(0, pred_horizn*sampling_time)
    # plt.tight_layout(pad=2.0)
    # Save the plot if requested
    if save_path is not None and step is not None:
        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"nmpc_step_{step:04d}.png"), bbox_inches='tight')

    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


    # # --- Calculate Tracking Error (Optional) ---
    # # Positional error
    # pos_error = X_opt[0:2, :] - X_ref_traj[0:2, :]
    # norm_pos_error = np.linalg.norm(pos_error, axis=0)
    # # Orientation error (wrapped)
    # theta_error_opt = X_opt[2, :] - X_ref_traj[2, :]
    # theta_error_opt_wrapped = np.arctan2(np.sin(theta_error_opt), np.cos(theta_error_opt))

    # print(f"\nMean Absolute Position Error: {np.mean(norm_pos_error):.4f} m")
    # print(f"Max Absolute Position Error:  {np.max(norm_pos_error):.4f} m")
    # print(f"Mean Absolute Orientation Error: {np.mean(np.abs(theta_error_opt_wrapped)):.4f} rad")
    # print(f"Max Absolute Orientation Error:  {np.max(np.abs(theta_error_opt_wrapped)):.4f} rad")


    # except Exception as e:
    # print(f"\n--- ERROR ---")
    # print(f"An error occurred during optimization or post-processing: {e}")
    # import traceback
    # traceback.print_exc()