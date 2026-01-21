"""
Script to read and plot CSV data from experimental runs
"""

import numpy as np
from matplotlib import pyplot as plt

# Load CSV data
DATA_FILE_PATH = "data/exp/left_leg_trajectory_run_03_ok.csv"
saved_data = np.loadtxt(DATA_FILE_PATH, delimiter=',', skiprows=1)
print("\nLoaded CSV data")
print("Data shape:", saved_data.shape)

time_data = saved_data[:, 0]

# Extract left leg joint data (positions, velocities, efforts)
# Based on CSV structure: time, stamp, then left leg joints
# Pattern: effort, position, velocity for each joint
left_joints = ['left_ankle_pitch', 'left_ankle_roll', 'left_hip_pitch', 
               'left_hip_roll', 'left_hip_yaw', 'left_knee']

torque_data = np.zeros((len(time_data), 6))
q_feb_data = np.zeros((len(time_data), 6))
dq_feb_data = np.zeros((len(time_data), 6))

for i in range(6):
    base_col = 2 + i * 3  # Each joint has 3 columns: effort, position, velocity
    torque_data[:, i] = saved_data[:, base_col]      # effort
    q_feb_data[:, i] = saved_data[:, base_col + 1]   # position
    dq_feb_data[:, i] = saved_data[:, base_col + 2]  # velocity

print(f"Time range: {time_data[0]:.2f} to {time_data[-1]:.2f} seconds")
print(f"Number of samples: {len(time_data)}")
print(f"Sampling rate: {1/(time_data[1] - time_data[0]):.2f} Hz")

# Plot the data
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

joint_names = ['Ankle Pitch', 'Ankle Roll', 'Hip Pitch', 'Hip Roll', 'Hip Yaw', 'Knee']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Plot positions
for i in range(6):
    axes[0].plot(time_data, q_feb_data[:, i], label=joint_names[i], color=colors[i], linewidth=1.5)
axes[0].set_ylabel('Position (rad)', fontsize=11)
axes[0].set_title('Joint Positions - Left Leg', fontsize=12, fontweight='bold')
axes[0].legend(loc='best', ncol=2)
axes[0].grid(True, alpha=0.3)

# Plot velocities
for i in range(6):
    axes[1].plot(time_data, dq_feb_data[:, i], label=joint_names[i], color=colors[i], linewidth=1.5)
axes[1].set_ylabel('Velocity (rad/s)', fontsize=11)
axes[1].set_title('Joint Velocities - Left Leg', fontsize=12, fontweight='bold')
axes[1].legend(loc='best', ncol=2)
axes[1].grid(True, alpha=0.3)

# Plot torques
for i in range(6):
    axes[2].plot(time_data, torque_data[:, i], label=joint_names[i], color=colors[i], linewidth=1.5)
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Torque (Nm)', fontsize=11)
axes[2].set_title('Joint Torques - Left Leg', fontsize=12, fontweight='bold')
axes[2].legend(loc='best', ncol=2)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('left_leg_data_plot.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'left_leg_data_plot.png'")
plt.show()
