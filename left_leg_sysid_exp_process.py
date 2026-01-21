# Author:
# Van Thanh Nguyen

import numpy as np
from SysIDUtils import helpers as sysid_helpers
from sys import argv
from pathlib import Path
from matplotlib import pyplot as plt
from scipy import signal
import os
import mujoco


# Path to mjcf .xml model
LEFT_LEG_MJCF = "robot_models/mjcf/left_leg.xml"
mj_model = mujoco.MjModel.from_xml_path(LEFT_LEG_MJCF)
mj_data = mujoco.MjData(mj_model)


# load saved data from CSV
DATA_FILE_PATH = "data/exp/left_leg_trajectory_run_03_ok.csv"
saved_data = np.loadtxt(DATA_FILE_PATH, delimiter=',', skiprows=1)
print("\nsaved data shape:", saved_data.shape)
time_data = saved_data[:, 0] - saved_data[0, 0]  # Convert to seconds starting from 0

# Extract left leg joint data (positions, velocities, efforts)
# Based on CSV structure: time, stamp, then left leg joints
left_ankle_pitch_idx = 2  # effort starts at column 2
left_joints = ['left_ankle_pitch', 'left_ankle_roll', 'left_hip_pitch', 
               'left_hip_roll', 'left_hip_yaw', 'left_knee']

# Extract effort (torque), position, velocity for each joint
# Pattern: effort, position, velocity for each joint
torque_data = np.zeros((len(time_data), 6))
q_feb_data = np.zeros((len(time_data), 6))
dq_feb_data = np.zeros((len(time_data), 6))
# differentiate dq_feb_data to get ddq_feb_data (measured acceleration)
ddq_feb_data = np.gradient(dq_feb_data, axis=0) / np.gradient(time_data, axis=0)[:, None]
for i in range(6):
    base_col = 2 + i * 3  # Each joint has 3 columns: effort, position, velocity
    torque_data[:, i] = saved_data[:, base_col]      # effort
    q_feb_data[:, i] = saved_data[:, base_col + 1]   # position
    dq_feb_data[:, i] = saved_data[:, base_col + 2]  # velocity

# Apply filter AFTER extracting the data
b, a = signal.butter(3, 0.1)  # 3rd order Butterworth filter with cutoff frequency of 0.1*Nyquist
dq_feb_data_filted = signal.filtfilt(b, a, dq_feb_data, axis=0)

# Plot the data
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

joint_names = ['Ankle Pitch', 'Ankle Roll', 'Hip Pitch', 'Hip Roll', 'Hip Yaw', 'Knee']

# Plot positions
for i in range(6):
    axes[0].plot(time_data, q_feb_data[:, i], label=joint_names[i])
axes[0].set_ylabel('Position (rad)')
axes[0].set_title('Joint Positions')
axes[0].legend()
axes[0].grid(True)

# Plot velocities
for i in range(6):
    axes[1].plot(time_data, dq_feb_data[:, i], label=joint_names[i])
axes[1].set_ylabel('Velocity (rad/s)')
axes[1].set_title('Joint Velocities')
axes[1].legend()
axes[1].grid(True)

# Plot torques
for i in range(6):
    axes[2].plot(time_data, torque_data[:, i], label=joint_names[i])
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Torque (Nm)')
axes[2].set_title('Joint Torques')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()

# compare filtered and unfiltered velocity
plt.figure()
for i in range(6):
    plt.plot(time_data, dq_feb_data[:, i], label=f'Joint {i+1} Unfiltered', linestyle='--')
    plt.plot(time_data, dq_feb_data_filted[:, i], label=f'Joint {i+1} Filtered', linestyle='-')
    plt.title(f'Joint {i+1} Velocity Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()


