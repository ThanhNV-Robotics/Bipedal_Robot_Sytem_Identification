# Author:
# Van Thanh Nguyen

import pinocchio as pin
import numpy as np
from SysIDUtils import helpers as sysid_helpers
from sys import argv
from pathlib import Path
from matplotlib import pyplot as plt
from scipy import signal
import os

# PATH_TO_URDF_FILE = 'robot_models/left_leg/left_leg.urdf'
package_dirs = Path("robot_models/left_leg")
urdf_filename = (
    package_dirs / "left_leg.urdf"
    if len(argv) < 2
    else argv[1]
)

# 1. create model
model = pin.buildModelFromUrdf(urdf_filename)
print("model name: " + model.name)
# create data
data = model.createData()

n_samples = 30

# get vectors of joint limits, velocity limits
joint_lower_limits = []
joint_upper_limits = []
joint_velocity_limits = []
joint_lower_limits = model.lowerPositionLimit.copy()
joint_upper_limits = model.upperPositionLimit.copy()
joint_velocity_limits = model.velocityLimit.copy()
joint_lower_limits = np.array(joint_lower_limits)
joint_upper_limits = np.array(joint_upper_limits)
joint_velocity_limits = np.array(joint_velocity_limits)
# default limit qdd to 6*pi rad/s2
joint_acceleration_limits = 6 * np.pi * np.ones(model.nv)
# generate 30 samples of q_rand, dq_rand, ddq_rand
q_rand = np.array([pin.randomConfiguration(model) for _ in range(n_samples)])
dq_rand = np.random.uniform(low=-joint_velocity_limits, high=joint_velocity_limits, size=(n_samples, model.nv))
ddq_rand = np.random.uniform(low=-joint_acceleration_limits, high=joint_acceleration_limits, size=(n_samples, model.nv))

# construct standard regressor matrix
W_standard = sysid_helpers.calculate_standard_regressor(model, data, q_rand, dq_rand, ddq_rand)
print("\nStandard regressor matrix W_standard shape: ", W_standard.shape)

# # get standard parameter dictionary
# standard_param_dict = sysid_helpers.get_standard_parameters_dict(model)

# # Convert dictionary to 2D array for saving
# param_data = np.array([[key, value] for key, value in standard_param_dict.items()])

# # save the standard parameter dictionary to csv
# param_save_path = os.path.join("data", "left_leg_standard_param_dict.csv")
# np.savetxt(param_save_path, param_data, delimiter=',', header='Parameter,Value', comments='', fmt='%s')
# print(f"\nStandard parameter dictionary saved to '{param_save_path}'")


standard_param_values = sysid_helpers.get_standard_parameters_values(model)
_,base_param_values, P1, P2 = sysid_helpers.compute_base_model(W_standard,standard_param_values)

# load saved data
data_file_path = os.path.join("data", "left_leg_simulation_data.npy")
simulation_data = np.load(data_file_path)
print("\nsaved data shape:", simulation_data.shape)
time_data = simulation_data[:,0]
q_ref_data = simulation_data[:,1:model.nq+1]
q_feb_data = simulation_data[:,model.nq+1:2*model.nq+1]
dq_feb_data = simulation_data[:,2*model.nq+1:3*model.nq+1]
torque_data = simulation_data[:,3*model.nq+1:4*model.nq+1]
ref_acceleration_data = simulation_data[:,4*model.nq+1:]
    
time_step = time_data[1] - time_data[0]
init_duration = 2.0  # seconds
init_steps = int(init_duration / time_step)

# remove the initial 2 seconds data since first 2 seconds is used to ramp up the trajectory
time_data = time_data[init_steps:]
q_ref_data =  q_ref_data[init_steps:,:]
q_feb_data =  q_feb_data[init_steps:,:]
dq_feb_data =  dq_feb_data[init_steps:,:]
torque_data =  torque_data[init_steps:,:]
#torque_data = torque_data.reshape(-1, 1)

# differentiate dq_feb_data to get ddq_feb_data (measured acceleration)
ddq_feb_data = np.gradient(dq_feb_data, axis=0) / np.gradient(time_data, axis=0)[:, None]

b, a = signal.butter(3, 0.03)  # 3rd order Butterworth filter with cutoff frequency of 0.04*Nyquist
ddq_feb_data_filted = signal.filtfilt(b, a, ddq_feb_data, axis=0)

# plot acceleration data to check
# plt.figure()
# for i in range(model.nv):
#     plt.subplot(3,2,i+1)
#     plt.plot(time_data, ddq_feb_data[:,i], label='Joint Acceleration')
#     plt.plot(time_data, ddq_feb_data_filted[:,i], label='Filtered Joint Acceleration')
#     plt.title(f'Joint {i+1} Acceleration')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Acceleration (rad/s²)')
#     plt.legend()
#     plt.grid()
# plt.tight_layout()
# plt.show()


# # plot the saved data to check
# plt.figure()
# for i in range(model.nq):
#     plt.subplot(3,2,i+1)
#     plt.plot(time_data, q_ref_data[:,i], label='Reference Angle')
#     plt.plot(time_data, q_feb_data[:,i], label='Joint Position')
#     plt.title(f'Joint {i+1} Position Tracking')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Angle (rad)')
#     plt.legend()
#     plt.grid()

# # plot the torque data
# plt.figure()
# for i in range(model.nv):
#     plt.subplot(3,2,i+1)
#     plt.plot(time_data, torque_data[:,i], label='Control Torque')
#     plt.title(f'Joint {i+1} Control Torque')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Torque (N.m)')
#     plt.legend()
#     plt.grid()
# plt.tight_layout()
# plt.show()

# construct standard regressor matrix for the loaded trajectory data
W_standard_traj = sysid_helpers.calculate_standard_regressor(model, data, q_feb_data, dq_feb_data, ref_acceleration_data)
print("\nStandard regressor matrix W_standard_traj shape: ", W_standard_traj.shape)
zero_cols_idx = sysid_helpers.get_unidentificable_parameter_index (W_standard_traj, tol = 1e-6)
# Process to archive base regressor matrix
# remove zero cols and build a zero columns free regressor matrix
W_standard_traj_reduced = np.delete(W_standard_traj, zero_cols_idx, axis=1) # remove the 0 columns

# remove the dependent columns to get the base regressor matrix
W_base_traj = W_standard_traj_reduced@P1

# check W_base and base_param_values
torque_reconstructed = W_standard_traj @ standard_param_values
# torque_reconstructed = W_base_traj @ base_param_values
torque_reconstructed = torque_reconstructed.reshape(torque_data.shape)
# plot comparison between measured torque and reconstructed torque
plt.figure()
for i in range(model.nv):
    plt.subplot(3,2,i+1)
    plt.plot(time_data, torque_data[:,i], label='Measured Torque')
    plt.plot(time_data, torque_reconstructed[:,i], label='Reconstructed Torque')
    plt.title(f'Joint {i+1} Torque Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N.m)')
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()
# # check rank of W_base_traj
# print("\nRank of the base regressor matrix W_base_traj: ")
# print(np.linalg.matrix_rank(W_base_traj))

# solution to the least square problem to estimate base parameters

#base_param_estimate, residuals, rank, s = np.linalg.lstsq(W_base_traj, torque_data, rcond=None)

# Q, R = np.linalg.qr(W_base_traj)
# base_param_estimate = np.linalg.solve(R, Q.T @ torque_data)
# # stack base_param_estimate and base_param_values side by side and save to csv
# base_param_comparison = np.hstack((np.array(base_param_values).reshape(-1,1), np.array(base_param_estimate).reshape(-1,1)))
# param_save_path = os.path.join("data", "left_leg_base_parameters_estimation.csv")
# # save to csv
# np.savetxt(param_save_path, base_param_comparison, delimiter=',', header='True Base Parameters, Estimated Base Parameters', comments='', fmt='%.6f')
# print(f"\nBase parameters comparison saved to '{param_save_path}'")


