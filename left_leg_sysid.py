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
import mujoco

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

# load mujoco model
mj_model = mujoco.MjModel.from_xml_path(str(urdf_filename))
mj_data = mujoco.MjData(mj_model)

X_base_syms = sysid_helpers.calculate_base_parameters_symbols(model,data, TOL_QR=1e-6)

X_standard_values = sysid_helpers.get_standard_parameters_values(model)
# X_base is the list of base parameter symbols
# P1, and P2 are two Permutation matrices
# P1 corresponds to the independent columns of the regressor
# P2 corresponds to the dependent columns of the regressor
# Y_base = Y_reduced @ P1 -> where Y_reduced is the regressor matrix after removing zero columns
X_base_values, P1, P2 = sysid_helpers.compute_base_parameter_values(model, data, TOL_QR=1e-6)

# # get standard parameter dictionary
# standard_param_dict = sysid_helpers.get_standard_parameters_dict(model)

# # Convert dictionary to 2D array for saving
# param_data = np.array([[key, value] for key, value in standard_param_dict.items()])

# # save the standard parameter dictionary to csv
# param_save_path = os.path.join("data", "left_leg_standard_param_dict.csv")
# np.savetxt(param_save_path, param_data, delimiter=',', header='Parameter,Value', comments='', fmt='%s')
# print(f"\nStandard parameter dictionary saved to '{param_save_path}'")


# load saved data
data_file_path = os.path.join("data", "left_leg_simulation_data.npy")
saved_data = np.load(data_file_path)
print("\nsaved data shape:", saved_data.shape)
time_data = saved_data[:,0]
q_ref_data = saved_data[:,1:7]
q_feb_data = saved_data[:,7:13]
dq_feb_data = saved_data[:,13:19]
torque_data = saved_data[:,19:25]
ddq_feb_data = saved_data[:,25:31]
    
time_step = time_data[1] - time_data[0]
init_duration = 2.0  # seconds
init_steps = int(init_duration / time_step)

# remove the initial 2 seconds data since first 2 seconds is used to ramp up the trajectory
time_data = time_data[init_steps:]
q_ref_data =  q_ref_data[init_steps:,:]
q_feb_data =  q_feb_data[init_steps:,:]
dq_feb_data =  dq_feb_data[init_steps:,:]
torque_data =  torque_data[init_steps:,:]
ddq_feb_data =  ddq_feb_data[init_steps:,:]

# differentiate dq_feb_data to get ddq_feb_data (measured acceleration)
ddq_feb_data = np.gradient(dq_feb_data, axis=0) / np.gradient(time_data, axis=0)[:, None]

b, a = signal.butter(3, 0.03)  # 3rd order Butterworth filter with cutoff frequency of 0.04*Nyquist
ddq_feb_data_filted = signal.filtfilt(b, a, ddq_feb_data, axis=0)

# #plot acceleration data to check
# plt.figure()
# for i in range(model.nv):
#     plt.subplot(3,2,i+1)
#     plt.plot(time_data, ddq_feb_data[:,i], label='Joint Acceleration')

#     plt.title(f'Joint {i+1} Acceleration')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Acceleration (rad/s²)')
#     plt.legend()
#     plt.grid()
# plt.tight_layout()



# # # plot the saved data to check
# plt.figure()
# for i in range(model.nq):
#     plt.subplot(3,2,i+1)
#     plt.plot(time_data, q_feb_data[:,i], label='Joint position')
#     plt.title(f'Joint {i+1} Position Tracking')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Position (rad)')
#     plt.legend()
#     plt.grid()

# plt.show()
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
torque_pinocchio = np.zeros((len(time_data), model.nv))
mjc_torque_inverse = np.zeros((len(time_data), model.nv))
mujoco.mj_resetData(mj_model, mj_data)
# check the inverse dynamics
for i in range(len(time_data)):
    torque_pinocchio[i, :] = pin.rnea(model, data, q_feb_data[i, :], dq_feb_data[i, :], ddq_feb_data[i, :])
    # set mj_data qpos, qvel, qacc
    mj_data.qpos[:] = q_feb_data[i, :]
    mj_data.qvel[:] = dq_feb_data[i, :]
    mj_data.qacc[:] = ddq_feb_data[i, :]
    # call mj_inverse to compute torques
    mujoco.mj_inverse(mj_model, mj_data)
    mjc_torque_inverse[i, :] = mj_data.qfrc_inverse[:]

# construct standard regressor matrix for the loaded trajectory data
W_standard_traj = sysid_helpers.calculate_standard_regressor(model, data, q_feb_data, dq_feb_data, ddq_feb_data_filted)

# construct the base regressor matrix
zero_cols_idx = sysid_helpers.get_unidentificable_parameter_index (model, data, tol = 1e-6) # get zero columns index
W_standard_traj_reduced = np.delete(W_standard_traj, zero_cols_idx, axis=1) # remove the 0 columns

W_base_traj = W_standard_traj_reduced@P1 # Calculate the base regressor matrix, P1 is the permutation matrix from QR decomposition


# check W_base and base_param_values
torque_standard_reg = W_standard_traj @ X_standard_values
# torque_standard_reg = W_base_traj @ base_param_values
torque_standard_reg = torque_standard_reg.reshape(-1, model.nv)

torque_base_reg = W_base_traj @ X_base_values
torque_base_reg = torque_base_reg.reshape((len(time_data), model.nv)) #

# plt.figure()
# for i in range(model.nv):
#     plt.subplot(3,2,i+1)
#     plt.plot(time_data, torque_data[:,i], label='Measured Torque')
#     plt.plot(time_data, torque_base_reg[:,i], label='Torque base regressor model')
#     # plt.plot(time_data, mjc_torque_inverse[:,i], label='Mujoco Torque')
#     plt.title(f'Joint {i+1} Torque Comparison')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Torque (N.m)')
#     plt.legend()
#     plt.grid()
# plt.tight_layout()
# plt.show()

# check rank of W_base_traj
print("\nRank of the base regressor matrix W_base_traj: ")
print(np.linalg.matrix_rank(W_base_traj))

# solution to the least square problem to estimate base parameters
torque_data = torque_data.reshape(-1,1)  # reshape torque_data to 1D array
Q, R = np.linalg.qr(W_base_traj)
base_param_estimate = np.linalg.solve(R, Q.T @ torque_data)

# stack base_param_estimate and base_param_values side by side and save to csv
base_param_comparison = np.hstack((np.array(X_base_values).reshape(-1,1), np.array(base_param_estimate).reshape(-1,1)))
param_save_path = os.path.join("data", "left_leg_base_parameters_estimation.csv")
# save to csv
np.savetxt(param_save_path, base_param_comparison, delimiter=',', header='True Base Parameters, Estimated Base Parameters', comments='', fmt='%.6f')
print(f"\nBase parameters comparison saved to '{param_save_path}'")

# compare torque with estimated base parameters
torque_base_reg_estimated = W_base_traj @ base_param_estimate
torque_base_reg_estimated = torque_base_reg_estimated.reshape((len(time_data), model.nv)) #
torque_data = torque_data.reshape((len(time_data), model.nv))
plt.figure()
for i in range(model.nv):
    plt.subplot(3,2,i+1)
    plt.plot(time_data, torque_data[:,i], label='Measured Torque')
    plt.plot(time_data, torque_base_reg_estimated[:,i], label='Identified model')
    plt.title(f'Joint {i+1}')
    plt.ylabel('Torque (N.m)')
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()
