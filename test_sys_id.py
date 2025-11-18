# Author:
# Van Thanh Nguyen

import pinocchio as pin
import numpy as np
from SysIDUtils import helpers as sysid_helpers
from sys import argv
from pathlib import Path
from matplotlib import pyplot as plt

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

# infere some basic information of the model
print("number of joints (njoints): " + str(model.njoints))
print("number of dofs (nq): " + str(model.nq))
print("number of bodies: " + str(len(model.inertias)-1))  #exclude the universe body
print("number of actuated joints (nv): " + str(model.nv))

# Print all joint names to understand the structure
print("\nJoint names:")
for i, name in enumerate(model.names):
    print(f"  Joint {i}: {name}")

# generate a list containing the full set of standard parameters
standard_params = sysid_helpers.get_standard_parameters_dict(model) # return a dictionary of standard parameters including symbols and values
standard_param_symbols_list = sysid_helpers.get_list_standard_param_symbols(model) # return a list of standard parameter symbols
print("\n total of Standard parameters: " + str(len(standard_param_symbols_list)))
print("\nStandard parameter symbols list: ")
print(standard_param_symbols_list)

n_samples = 30 # sample size to generate random configurations

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

print("\nNumerical Rank of the standard regressor matrix W_standard: ")
print(sysid_helpers.compute_numerical_rank(W_standard))

# check numerical rank of W_reduced
X_base, beta, indep_idx, depe_idx = sysid_helpers.calculate_base_parameters_symbols(W_standard,
                                                                                    standard_param_symbols_list,
                                                                                    TOL_QR=1e-6)
W_base, beta, indep_idx, depe_idx = sysid_helpers.compute_base_regressor(W_standard, TOL_QR=1e-6)
print("\nBase regressor matrix W_base shape: ", W_base.shape)


# Generate sine wave trajectory for each joint
duration = 4 # sec, duration of the trajectory
time_step = 0.02 # sec, time step
n_samples = int(duration / time_step) + 1
time = np.linspace(0, duration, n_samples) # time array vector
traj_freq = np.array([0.5, 0.7, 0.4, 0.6, 0.8, 0.5])  # frequency for each joint
traj_amp = np.pi/6*np.ones(model.nq)  # amplitude for each joint

q_traj = np.zeros((n_samples, model.nq))
qd_traj = np.zeros((n_samples, model.nv))
qdd_traj = np.zeros((n_samples, model.nv))

for i in range(model.nq):
    q_traj[:, i] = traj_amp[i] * np.sin(2 * np.pi * traj_freq[i] * time)
    qd_traj[:, i] = 2 * np.pi * traj_freq[i] * traj_amp[i] * np.cos(2 * np.pi * traj_freq[i] * time)
    qdd_traj[:, i] = - (2 * np.pi * traj_freq[i])**2 * traj_amp[i] * np.sin(2 * np.pi * traj_freq[i] * time)

# Torque computed by pinocchio RNEA
torque_pin = np.zeros((n_samples, model.nv))
for i in range(n_samples):
    torque_pin[i, :] = pin.rnea(model, data, q_traj[i, :], qd_traj[i, :], qdd_traj[i, :])

# Torque computed by standard regressor model
standard_param_values = sysid_helpers.get_standard_parameters_values(model)
W_standard_traj = sysid_helpers.calculate_standard_regressor(model, data, q_traj, qd_traj, qdd_traj)
tau_standard_reg = W_standard_traj @ standard_param_values
tau_standard_reg = tau_standard_reg.reshape((n_samples, model.nv)) # reshape tau_standard_reg to (n_samples, nv)

# torque computed by reduced regressor model: remove 0 columns in W_standard_traj
unidentified_idx = sysid_helpers.get_unidentificable_parameter_index (W_standard_traj, tol = 1e-6)
W_standard_traj_reduced = np.delete(W_standard_traj, unidentified_idx, axis=1) # remove the 0 columns
X_reduced = np.delete(standard_param_values, unidentified_idx, axis=0) # remove the unidentified parameters
torque_reduced_reg = W_standard_traj_reduced @ X_reduced
torque_reduced_reg = torque_reduced_reg.reshape((n_samples, model.nv)) # reshape torque_reduced_reg to (n_samples, nv)


# Torque computed by base regressor model
W_base_traj, base_param_values = sysid_helpers.compute_base_model(W_standard_traj, standard_param_values, TOL_QR=1e-6)
torque_base_reg = W_base_traj @ base_param_values
torque_base_reg = torque_base_reg.reshape((n_samples, model.nv)) # reshape torque_base_reg to (n_samples, nv)

# plot joint trajectories
plt.figure()
for i in range(model.nq):
    plt.plot(time, q_traj[:, i], label='q'+str(i+1))
plt.xlabel('Time [s]')
plt.ylabel('Joint ' + str(i+1))
plt.legend()

#plot joint torques
plt.figure()
for i in range(model.nv):
    plt.plot(time, torque_pin[:, i], label='tau_pin'+str(i+1))
    plt.plot(time, tau_standard_reg[:, i], '--', label='tau_reduced_reg'+str(i+1))
    plt.plot(time, torque_base_reg[:, i], ':', label='tau_base_reg'+str(i+1))
plt.xlabel('Time [s]')
plt.ylabel('Joint Torque')
plt.legend()
plt.show()



