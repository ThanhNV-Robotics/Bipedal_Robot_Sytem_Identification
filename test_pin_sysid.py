# Author:
# Van Thanh Nguyen

import pinocchio as pin
import numpy as np
from SysIDUtils import helpers as sysid_helpers
from SysIDUtils import mjc_regressors as mjc_reg
from SysIDUtils import mjc_parameters as mjc_params
from sys import argv
from pathlib import Path
from matplotlib import pyplot as plt
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


# Generate sine wave trajectory for each joint
duration = 4 # sec, duration of the trajectory
time_step = 0.02 # sec, time step
n_samples = int(duration / time_step) + 1
time = np.linspace(0, duration, n_samples) # time array vector
traj_freq = np.array([0.5, 0.7, 0.4, 0.6, 0.8, 0.5])  # frequency for each joint
traj_amp = np.pi/6*np.ones(model.nq)  # amplitude for each joint


standard_param_values = sysid_helpers.get_standard_parameters_values(model)
torque_standard_regressor = np.zeros((n_samples, model.nv))
torque_base_regressor = np.zeros((n_samples, model.nv))

torque_pinocchio = np.zeros((n_samples, model.nv))
torque_mujoco = np.zeros((n_samples, model.nv))

# reset mujoco data
mujoco.mj_resetData(mj_model, mj_data)

for i in range(len(time)):
    q_traj = traj_amp * np.sin(2 * np.pi * traj_freq * time[i])
    qd_traj = 2 * np.pi * traj_freq * traj_amp * np.cos(2 * np.pi * traj_freq * time[i])
    qdd_traj = - (2 * np.pi * traj_freq)**2 * traj_amp * np.sin(2 * np.pi * traj_freq * time[i])

    # compute torque using standard regressor
    W_t = sysid_helpers.calculate_standard_regressor(model, data, q_traj.reshape(1,-1), # reshape to column vector
                                                    qd_traj.reshape(1,-1), 
                                                    qdd_traj.reshape(1,-1))
    torque_standard_regressor[i,:] = W_t@standard_param_values

    W_base, X_base,_,_ = sysid_helpers.compute_base_model(W_t,standard_param_values,TOL_QR=1e-6)
    torque_base_regressor[i,:] = W_base@X_base

    # compute torque using pinocchio RNEA
    torque_pinocchio[i, :] = pin.rnea(model, data, q_traj, qd_traj, qdd_traj)

    # torque compute using mj_inverse
    # set mujoco q, dq, ddq
    mj_data.qpos[:] = q_traj
    mj_data.qvel[:] = qd_traj
    mj_data.qacc[:] = qdd_traj
    # call mj_inverse to compute torques
    mujoco.mj_inverse(mj_model, mj_data)
    # get the computed torques
    torque_mujoco[i, :] = mj_data.qfrc_inverse[:]

# plot the results
plt.figure()
for j in range(model.nv):
    plt.plot(model.nv,1,j+1)
    plt.plot(time, torque_standard_regressor[:,j], label='Standard Regressor Torque', linestyle='--')
    plt.plot(time, torque_pinocchio[:,j], label='Pinocchio RNEA Torque', linestyle=':')
    plt.plot(time, torque_base_regressor[:,j], label='Base Regressor Torque', linestyle='-')
    plt.title(f'Joint {j+1} Torque Comparison')
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()





