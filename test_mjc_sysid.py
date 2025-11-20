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
pinmodel = pin.buildModelFromUrdf(urdf_filename)
print("model name: " + pinmodel.name)
# create data
pindata = pinmodel.createData()

# load mujoco model
mj_model = mujoco.MjModel.from_xml_path(str(urdf_filename))
mj_data = mujoco.MjData(mj_model)

SAMPLES = 10

theta = np.concatenate([mjc_params.get_dynamic_parameters(mj_model, i) for i in mj_model.jnt_bodyid])

for _ in range(SAMPLES):
    q, v, dv = np.random.rand(pinmodel.nq), np.random.rand(pinmodel.nv), np.random.rand(pinmodel.nv)
    pin.rnea(pinmodel, pindata, q, v, dv)

    mj_data.qpos[:] = q
    mj_data.qvel[:] = v
    mj_data.qacc[:] = dv
    mujoco.mj_inverse(mj_model, mj_data)
    mujoco.mj_rnePostConstraint(mj_model, mj_data)

    pinY = pin.computeJointTorqueRegressor(pinmodel, pindata, q, v, dv)
    mjY = mjc_reg.joint_torque_regressor(mj_model, mj_data)

    #tau = pin.rnea(pinmodel, pindata, q, v, dv)
    tau = mj_data.qfrc_inverse[:]

    # assert np.allclose(mjY @ theta, tau, atol=1e-6), f"Norm diff: {np.linalg.norm(mjY @ theta - tau)}"
    # assert np.allclose(mjY, pinY, atol=1e-6), f"Norm diff: {np.linalg.norm(mjY - pinY)}"
    print(f"Norm diff tau: {np.linalg.norm(pinY @ theta - tau)}")
    print(f"Norm diff Y: {np.linalg.norm(mjY - pinY)}")


# # Check some properties of the mujoco model
# print("\n=== MuJoCo Model Properties ===")
# print("number of joints (njnt): " + str(mj_model.njnt))
# print("number of generalized coordinates (nq): " + str(mj_model.nq))
# print("number of degrees of freedom (nv): " + str(mj_model.nv))
# print("number of bodies (nbody): " + str(mj_model.nbody))
# print("number of actuators (nu): " + str(mj_model.nu))
# print("number of sensors (nsensor): " + str(mj_model.nsensor))
# print("number of geoms (ngeom): " + str(mj_model.ngeom))
# print("timestep: " + str(mj_model.opt.timestep))

# # Print body names
# print("\nMuJoCo Body names:")
# for i, body_id in enumerate(mj_model.jnt_bodyid):
#     body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
#     print(f"  Joint {i} is attached to Body {body_id}: {body_name}")

# # Print joint names
# print("\nMuJoCo Joint names:")
# for i in range(mj_model.njnt):
#     joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
#     joint_type = mj_model.jnt_type[i]
#     print(f"  Joint {i}: {joint_name} (type: {joint_type})")

# # get a list containing the full set of standard parameters
# standard_params_dict = mjc_params.get_full_dynamics_parameters_dictionary(mj_model)
# print("\n=== Full set of standard dynamic parameters ===")
# for param_name, param_value in standard_params_dict.items():
#     print(f"  {param_name}: {param_value}")

# #standard_param_values = mjc_params.get_full_dynamics_parameters_value(mj_model)
# standard_param_values = np.concatenate([mjc_params.get_dynamic_parameters(mj_model, i) for i in mj_model.jnt_bodyid])
# print("\n total number of standard dynamic parameters: " + str(len(standard_param_values)))

# n_samples = 1 # sample size to generate random configurations
# # get vectors of joint limits, velocity limits
# joint_lower_limits = []
# joint_upper_limits = []
# joint_velocity_limits = []
# joint_lower_limits = mj_model.jnt_range[:, 0].copy()
# joint_upper_limits = mj_model.jnt_range[:, 1].copy()
# joint_velocity_limits = mj_model.jnt_range[:, 1].copy()
# joint_lower_limits = np.array(joint_lower_limits)
# joint_upper_limits = np.array(joint_upper_limits)
# joint_velocity_limits = np.array(joint_velocity_limits)
# # default limit qdd to 6*pi rad/s2
# joint_acceleration_limits = 6 * np.pi * np.ones(mj_model.nv)

# # generate 30 samples of q_rand, dq_rand, ddq_rand
# q_rand = np.random.uniform(low=joint_lower_limits, high=joint_upper_limits, size=(n_samples, mj_model.nq))
# dq_rand = np.random.uniform(low=-joint_velocity_limits, high=joint_velocity_limits, size=(n_samples, mj_model.nv))
# ddq_rand = np.random.uniform(low=-joint_acceleration_limits, high=joint_acceleration_limits, size=(n_samples, mj_model.nv))

# # construct standard regressor matrix
# W_standard = mjc_reg.calculate_regressor(mj_model, mj_data, q_rand, dq_rand, ddq_rand)
# tau_standard = W_standard @ standard_param_values

# for i in range(n_samples):
#     mujoco.mj_resetData(mj_model, mj_data)
#     mj_data.qpos[:] = q_rand[i, :]
#     mj_data.qvel[:] = dq_rand[i, :]
#     mj_data.qacc[:] = ddq_rand[i, :]
#     mujoco.mj_inverse(mj_model, mj_data)
#     tau_mujoco = mj_data.qfrc_inverse[:]

# print("residual between torque from mujoco.mj_inverse and standard regressor model: ", tau_mujoco - tau_standard)

# print("\nStandard regressor matrix W_standard shape: ", W_standard.shape)

# print("\nNumerical Rank of the standard regressor matrix W_standard: ")
# print(sysid_helpers.compute_numerical_rank(W_standard))

# # # check numerical rank of W_reduced
# # X_base, beta, indep_idx, depe_idx = sysid_helpers.calculate_base_parameters_symbols(W_standard,
# #                                                                                     standard_param_symbols_list,
# #                                                                                     TOL_QR=1e-6)
# # W_base, beta, indep_idx, depe_idx = sysid_helpers.compute_base_regressor(W_standard, TOL_QR=1e-6)
# # print("\nBase regressor matrix W_base shape: ", W_base.shape)


# # Generate sine wave trajectory for each joint
# duration = 4 # sec, duration of the trajectory
# time_step = 0.02 # sec, time step
# n_samples = int(duration / time_step) + 1
# time = np.linspace(0, duration, n_samples) # time array vector
# traj_freq = np.array([0.5, 0.7, 0.4, 0.6, 0.8, 0.5])  # frequency for each joint
# traj_amp = np.pi/6*np.ones(mj_model.nq)  # amplitude for each joint

# q_traj = np.zeros((n_samples, mj_model.nq))
# qd_traj = np.zeros((n_samples, mj_model.nv))
# qdd_traj = np.zeros((n_samples, mj_model.nv))

# torque_mujoco = np.zeros((n_samples, mj_model.nv))
# torque_standard_reg = np.zeros((n_samples, mj_model.nv))

# # reset the state
# mujoco.mj_resetData(mj_model, mj_data)

# for i in range(len(time)):
#     q_traj = traj_amp * np.sin(2 * np.pi * traj_freq * time[i])
#     qd_traj = 2 * np.pi * traj_freq * traj_amp * np.cos(2 * np.pi * traj_freq * time[i])
#     qdd_traj = - (2 * np.pi * traj_freq)**2 * traj_amp * np.sin(2 * np.pi * traj_freq * time[i])
    
#     mj_data.qpos[:] = q_traj
#     mj_data.qvel[:] = qd_traj
#     mj_data.qacc[:] = qdd_traj
#     mujoco.mj_inverse(mj_model, mj_data)
#     torque_mujoco[i, :] = mj_data.qfrc_inverse[:]
#     mjY = mjc_reg.joint_torque_regressor(mj_model, mj_data)
#     torque_standard_reg[i, :] = mjY @ standard_param_values

# # # plot joint trajectories
# # plt.figure()
# # for i in range(mj_model.nq):
# #     plt.plot(time, q_traj[:, i], label='q'+str(i+1))
# # plt.xlabel('Time [s]')
# # plt.ylabel('Joint ' + str(i+1))
# # plt.legend()

# #plot joint torques
# plt.figure()
# for i in range(mj_model.nv):
#     plt.plot(time, torque_mujoco[:, i], label='torque_mujoco'+str(i+1))
#     # plt.plot(time, torque_pin[:, i], '--', label='torque_pin'+str(i+1))
#     plt.plot(time, torque_standard_reg[:, i], ':', label='tau_standard_reg'+str(i+1))
# plt.xlabel('Time [s]')
# plt.ylabel('Joint Torque')
# plt.legend()
# plt.show()




