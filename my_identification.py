# Author:
# Van Thanh Nguyen

import pinocchio as pin
import numpy as np
from SysIDUtils import helpers as sysid_helpers
from sys import argv
from pathlib import Path

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

# 2. generate a list containing the full set of standard parameters
standard_params = sysid_helpers.get_standard_parameters(model)
standard_param_symbols_list = sysid_helpers.get_list_standard_param_symbols(model)
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
# Test with zeros - use 1D arrays (shape: (6,)) not 2D column vectors
q = np.zeros(model.nq, dtype=np.float64)
v = np.zeros(model.nv, dtype=np.float64)
a = np.zeros(model.nv, dtype=np.float64)
Y_temp = pin.computeJointTorqueRegressor(model, data, q, v, a)

# Now compute the full regressor matrix using random samples
W_standard = sysid_helpers.calculate_standard_regressor(model, data, q_rand, dq_rand, ddq_rand)
print("\nStandard regressor matrix W_standard shape: ", W_standard.shape)
print(W_standard[0, :])

print("\nRank of the standard regressor matrix W_standard: ")
print(np.linalg.matrix_rank(W_standard))

# determine 0-columns (unidentificable parameters) in the standard regressor matrix
unidentificable_idx = sysid_helpers.get_unidentificable_parameter_index(W_standard, tol=1e-6)

# show unidentificable parameter 
unidentificable_params = []
for i in unidentificable_idx:
    unidentificable_params.append(standard_param_symbols_list[i])
print("Unidentificable parameters: ")
for p in unidentificable_params:
    print(p)

# remove 0-columns from the standard regressor matrix
W_reduced = np.delete(W_standard, unidentificable_idx, axis=1)
print("deleted columns' indices: ", unidentificable_idx)
print("\nReduced regressor matrix W_reduced shape: ", W_reduced.shape)

# check numerical rank of W_reduced
num_rank_W_reduced = sysid_helpers.calculate_base_parameters(W_standard,standard_param_symbols_list, TOL_QR=1e-6)
print("\nNumerical rank of the reduced regressor matrix W_reduced: ", num_rank_W_reduced)



