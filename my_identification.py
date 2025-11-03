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
print("number of joints: " + str(model.njoints))
print("number of dofs: " + str(model.nq))
print("number of bodies: " + str(len(model.inertias)-1))  #exclude the universe body
print("number of actuated joints: " + str(model.nv))

# 2. generate a list containing the full set of standard parameters
standard_params = sysid_helpers.get_standard_parameters(model)
print(standard_params)

# 3. determine a based parameters
# 3.1 first we build a standard regressor matrix corresponding to the standard parameter vector

