import pinocchio as pin
import numpy as np
from scipy import linalg, signal

def get_standard_parameters(model):
    """
    This function returns a dictionary of the standard dynamic parameters of the robot model.
    """
    Y = []
    params = []

    params_name = (
        "m",
        "mx",
        "my",
        "mz",
        "Ixx",
        "Ixy",
        "Ixz",
        "Iyy",
        "Iyz",
        "Izz",
    )
    # range_ = len(model.inertias) # = 7, including the world body
    for i in range(1, len(model.inertias)):
        # model.inertias[i].toDynamicParameters() return dynamic parameters in the order of
        # ['m', 'mx','my','mz','Ixx','Ixy','Ixz', 'Iyy', 'Iyz','Izz'] of body i
        P = model.inertias[i].toDynamicParameters()
        P_mod = np.zeros(P.shape[0])
        P_mod[0] = P[0]  # m
        P_mod[1] = P[1]  # mx
        P_mod[2] = P[2]  # my
        P_mod[3] = P[3]  # mz
        P_mod[4] = P[4]  # Ixx
        P_mod[5] = P[5]  # Ixy
        P_mod[6] = P[6]  # Ixz
        P_mod[7] = P[8]  # Iyy
        P_mod[8] = P[9]  # Iyz
        P_mod[9] = P[7]  # Izz

        for j in params_name:
            params.append(j + str(i))
        for k in P_mod:
            Y.append(k)
    params_std = dict(zip(params, Y)) # (mi: value, mxi: value, ...)
    return params_std

def get_list_standard_param_symbols (model):
    param_sysmbols = []
    params_name = (
        "m",
        "mx",
        "my",
        "mz",
        "Ixx",
        "Ixy",
        "Ixz",
        "Iyy",
        "Iyz",
        "Izz",
    )
    for i in range(1, len(model.inertias)):
        for j in params_name:
            param_sysmbols.append(j + str(i))
    return param_sysmbols

def calculate_standard_regressor (pin_model, pin_data, q_rand, dq_rand, ddq_rand):
    """
    This function calculates the standard regressor matrix of the robot model.
    Equation of motion:

    tau = Y(q, dq, ddq) * X

    where:
    tau: joint torques with size = ndof
    Y: standard regressor matrix with size = ndof x np
    np: number of standard parameters = 10n, n is the number of links
    ndof: number of degrees of freedom
    X: vector of the standard parameter = [m1, mx1, my1, mz1, Ixx1, Ixy1, Ixz1, Iyy1, Iyz1, Izz1,...] with size = 10n, n is the number of links
    we can extend to include friction parameters if needed

    model: pinocchio model
    data: pinocchio data
    q_rand: vector of measured joint positions
    dq_rand: vector of joint velocities
    ddq_rand: vector of joint accelerations
    """
    nb_samples = q_rand.shape[0] # q_rand samples are row-stacked
    no_dof = pin_model.nv # number of actuated joints = number of moving bodies
    no_params = 10 * pin_model.nv  # exclude the universe body

    Y = np.zeros((nb_samples * no_dof, no_params)) # initialize the regressor matrix
    for i in range(nb_samples):
        Y_temp = pin.computeJointTorqueRegressor(pin_model, pin_data, q_rand[i, :], dq_rand[i, :], ddq_rand[i, :])
        # stack the regressor matrices in row
        Y[i * no_dof : (i + 1) * no_dof, :] = Y_temp
    return Y    
def get_unidentificable_parameter_index (Y, tol = 1e-6):
    """
    This function returns the index of unidentifiable parameters in the standard regressor matrix.
    Y: standard regressor matrix
    tol: tolerance for identifying unidentifiable parameters
    """
    # check each column of Y if its norm is less than tol
    idx_unidentifiable = []
    for i in range(Y.shape[1]):
        if np.linalg.norm(Y[:, i]) < tol:
            idx_unidentifiable.append(i)
    return idx_unidentifiable


def calculate_base_parameters (Y, param_st, TOL_QR=1e-6):
    """
    This function calculates the base parameters of the robot model.

    Y: standard regressor matrix
    param_st: standard parameter vector
    TOL_QR: tolerance for QR decomposition

    returns:
    Y_base: base regressor matrix
    param_base: base parameter vector
    """
    # step 1: remove zero columns from Y
    idx_unidentifiable = get_unidentificable_parameter_index(Y)
    Y_reduced = np.delete(Y, idx_unidentifiable, axis=1) # remove the 0 columns
    param_reduce = param_st
    for idx in idx_unidentifiable:
        # use pop(index) to remove the parameter at the specific index 
        param_reduce.pop(idx) # remove the corresponding parameters

    # Perform QR decomposition with column pivoting
    # p is pivot indices
    # the permutation matrix P can be constructed as P = np.eye(p.size)[p]
    # the decomposition equation is:
    # Y_reduced @ P.T = Q @ R
    # if Y_reduced is rank deficient, R = [ R1 R2]
    # where R1 is full rank upper triangular matrix
    # numerical rank of Y_reduced is determined by checking the diagonal elements of R
    # if abs(R[i,i]) < TOL_QR, then rank < i
    Q, R, p = linalg.qr(Y_reduced, pivoting=True)
    num_rank = min(Y_reduced.shape)

    for i in range(min(R.shape)): # scanning the diagonal elements of R
        if abs(R[i, i]) < TOL_QR:
            numerical_rank_Y = i
            break
    R1 = R[:numerical_rank_Y, :numerical_rank_Y]
    Q1 = Q[:, :numerical_rank_Y]
    R2 = R[:numerical_rank_Y, numerical_rank_Y:]

    return numerical_rank_Y

    