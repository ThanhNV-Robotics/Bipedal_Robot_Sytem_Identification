# Authors: Van Thanh Nguyen

import pinocchio as pin
import numpy as np
from scipy import linalg

def get_standard_parameters_dict(model):
    """
    This function returns a dictionary of the standard dynamic parameters of the robot model.
    Input: model -> pinocchio model
    Output: params_std -> dictionary of standard parameters
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

def get_standard_parameters_values(model):
    """
    This function returns a list of standard dynamic parameter values of the robot model.
    Input: model -> pinocchio model
    Output: params_values -> list of standard parameter values
    """
    Y = []

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

        for k in P_mod:
            Y.append(k)
    return Y

def get_list_standard_param_symbols (model):
    """
    This function returns a list of standard dynamic parameter symbols of the robot model.
    Input: model -> pinocchio model
    Output: param_symbols -> list of standard parameter symbols
    """
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

    #Y = np.zeros((nb_samples * no_dof, no_params)) # initialize the regressor matrix
    Y = np.empty((0, no_params))  # initialize the regressor matrix as a list
    for i in range(nb_samples):
        Y_temp = pin.computeJointTorqueRegressor(pin_model, pin_data, q_rand[i, :], dq_rand[i, :], ddq_rand[i, :])
        # stack the regressor matrices in row
        # Y[i * no_dof : (i + 1) * no_dof, :] = Y_temp
        Y = np.vstack((Y, Y_temp))
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

def compute_numerical_rank (Y, TOL_QR=1e-6):
    """
    This function computes the numerical rank of the matrix Y using QR decomposition with column pivoting.
    Y: input matrix
    TOL_QR: tolerance for determining numerical rank
    returns:
    numerical_rank: numerical rank of the matrix Y
    """
    # Perform QR decomposition with pivoting
    # p is pivot indices
    # the permutation matrix P can be constructed as P = np.eye(p.size)[p]
    # the decomposition equation is:
    # Y_reduced @ P.T = Q @ R
    # if Y_reduced is rank deficient, R = [ R1 R2]
    # where R1 is full rank upper triangular matrix
    # numerical rank of Y_reduced is determined by checking the diagonal elements of R
    # if abs(R[i,i]) < TOL_QR, then rank < i
    Q, R, p = linalg.qr(Y, pivoting=True)
    numerical_rank = min(Y.shape) #initialize numerical rank
    for i in range(min(R.shape)): # scanning the diagonal elements of R
        if abs(R[i, i]) < TOL_QR:
            numerical_rank = i
            break
    return numerical_rank

def calculate_base_parameters_symbols (Y, param_st, TOL_QR=1e-6):
    """
    This function calculates the base parameters of the robot model.

    Y: standard regressor matrix
    param_st: standard parameter vector /full paramter symbol list
    TOL_QR: tolerance for QR decomposition

    returns:
    X_base: base parameter vector
    beta: coefficient matrix for dependent parameters
    independent_idx: indices of independent parameters in param_st
    dependent_idx: indices of dependent parameters in param_st
    """
    machine_eps = np.finfo(float).eps  # Approx. 2.22e-16
    tolerance = np.sqrt(machine_eps)  # Approx. 1.49e-8

    # step 1: remove zero columns from Y
    idx_unidentifiable = get_unidentificable_parameter_index(Y)
    Y_reduced = np.delete(Y, idx_unidentifiable, axis=1) # remove the 0 columns
    X = param_st 
    for idx in idx_unidentifiable: # remove unidentifiable parameters
        # use pop(index) to remove the parameter at the specific index 
        X.pop(idx) # remove the corresponding parameters

    # Perform QR decomposition with pivoting
    # p is pivot indices
    # the permutation matrix P can be constructed as P = np.eye(p.size)[p]
    # the decomposition equation is:
    # Y_reduced @ P.T = Q @ R
    # if Y_reduced is rank deficient, R = [ R1 R2]
    # where R1 is full rank upper triangular matrix
    # numerical rank of Y_reduced is determined by checking the diagonal elements of R
    # if abs(R[i,i]) < TOL_QR, then rank < i
    Q, R, p = linalg.qr(Y_reduced, pivoting=True)

    # Compute numerical rank
    numerical_rank_Y = min(Y_reduced.shape)
    for i in range(min(R.shape)): # scanning the diagonal elements of R
        if abs(R[i, i]) < TOL_QR:
            numerical_rank_Y = i
            break
    R1 = R[:numerical_rank_Y, :numerical_rank_Y] # R1 is full rank, squared, upper triangular matrix
    R2 = R[:numerical_rank_Y, numerical_rank_Y:]
    P = np.eye(p.size)[p] # permutation matrix

    beta = np.linalg.solve(R1, R2) # beta = R1^{-1} * R2, linear combination coefficient of base parameters
    #beta[np.abs(beta) < np.sqrt(np.finfo(float).eps)] = 0 # check for very small values and set them to zero
    beta[np.abs(beta) < tolerance] = 0
    # Check relationship in regressor matrix Y
    # we have to transpose P because the relation is:
    # Y_reduced @ P.T = Q @ R
    P1 = np.transpose(P)[:, :numerical_rank_Y] # correspond to independent part
    P2 = np.transpose(P)[:, numerical_rank_Y:] # correspond to dependent part
    Y1 = Y_reduced@P1 # columns correspond to independent part 
    Y2 = Y_reduced@P2 # columns correspond to dependent part
    Y2_estimated = Y1 @ beta # beta is the coefficient matrix
    residual = Y2 - Y2_estimated
    max_residual = np.max(np.abs(residual))
    if max_residual > 1e-6:
        print("Warning: High residual in base parameter calculation: ", max_residual)
    else:
        print("Relationship in regressor matrix found")
        print("Max residual in base parameter calculation: ", max_residual)

    # Find base parameters
    independent_idx = p[:numerical_rank_Y] # indices of independent parameters in X
    dependent_idx = p[numerical_rank_Y:] # indices of dependent parameters in X
    
    X1 = [] # independent parameters
    X2 = [] # dependent parameters

    for idx in independent_idx: # get independent parameters
        X1.append(X[idx])
    for idx in dependent_idx: # get dependent parameters
        X2.append(X[idx])
    X_base = [] # base parameters
    # X_base = X1 + beta@X2

    # As X1 and X2 are string, we can not perform matrix multiplication directly
    # here we loop manually to create the string representation of the base parameters
    for i in range(len(X1)):
        param = X1[i]
        for j in range(len(X2)):
            coef = beta[i, j]
            if coef != 0:  # Skip zero coefficients
                sign = '+' if coef >= 0 else ''
                param += f" {sign}{coef}*{X2[j]}"
        X_base.append(param)
    return X_base, beta, independent_idx, dependent_idx

def compute_base_regressor (Y, TOL_QR=1e-6):
    """
    This function computes the base regressor matrix of the robot model.

    Y: standard regressor matrix
    TOL_QR: tolerance for QR decomposition

    returns:
    Y_base: base regressor matrix
    """
    machine_eps = np.finfo(float).eps  # Approx. 2.22e-16
    tolerance = np.sqrt(machine_eps)  # Approx. 1.49e-8

    # step 1: remove zero columns from Y
    idx_unidentifiable = get_unidentificable_parameter_index(Y)
    Y_reduced = np.delete(Y, idx_unidentifiable, axis=1) # remove the 0 columns

    # Perform QR decomposition with pivoting
    Q, R, p = linalg.qr(Y_reduced, pivoting=True)

    # Compute numerical rank
    numerical_rank_Y = min(Y_reduced.shape)
    for i in range(min(R.shape)): # scanning the diagonal elements of R
        if abs(R[i, i]) < TOL_QR:
            numerical_rank_Y = i
            break

    R1 = R[:numerical_rank_Y, :numerical_rank_Y] # R1 is full rank upper triangular matrix
    R2 = R[:numerical_rank_Y, numerical_rank_Y:]
    P = np.eye(p.size)[p] # permutation matrix

    beta = np.linalg.solve(R1, R2) # beta = R1^{-1} * R2, linear combination coefficient of base parameters
    #beta[np.abs(beta) < np.sqrt(np.finfo(float).eps)] = 0 # check for very small values and set them to zero
    beta[np.abs(beta) < tolerance] = 0
    # Check relationship in regressor matrix Y
    # we have to transpose P because the relation is:
    # Y_reduced @ P.T = Q @ R
    P1 = np.transpose(P)[:, :numerical_rank_Y] # correspond to independent part
    P2 = np.transpose(P)[:, numerical_rank_Y:] # correspond to dependent part
    Y1 = Y_reduced@P1 # columns correspond to independent part 
    Y2 = Y_reduced@P2 # columns correspond to dependent part
    Y2_estimated = Y1 @ beta # beta is the coefficient matrix
    residual = Y2 - Y2_estimated
    max_residual = np.max(np.abs(residual))

    # return the index also
    independent_idx = p[:numerical_rank_Y] # indices of independent parameters in X
    dependent_idx = p[numerical_rank_Y:] # indices of dependent parameters in X

    if max_residual > 1e-6:
        print("Warning: High residual in base parameter calculation: ", max_residual)
        print("Returning reduced regressor matrix without base parameter reduction.")
        return Y_reduced, beta, independent_idx, dependent_idx
    else:
        print("Relationship in regressor matrix found")
        print("Max residual in base parameter calculation: ", max_residual)
        Y_base = Y1 # Y_base is the independent part Y1
        return Y_base, beta, independent_idx, dependent_idx
    
def compute_base_model (Y_standard, standard_param_values,TOL_QR=1e-6):
    """
    This function computes the base parameter values of the robot model.

    Y_standard: standard regressor matrix
    standard_param_values: standard parameter values

    returns:
    base_regressor: base regressor matrix
    base_param_values: base parameter values
    """
    machine_eps = np.finfo(float).eps  # Approx. 2.22e-16
    tolerance = np.sqrt(machine_eps)  # Approx. 1.49e-8

    # step 1: remove zero columns from Y
    idx_unidentifiable = get_unidentificable_parameter_index(Y_standard)
    Y_reduced = np.delete(Y_standard, idx_unidentifiable, axis=1) # remove the 0 columns
    X_reduced = np.delete(standard_param_values, idx_unidentifiable, axis=0) # remove the unidentified parameters

    # Perform QR decomposition with pivoting
    # p is pivot indices
    # the permutation matrix P can be constructed as P = np.eye(p.size)[p]
    # the decomposition equation is:
    # Y_reduced @ P.T = Q @ R
    # if Y_reduced is rank deficient, R = [ R1 R2]
    # where R1 is full rank upper triangular matrix
    # numerical rank of Y_reduced is determined by checking the diagonal elements of R
    # if abs(R[i,i]) < TOL_QR, then rank < i
    Q, R, p = linalg.qr(Y_reduced, pivoting=True)

    # Compute numerical rank
    numerical_rank_Y = min(Y_reduced.shape)
    for i in range(min(R.shape)): # scanning the diagonal elements of R
        if abs(R[i, i]) < TOL_QR:
            numerical_rank_Y = i
            break
    R1 = R[:numerical_rank_Y, :numerical_rank_Y] # R1 is full rank upper triangular matrix
    R2 = R[:numerical_rank_Y, numerical_rank_Y:]
    P = np.eye(p.size)[p] # permutation matrix

    beta = np.linalg.solve(R1, R2) # beta = R1^{-1} * R2, linear combination coefficient of base parameters
    #beta[np.abs(beta) < np.sqrt(np.finfo(float).eps)] = 0 # check for very small values and set them to zero
    beta[np.abs(beta) < tolerance] = 0
    # Check relationship in regressor matrix Y
    # we have to transpose P because the relation is:
    # Y_reduced @ P.T = Q @ R
    P1 = np.transpose(P)[:, :numerical_rank_Y] # correspond to independent part
    P2 = np.transpose(P)[:, numerical_rank_Y:] # correspond to dependent part
    Y1 = Y_reduced@P1 # columns correspond to independent part 
    Y2 = Y_reduced@P2 # columns correspond to dependent part
    Y2_estimated = Y1 @ beta # beta is the coefficient matrix
    residual = Y2 - Y2_estimated
    max_residual = np.max(np.abs(residual))
    if max_residual > 1e-6:
        print("Warning: High residual in base parameter calculation: ", max_residual)
    else:
        print("Relationship in regressor matrix found")
        print("Max residual in base parameter calculation: ", max_residual)

    # Find base parameters
    X1 = np.transpose(P1)@X_reduced
    X2 = np.transpose(P2)@X_reduced
    # independent_idx = p[:numerical_rank_Y] # indices of independent parameters in X
    # dependent_idx = p[numerical_rank_Y:] # indices of dependent parameters in X
    
    # X1 = [] # independent parameters
    # X2 = [] # dependent parameters

    # for idx in independent_idx: # get independent parameters
    #     X1.append(X_reduced[idx])
    # for idx in dependent_idx: # get dependent parameters
    #     X2.append(X_reduced[idx])

    X_base = X1 + beta@X2
    Y_base = Y1

    return Y_base, X_base, P1, P2

    