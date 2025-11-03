import pinocchio as pin
import numpy as np

def get_standard_parameters(model):
    """
    This function returns a dictionary of the standard dynamic parameters of the robot model.
    """
    phi = []
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
            phi.append(k)
    params_std = dict(zip(params, phi)) # (mi: value, mxi: value, ...)
    return params_std

def calculate_standard_regressor (pin_model, pin_data, q_rand, dq_rand, ddq_rand):
    """
    This function calculates the standard regressor matrix of the robot model.
    Equation of motion:

    tau = Phi(q, dq, ddq) * X

    where:
    tau: joint torques with size = ndof
    Phi: standard regressor matrix with size = ndof x np
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
    nb_samples = q_rand.shape[0]
    no_dof = pin_model.nv # number of actuated joints = number of moving bodies
    no_params = 10 * pin_model.nv  # exclude the universe body

    Phi = np.zeros((nb_samples * no_dof, no_params)) # initialize the regressor matrix
    Phi_model = np.zeros((nb_samples * no_dof, no_params))    

def calculate_base_parameters (Phi, X, zero_q, zero_qd, zero_qdd, zero_g, no_dof):
    """
    This function calculates the base parameters of the robot model.
    Phi: standard regressor matrix
    X: standard parameter vector
    zero_q: bool type, if True, the joint positions are zero
    zero_qd: bool type, if True, the joint velocities are zero
    zero_qdd: bool type, if True, the joint accelerations are zero
    zero_g: bool type, if True, the gravity vector is zero
    no_dof: int, number of degrees of freedom
    """
    pass # To be implemented