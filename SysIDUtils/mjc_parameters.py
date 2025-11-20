import mujoco
import numpy as np
import numpy.typing as npt


def skew(vector):
    return np.cross(np.eye(vector.size), vector.reshape(-1))


def get_body_dynamic_parameters(mjmodel, body_id) -> npt.ArrayLike:
    """Get the dynamic parameters \theta of a body
    theta = [m, mx, my, mz, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]

    Args:
        mjmodel (mujoco.MjModel): The mujoco model
        body_id (int): The id of the body

    Returns:
        npt.ArrayLike: theta of the body
    """
    mass = mjmodel.body(body_id).mass[0]
    rc = mjmodel.body(body_id).ipos
    diag_inertia = mjmodel.body(body_id).inertia

    # get the orientation of the body
    r_flat = np.zeros(9)
    mujoco.mju_quat2Mat(r_flat, mjmodel.body(body_id).iquat)

    R = r_flat.reshape(3, 3)

    shift = mass * skew(rc) @ skew(rc)
    mjinertia = R @ np.diag(diag_inertia) @ R.T - shift

    upper_triangular = np.array(
        [
            mjinertia[0, 0],
            mjinertia[0, 1],
            mjinertia[1, 1],
            mjinertia[0, 2],
            mjinertia[1, 2],
            mjinertia[2, 2],
        ]
    )

    return np.concatenate([[mass], mass * rc, upper_triangular])


def get_full_dynamics_parameters_value(mjmodel) -> npt.ArrayLike:
    """Get the full set of dynamic parameters of the model

    Args:
        mjmodel (mujoco.MjModel): The mujoco model

    Returns:
        npt.ArrayLike: The full set of dynamic parameters of the model
    """
    nbody = mjmodel.nbody
    all_params = np.zeros(nbody * 10)

    for i in range(nbody):
        theta = get_body_dynamic_parameters(mjmodel, i)
        all_params[10 * i : 10 * (i + 1)] = theta
    # remove the 1st 10 parameters corresponding to the world body
    all_params = all_params[10:]
    return all_params

def get_full_dynamics_parameters_dictionary(mjmodel) -> dict:
    """Get the full set of dynamic parameters of the model as a dictionary

    Args:
        mjmodel (mujoco.MjModel): The mujoco model

    Returns:
        dict: The full set of dynamic parameters of the model as a dictionary
    """
    nbody = mjmodel.nbody
    all_params_dict = {}

    for i in range(nbody):
        theta = get_body_dynamic_parameters(mjmodel, i)
        param_names = [
            f"m_{i}",
            f"mx_{i}",
            f"my_{i}",
            f"mz_{i}",
            f"I_xx_{i}",
            f"I_xy_{i}",
            f"I_yy_{i}",
            f"I_xz_{i}",
            f"I_yz_{i}",
            f"I_zz_{i}",
        ]
        for name, value in zip(param_names, theta):
            all_params_dict[name] = value
    # remove the 1st 10 parameters corresponding to the world body
    for name in list(all_params_dict.keys())[:10]:
        del all_params_dict[name]
    return all_params_dict

def set_dynamic_parameters(mjmodel, body_id, theta: npt.ArrayLike) -> None:
    """Set the dynamic parameters to a body

    Args:
        mjmodel (mujoco.MjModel): The mujoco model
        body_id (int): The id of the body
        theta (npt.ArrayLike): The dynamic parameters of the body
    """

    mass = theta[0]
    rc = theta[1:4] / mass
    inertia = theta[4:]
    inertia_full = np.array(
        [
            [inertia[0], inertia[1], inertia[3]],
            [inertia[1], inertia[2], inertia[4]],
            [inertia[3], inertia[4], inertia[5]],
        ]
    )

    # shift the inertia
    inertia_full += mass * skew(rc) @ skew(rc)

    # eigen decomposition
    eigval, eigvec = np.linalg.eigh(inertia_full)
    R = eigvec
    diag_inertia = eigval

    # check if singular, then abort
    if np.any(np.isclose(diag_inertia, 0)):
        raise ValueError("Cannot deduce inertia matrix because RIR^T is singular.")

    # set the mass
    mjmodel.body(body_id).mass = np.array([mass])
    mjmodel.body(body_id).ipos = rc

    # set the orientation
    mujoco.mju_mat2Quat(mjmodel.body(body_id).iquat, R.flatten())

    # set the inertia
    mjmodel.body(body_id).inertia = diag_inertia

def get_dynamic_parameters(mjmodel, body_id) -> npt.ArrayLike:
    """Get the dynamic parameters \theta of a body
    theta = [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]

    Args:
        mjmodel (mujoco.MjModel): The mujoco model
        body_id (int): The id of the body

    Returns:
        npt.ArrayLike: theta of the body
    """
    mass = mjmodel.body(body_id).mass[0]
    rc = mjmodel.body(body_id).ipos
    diag_inertia = mjmodel.body(body_id).inertia

    # get the orientation of the body
    r_flat = np.zeros(9)
    mujoco.mju_quat2Mat(r_flat, mjmodel.body(body_id).iquat)

    R = r_flat.reshape(3, 3)

    shift = mass * skew(rc) @ skew(rc)
    mjinertia = R @ np.diag(diag_inertia) @ R.T - shift

    upper_triangular = np.array(
        [
            mjinertia[0, 0],
            mjinertia[0, 1],
            mjinertia[1, 1],
            mjinertia[0, 2],
            mjinertia[1, 2],
            mjinertia[2, 2],
        ]
    )

    return np.concatenate([[mass], mass * rc, upper_triangular])