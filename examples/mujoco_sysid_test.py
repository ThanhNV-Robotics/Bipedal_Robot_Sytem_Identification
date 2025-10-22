import copy
import mujoco
from dm_control import mjcf
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import scipy
import matplotlib.pyplot as plt
from mujoco.viewer import launch_passive

CART_POLE_MJCF = "data/models/cart_pole/scene.xml"

sim_model = mujoco.MjModel.from_xml_path(CART_POLE_MJCF)
sim_data = mujoco.MjData(sim_model)
renderer = mujoco.Renderer(sim_model, height=480, width=640)

print(sim_model)

def compute_gains(model, data, configuration, Q, R):
    # 1. Set configuration and find control that stabilizes it (ctrl0)
    newdata = mujoco.MjData(model)
    newdata = copy.copy(data)

    mujoco.mj_resetData(model, newdata)
    newdata.qpos = configuration
    # compute the control
    mujoco.mj_forward(model, newdata)
    newdata.qacc = 0
    mujoco.mj_inverse(model, newdata)

    # define control and configuration to linearize around
    print(newdata.qfrc_actuator)
    #ctrl0 = newdata.qfrc_inverse.copy() @ np.linalg.pinv(newdata.actuator_moment)
    ctrl0 = 0.0
    qpos0 = newdata.qpos.copy()

    # 2. Compute LQR gains given weights
    mujoco.mj_resetData(model, newdata)
    newdata.ctrl = ctrl0
    newdata.qpos = qpos0

    # 3. Allocate the A and B matrices, compute them.
    A = np.zeros((2 * model.nv, 2 * model.nv))
    B = np.zeros((2 * model.nv, model.nu))
    epsilon = 1e-6
    flg_centered = True
    mujoco.mjd_transitionFD(model, newdata, epsilon, flg_centered, A, B, None, None)

    # Solve discrete Riccati equation.
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # Compute the feedback gain matrix K.
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    return ctrl0, K

if __name__ == "__main__":
    # Parameters.
    DURATION = 3  # seconds
    BALANCE_STD = 0.2  # actuator units

    dq = np.zeros(sim_model.nv)

    qpos0 = np.array([0, np.pi-0.2])
    target = np.array([0, np.pi])

    Q = np.diag([1, 1, 1, 1]) * 1e2
    R = np.eye(sim_model.nu)

    # ctrl0, K = compute_gains(uncertain_model, uncertain_data, target)
    ctrl0, K = compute_gains(sim_model, sim_data, target, Q, R)

    # Reset data, set initial pose.
    mujoco.mj_resetData(sim_model, sim_data)
    sim_data.qpos = qpos0

    qhist = []

    viewer = launch_passive(sim_model, sim_data)

    #while sim_data.time < DURATION:
    while viewer.is_running():
        # Get state difference dx.
        mujoco.mj_differentiatePos(sim_model, dq, 1, target, sim_data.qpos)
        dx = np.hstack((dq, sim_data.qvel)).T

        # LQR control law.
        sim_data.ctrl = ctrl0 - K @ dx + np.random.randn(sim_model.nu) * BALANCE_STD

        # Step the simulation.
        mujoco.mj_step(sim_model, sim_data)
        viewer.sync()

    # Save history.
        qhist.append(sim_data.qpos.copy())
    qhist = np.array(qhist)
    plt.title("Stabilization of ideal model")
    plt.plot(qhist[:, 0], label="x")
    plt.plot(qhist[:, 1], label="theta")

    plt.hlines(0, 0, len(qhist), color="black", linestyle="--")
    plt.hlines(np.pi, 0, len(qhist), color="black", linestyle="--")

    plt.grid()
    plt.legend()
    plt.show()