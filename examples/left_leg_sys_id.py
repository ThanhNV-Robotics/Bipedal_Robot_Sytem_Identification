import copy
import mujoco
from dm_control import mjcf
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import scipy
import matplotlib.pyplot as plt
from mujoco.viewer import launch_passive

class PD_Impedence_Control: # MIMO, joint position PD control
    def __init__(self, Kp, Kd):
        n = np.shape(Kp)[1]
        self.Kp = Kp
        self.Kd = Kd
        
    def PD_Control_Calculate (self, ref_pos, ref_vel, fb_pos, fb_vel): # MIMO
        # tqr_ff is the feedforward torque

        torque = self.Kp@(ref_pos - fb_pos) + self.Kd@(ref_vel - fb_vel)
        return torque # return torque vector applied to robot's joints

def step_ref (t, T, ref_angl): # to generate referenc angle
    if  t <= T:
        angle = (6*(t/T)**5 - 15*(t/T)**4 + 10*(t/T)**3)*ref_angl
        velocity = (30*t**4/T**5 - 60*t**3/T**4 + 30*t**2/T**3)*ref_angl
        return angle, velocity
    else:
        return ref_angl, 0
 
def generate_ref_trajectory(joint_range, init_duration, trajectory_duration, frequency, feedback_angle, t):
    """
    Generate a simple sinusoidal reference trajectory within joint limits.
    t: time input
    joint_range: contains lower and upper limits
    feed_back_angle: current joint position
    """
    ref_angle, ref_vel = 0.0, 0.0
    starting_point = (joint_range[0] + joint_range[1]) / 2
    if t>= trajectory_duration + init_duration:
        return np.array([feedback_angle, 0.0])
    if t<= init_duration:
        ref_angle, ref_vel = step_ref(t, init_duration, starting_point)
        return np.array([ref_angle, ref_vel])
    
    if t > init_duration:
        ref_angle = starting_point + 0.9*(joint_range[1] - joint_range[0]) / 2 * np.sin(2 * np.pi * frequency * (t - init_duration))
        ref_vel = 0.9*(joint_range[1] - joint_range[0]) / 2 * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * (t - init_duration))
        return np.array([ref_angle, ref_vel])
    return np.array([ref_angle, ref_vel])    

    
LEFT_LEG_MJCF = "data/models/Left_Leg.xml"
mj_model = mujoco.MjModel.from_xml_path(LEFT_LEG_MJCF)
mj_data = mujoco.MjData(mj_model)

# get joint ranges (lower and upper limits) in the model
joint_ranges = np.zeros((mj_model.njnt, 2))
for i in range(mj_model.njnt):
    joint_ranges[i, :] = mj_model.jnt_range[i,:]

duration = 20 # seconds, time of trajectory running

# ──────────────────────────────────────────────────────────────────────
#   Init Low-level PD impedence controller
# ──────────────────────────────────────────────────────────────────────
Joint_Kp = np.diag([650, 600, 150, 350, 250, 250]) # N.m/rad
Joint_Kd = np.diag([1.5, 1.5, 1.5, 1.5, 1.5, 1.5]) #N.ms/rad
PDController= PD_Impedence_Control(Joint_Kp, Joint_Kd) # init robot's joint PD controller


t = np.arange(0, duration, mj_model.opt.timestep)
# plot a reference trajectory for joint 0
ref_trajectory_joint1 = []
for time in t:
    ref_traj = generate_ref_trajectory(joint_ranges[3,:], 2, duration-2, 0.5, mj_data.qpos[3], time)
    ref_trajectory_joint1.append(ref_traj[0])

plt.plot(t, ref_trajectory_joint1)
plt.title("Reference Trajectory for Joint 1")
plt.xlabel("Time (s)")
plt.ylabel("Joint Angle (rad)")
plt.grid()
plt.show()