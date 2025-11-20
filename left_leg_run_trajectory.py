
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mujoco.viewer import launch_passive
from scipy import signal

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
        acceleration = (120*t**3/T**5 - 180*t**2/T**4 + 60*t/T**3)*ref_angl
        return angle.reshape((len(ref_angl),1)), velocity.reshape((len(ref_angl),1)), acceleration.reshape((len(ref_angl),1))
    else:
        return ref_angl.reshape((len(ref_angl),1)), np.zeros(len(ref_angl))

def Get_Joints_Pos (mj_model, mj_data):
    q = []
    joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
    for joint in joint_names:
        q.append(mj_data.joint(joint).qpos)
    n = len(q)
    if n> 12:
        q = q[1:] # ignore the velocity of the torso, only take the joint position
    return np.array(q)

def Get_Joint_Vel (mj_model, mj_data):
    q_vel = []
    joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
    for joint in joint_names:
        q_vel.append(mj_data.joint(joint).qvel)
    n = len(q_vel)
    if n> 12:
        q_vel = q_vel[1:] # ignore the velocity of the torso, only take the joint position
    return np.array(q_vel)
 
def generate_ref_trajectory(joint_range, init_duration, trajectory_duration, frequency, feedback_angle, t):
    """
    Generate a simple sinusoidal reference trajectory within joint limits.
    t: time input
    joint_range: contains lower and upper limits
    feed_back_angle: current joint position
    """
    ref_angle, ref_vel, ref_accel = np.zeros(len(feedback_angle)), np.zeros(len(feedback_angle)), np.zeros(len(feedback_angle)) # vectorized
    starting_point = (joint_range[:,0] + joint_range[:,1]) / 2
    if t>= trajectory_duration + init_duration:
        ref_angle = feedback_angle
        ref_vel = np.zeros(len(feedback_angle))
        ref_accel = np.zeros(len(feedback_angle))
        return ref_angle, ref_vel, ref_accel
    if t<= init_duration:
        ref_angle, ref_vel, ref_accel = step_ref(t, init_duration, starting_point)
        return ref_angle, ref_vel, ref_accel
    
    if t > init_duration:
        ref_angle = starting_point + 0.8*(joint_range[:,1] - joint_range[:,0]) / 2 * np.sin(2 * np.pi * frequency * (t - init_duration))
        ref_vel = 0.8*(joint_range[:,1] - joint_range[:,0]) / 2 * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * (t - init_duration))
        ref_accel = -0.8*(joint_range[:,1] - joint_range[:,0]) / 2 * (2 * np.pi * frequency)**2 * np.sin(2 * np.pi * frequency * (t - init_duration))
        return ref_angle.reshape((len(ref_angle),1)), ref_vel.reshape((len(ref_vel),1)), ref_accel.reshape((len(ref_accel),1))
    return ref_angle, ref_vel, ref_accel

class first_order_lfilter:
    """
    Apply first-order low-pass filter to input signal x with output y.
    x: input signal
    y: previous output signal
    """
    def __init__(self, a, w_c, Ts):
        self.a = a  # smoothing factor between 0 and 1
        self.w_c = w_c  # cutoff frequency
        self.Ts = Ts  # sampling time
        self.alpha = self.w_c * self.Ts / (1 + self.w_c * self.Ts)
        self.y_prev = 0.0  # previous output initialized to zero

    def filter(self, x, y_prev):
        y = self.alpha * x + (1 - self.alpha) * y_prev
        self.y_prev = y # update previous output
        return y
    

# ──────────────────────────────────────────────────────────────────────
#   Init Mujoco model and data
# ──────────────────────────────────────────────────────────────────────
LEFT_LEG_MJCF = "robot_models/mjcf/left_leg.xml"
mj_model = mujoco.MjModel.from_xml_path(LEFT_LEG_MJCF)
mj_data = mujoco.MjData(mj_model)
mj_time_step = mj_model.opt.timestep
control_freq = 250 # control frequency 250 Hz
dt_control = 1.0 / control_freq
sampling_freq = control_freq
simu_time = 0.0
time_count = 0
t_control = 0

# get joint ranges (lower and upper limits) in the model
joint_ranges = np.zeros((mj_model.njnt, 2))
for i in range(mj_model.njnt):
    joint_ranges[i, :] = mj_model.jnt_range[i,:]
trajectory_duration = 20 # seconds, time of trajectory running

# Init a first-order low-pass filter
w_c = 2 * np.pi * 0.5  # cutoff frequency 0.5 Hz
Ts = mj_time_step  # sampling time
lpf_pos = first_order_lfilter(a=0.1, w_c=w_c, Ts=Ts)
lpf_vel = first_order_lfilter(a=0.1, w_c=w_c, Ts=Ts)
traj_freq = np.array([0.2, 0.25, 0.3, 0.35, 0.35, 0.45])  # frequency for each joint

# #-----------------------------------------------
# # For testing the generated reference trajectory
# duration = 10 # seconds
# time = np.arange(0, duration, mj_model.opt.timestep)
# # plot a reference trajectory for joint 0
# ref_joint_angle = []
# ref_joint_velocity = []

# for t in time:
#     ref_pos, ref_vel = generate_ref_trajectory(joint_ranges, 2, duration-2, 0.5, mj_data.qpos[:], t)
#     ref_pos = lpf_pos.filter(ref_pos, lpf_pos.y_prev)
#     ref_vel = lpf_vel.filter(ref_vel, lpf_vel.y_prev)
#     ref_joint_angle.append(ref_pos)
#     ref_joint_velocity.append(ref_vel)

# plt.figure()
# plt.plot(time, np.array(ref_joint_angle)[:,0], label="ref joint angle")
# plt.title("Reference Trajectory for Joint 1")
# plt.xlabel("Time (s)")
# plt.ylabel("Joint Angle (rad)")
# plt.grid()

# plt.figure()
# plt.plot(time, np.array(ref_joint_velocity)[:,0], label="ref joint velocity")
# plt.title("Reference Trajectory for Joint 1")
# plt.xlabel("Time (s)")
# plt.ylabel("Joint Velocity (rad/s)")
# plt.grid()

# plt.legend()
# plt.show()

# ──────────────────────────────────────────────────────────────────────
#   Init Low-level PD impedence controller
# ──────────────────────────────────────────────────────────────────────
Joint_Kp = np.diag([500, 500, 150, 350, 250, 250]) # N.m/rad
Joint_Kd = np.diag([1.5, 1.5, 1.5, 1.5, 1.5, 1.5]) #N.ms/rad
PDController= PD_Impedence_Control(Joint_Kp, Joint_Kd) # init robot's joint PD controller

viewer = launch_passive(mj_model, mj_data)

save_data = []
ref_joint_angle = np.zeros((mj_model.njnt,1))   #initial reference joint angle
ref_joint_velocity = np.zeros((mj_model.njnt,1))#initial reference joint velocity
ref_joint_acceleration = np.zeros((mj_model.njnt,1))#initial reference joint acceleration

if __name__ == "__main__":
    while viewer.is_running() and simu_time < trajectory_duration:
        mujoco.mj_step(mj_model, mj_data) # forward dynamics
        simu_time = mj_data.time # simulation times        
        # ──────────────────────────────────────────────────────────────────────
        #   Get feedback joints' angles and velocity
        # ──────────────────────────────────────────────────────────────────────
        robot_joint_feb_angl = Get_Joints_Pos(mj_model, mj_data) # Get feedback position
        robot_joint_feb_vel = Get_Joint_Vel(mj_model, mj_data)
        # ──────────────────────────────────────────────────────────────────────
        #   Generate reference trajectory with frequency =  control frequency
        # ──────────────────────────────────────────────────────────────────────
        ref_joint_angle, ref_joint_velocity, ref_joint_acceleration = generate_ref_trajectory(joint_ranges, 2, trajectory_duration, traj_freq, robot_joint_feb_angl, t_control)
        ref_joint_angle = lpf_pos.filter(ref_joint_angle, lpf_pos.y_prev)
        ref_joint_velocity = lpf_vel.filter(ref_joint_velocity, lpf_vel.y_prev)

        # ──────────────────────────────────────────────────────────────────────
        #   Data saving at sampling frequency
        # ──────────────────────────────────────────────────────────────────────
        time_count += 1
        if time_count >= (int)(dt_control/mj_time_step):
            t_control += dt_control
            time_count = 0
            robot_joint_feb_angl = Get_Joints_Pos(mj_model, mj_data) # Get feedback position
            robot_joint_feb_vel = Get_Joint_Vel(mj_model, mj_data)
            joint_torque = mj_data.qfrc_actuator  # Get joint torque from actuators           
            # # stack data to save
            data_point = np.concatenate(([t_control], ref_joint_angle.flatten(), robot_joint_feb_angl.flatten(),robot_joint_feb_vel.flatten(), joint_torque.flatten(), ref_joint_acceleration.flatten()))
            save_data.append(data_point)
        # ──────────────────────────────────────────────────────────────────────
        #   Compute PD control torque
        # ──────────────────────────────────────────────────────────────────────
        control_torque = PDController.PD_Control_Calculate(ref_joint_angle, ref_joint_velocity, robot_joint_feb_angl, robot_joint_feb_vel)
        mj_data.ctrl = np.array(control_torque).reshape(1,-1)        
        viewer.sync()        
    viewer.close()


save_data = np.array(save_data)

# Create data directory if it doesn't exist
import os
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Save data to file
save_path_npy = os.path.join(data_dir, 'left_leg_simulation_data.npy')
np.save(save_path_npy, save_data)
print(f"Data saved to '{save_path_npy}' with shape: {save_data.shape}")

# Also save as CSV for easy viewing
header = 'time,' \
'ref_q1,ref_q2,ref_q3,ref_q4,ref_q5,ref_q6' \
',q1,q2,q3,q4,q5,q6,' \
'vel_q1,vel_q2,vel_q3,vel_q4,vel_q5,vel_q6,' \
'tau1,tau2,tau3,tau4,tau5,tau6,' \
'ref_acc1,ref_acc2,ref_acc3,ref_acc4,ref_acc5,ref_acc6'

save_path_csv = os.path.join(data_dir, 'left_leg_simulation_data.csv')
np.savetxt(save_path_csv, save_data, delimiter=',', header=header, comments='', fmt='%.6f')
print(f"Data also saved to '{save_path_csv}'")

# plot results after simulation

time_data = save_data[:,0]
ref_angle_data = save_data[:,1:7]
joint_pos_data = save_data[:,7:13]
joint_vel_data = save_data[:,13:19]
control_torque_data = save_data[:,19:25]

plt.figure()
for i in range(6):
    plt.subplot(3,2,i+1)
    plt.plot(time_data, ref_angle_data[:,i], label='Reference Angle')
    plt.plot(time_data, joint_pos_data[:,i], label='Joint Position')
    plt.title(f'Joint {i+1} Position Tracking')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid()
plt.tight_layout()

plt.figure()
for i in range(6):
    plt.subplot(3,2,i+1)
    plt.plot(time_data, control_torque_data[:,i], label='Control Torque')
    plt.title(f'Joint {i+1} Control Torque')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N.m)')
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()