import numpy as np

robot = 'h1'  # 'h1', 'aliengo', 'go2', 'b2', 'hyqreal2', 'z1', 'piper_l', 'a2', 'pegasus'

# Unitree H1 19-DOF model order. DDS uses 20 physical motor slots; physical
# motor ID 9 is unused and is deliberately absent from this mapping.
H1_JOINT_NAMES = (
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_joint", "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
)
H1_JOINT_TO_MOTOR = (7, 3, 4, 5, 10, 8, 0, 1, 2, 11, 6, 16, 17, 18, 19, 12, 13, 14, 15)
H1_UNUSED_MOTOR_ID = 9

# Gains actually used by Holosoma's h1_19dof_loco_real and
# h1_19dof_loco_wbt_real inference paths. Those configurations keep the
# real_go transport/motor mapping but override its standalone real-robot gains
# with h1_19dof.motor_kp/motor_kd. Values here remain in physical DDS order.
_H1_RUNTIME_MOTOR_KP = np.array([
    200., 200., 300., 200., 200., 300., 300., 200., 200., 0.,
    40., 40., 100., 100., 100., 100., 100., 100., 100., 100.,
])
_H1_RUNTIME_MOTOR_KD = np.array([
    5., 5., 6., 5., 5., 6., 6., 5., 5., 0.,
    2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
])


USE_MUJOCO_SIMULATION = False
USE_MUJOCO_RENDER = USE_MUJOCO_SIMULATION = False

# ----------------------------------------------------------------------------------------------------------------
if robot == "h1":
    Kp = _H1_RUNTIME_MOTOR_KP[np.asarray(H1_JOINT_TO_MOTOR)]
    Kd = _H1_RUNTIME_MOTOR_KD[np.asarray(H1_JOINT_TO_MOTOR)]

elif(robot == "aliengo"):
    # Order FL, FR, RL, RR
    Kp = np.array([25., 25., 25.,
                   25., 25., 25.,
                   25., 25., 25.,
                   25., 25., 25.])
                   
    Kd = np.array([1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5])

elif(robot == "go2"):
    # Order FL, FR, RL, RR
    Kp = np.array([25., 25., 25.,
                   25., 25., 25.,
                   25., 25., 25.,
                   25., 25., 25.])
                   
    Kd = np.array([1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5,
                   1.5, 1.5, 1.5])

elif(robot == "b2"):
    # Order FL, FR, RL, RR
    Kp = np.array([100., 100., 100.,
                   100., 100., 100.,
                   100., 100., 100.,
                   100., 100., 100.])

    Kd = np.array([5., 5., 5.,
                   5., 5., 5.,
                   5., 5., 5.,
                   5., 5., 5.])

elif(robot == "a2"):
    # Order FL, FR, RL, RR
    Kp = np.array([40., 40., 40.,
                   40., 40., 40.,
                   40., 40., 40.,
                   40., 40., 40.])

    Kd = np.array([2., 2., 2.,
                   2., 2., 2.,
                   2., 2., 2.,
                   2., 2., 2.])

elif(robot == "hyqreal2"):
    # Order FL, FR, RL, RR
    Kp = np.array([175., 175., 175.,
                   175., 175., 175.,
                   175., 175., 175.,
                   175., 175., 175.])

    Kd = np.array([20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.])

elif(robot == "pegasus"):
    # Order FL, FR, RL, RR
    Kp = np.array([120., 120., 120.,
                   120., 120., 120.,
                   120., 120., 120.,
                   120., 120., 120.])

    Kd = np.array([6., 6., 6.,
                   6., 6., 6.,
                   6., 6., 6.,
                   6., 6., 6.])

elif(robot =="z1"):
    # Order joint1-joint2-joint3-joint4-joint5-joint6-gripper
    Kp = np.array([80., 120., 120.,
                   80., 80., 80.,
                   80.])

    Kd = np.array([2.5, 2.5, 2.5,
                   2.5, 2.5, 2.5,
                   2.5])

elif(robot =="piper_l"):
    # Order joint1-joint2-joint3-joint4-joint5-joint6-gripper
    Kp = np.array([125.0, 125.0, 125.0,
                   7.5, 7.5, 7.5,
                   10.0])

    Kd = np.array([10.0, 10.0, 10.0,
                   0.6, 0.6, 0.6,
                   0.8])

else:
    raise ValueError(f"Robot {robot} not supported")

frequency_collection = 500  # Hz; H1 low-level DDS/control frequency

# Backward-compatible scalar defaults used only by old datasets without gains.
kp = float(np.median(Kp))
kd = float(np.median(Kd))
