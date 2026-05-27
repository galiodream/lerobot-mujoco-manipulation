"""OMY (Open Manipulator Y) robot profile."""

from robot_vla_mujoco.robots.base import RobotProfile

OMY_PROFILE = RobotProfile(
    name="omy",
    arm_dof=6,
    gripper_dof=1,
    action_dim=7,
    state_dim=7,
    joint_names=[
        "joint1", "joint2", "joint3",
        "joint4", "joint5", "joint6",
    ],
    gripper_open_value=1.0,
    gripper_close_value=0.0,
    end_effector_body="tcp_link",
    gripper_joint="rh_r1",
)
