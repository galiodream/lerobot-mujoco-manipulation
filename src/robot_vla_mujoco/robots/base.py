"""Robot profile base class."""

from dataclasses import dataclass, field


@dataclass
class RobotProfile:
    name: str
    arm_dof: int = 6
    gripper_dof: int = 1
    action_dim: int = 7
    state_dim: int = 7
    joint_names: list[str] = field(default_factory=list)
    gripper_open_value: float = 1.0
    gripper_close_value: float = 0.0
    end_effector_body: str = "tcp_link"
    gripper_joint: str = "rh_r1"
