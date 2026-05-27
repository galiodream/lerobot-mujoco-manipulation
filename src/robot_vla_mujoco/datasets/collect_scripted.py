"""Scripted oracle policy for pick-and-place demo collection.

Implements a staged pick-place pipeline:
  1. pre-grasp  — move EE above the target object
  2. descend    — lower EE to grasp height
  3. close      — close gripper
  4. lift       — raise object
  5. move-target — move above target location (plate)
  6. open       — open gripper to release
  7. retreat    — move EE away
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_vla_mujoco.common.registry import Registry

ORACLE_REGISTRY = Registry("oracle_policy")


@dataclass
class PickPlaceOracleConfig:
    object_body: str = "body_obj_mug_5"
    target_body: str = "body_obj_plate_11"
    ee_body: str = "tcp_link"
    pre_grasp_z_offset: float = 0.15
    grasp_z_offset: float = 0.03
    lift_z_offset: float = 0.20
    gripper_close_val: float = 0.0
    gripper_open_val: float = 1.0
    ik_stepsize: float = 1.0
    ik_eps: float = 1e-2
    place_z_offset: float = 0.08


class PickPlaceOracle:
    """Scripted pick-and-place oracle using IK."""

    STAGES = ["pre_grasp", "descend", "close", "lift", "move_target", "open", "retreat"]

    def __init__(self, env, config: PickPlaceOracleConfig | None = None, joint_names: list[str] | None = None):
        self._env = env  # MujocoManipulationEnv
        self._cfg = config or PickPlaceOracleConfig()
        self._joint_names = joint_names or ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self._stage = ""
        self._stage_progress = 0.0
        self._done = False
        self._gripper_open = True

    def reset(self, task: str = "") -> None:
        self._stage = "pre_grasp"
        self._stage_progress = 0.0
        self._done = False
        self._gripper_open = True

    def _get_object_xy(self) -> np.ndarray:
        parser = self._env.mujoco_parser
        p = parser.get_p_body(self._cfg.object_body)
        return np.array([p[0], p[1]])

    def _get_object_z(self) -> float:
        return float(self._env.mujoco_parser.get_p_body(self._cfg.object_body)[2])

    def _get_target_xy(self) -> np.ndarray:
        parser = self._env.mujoco_parser
        p = parser.get_p_body(self._cfg.target_body)
        return np.array([p[0], p[1]])

    def _get_target_z(self) -> float:
        return float(self._env.mujoco_parser.get_p_body(self._cfg.target_body)[2])

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        parser = self._env.mujoco_parser
        return parser.get_pR_body(self._cfg.ee_body)

    def _ik_to(self, target_pos: np.ndarray, target_rpy: np.ndarray | None = None) -> np.ndarray:
        from robot_vla_mujoco.mujoco_env.ik import solve_ik
        from robot_vla_mujoco.mujoco_env.transforms import rpy2r

        parser = self._env.mujoco_parser
        q_init = parser.get_qpos_joints(joint_names=self._joint_names)

        if target_rpy is None:
            target_rpy = np.deg2rad([90, 0, 90])

        R_trgt = rpy2r(target_rpy)

        q_result, ik_err, _ = solve_ik(
            env=parser,
            joint_names_for_ik=self._joint_names,
            body_name_trgt=self._cfg.ee_body,
            q_init=q_init,
            p_trgt=target_pos,
            R_trgt=R_trgt,
            max_ik_tick=200,
            ik_stepsize=self._cfg.ik_stepsize,
            ik_eps=self._cfg.ik_eps,
            ik_th=np.radians(5.0),
            verbose=False,
            verbose_warning=False,
        )
        return q_result

    def get_action(self) -> np.ndarray:
        """Return the next action: joint angles (6) + gripper (1) = 7D."""
        if self._done:
            q = self._env.mujoco_parser.get_qpos_joints(joint_names=self._joint_names)
            gripper = self._cfg.gripper_open_val
            return np.concatenate([q, [gripper]], dtype=np.float32)

        obj_xy = self._get_object_xy()
        obj_z = self._get_object_z()
        target_xy = self._get_target_xy()
        target_z = self._get_target_z()

        if self._stage == "pre_grasp":
            target_pos = np.array([obj_xy[0], obj_xy[1], obj_z + self._cfg.pre_grasp_z_offset])
            q = self._ik_to(target_pos)
            gripper = self._cfg.gripper_open_val
            if self._stage_progress > 0.95:
                self._stage = "descend"
                self._stage_progress = 0.0

        elif self._stage == "descend":
            target_pos = np.array([obj_xy[0], obj_xy[1], obj_z + self._cfg.grasp_z_offset])
            q = self._ik_to(target_pos)
            gripper = self._cfg.gripper_open_val
            if self._stage_progress > 0.9:
                self._stage = "close"
                self._stage_progress = 0.0

        elif self._stage == "close":
            current_q = self._env.mujoco_parser.get_qpos_joints(joint_names=self._joint_names)
            q = current_q
            gripper = self._cfg.gripper_close_val
            self._gripper_open = False
            if self._stage_progress > 0.5:
                self._stage = "lift"
                self._stage_progress = 0.0

        elif self._stage == "lift":
            current_ee, _ = self._get_ee_pose()
            target_pos = np.array([obj_xy[0], obj_xy[1], current_ee[2] + self._cfg.lift_z_offset])
            q = self._ik_to(target_pos)
            gripper = self._cfg.gripper_close_val
            if self._stage_progress > 0.9:
                self._stage = "move_target"
                self._stage_progress = 0.0

        elif self._stage == "move_target":
            target_pos = np.array([target_xy[0], target_xy[1], target_z + self._cfg.lift_z_offset])
            q = self._ik_to(target_pos)
            gripper = self._cfg.gripper_close_val
            if self._stage_progress > 0.95:
                self._stage = "open"
                self._stage_progress = 0.0

        elif self._stage == "open":
            current_q = self._env.mujoco_parser.get_qpos_joints(joint_names=self._joint_names)
            q = current_q
            gripper = self._cfg.gripper_open_val
            self._gripper_open = True
            if self._stage_progress > 0.5:
                self._stage = "retreat"
                self._stage_progress = 0.0

        elif self._stage == "retreat":
            current_ee, _ = self._get_ee_pose()
            target_pos = np.array([current_ee[0], current_ee[1], current_ee[2] + 0.1])
            q = self._ik_to(target_pos)
            gripper = self._cfg.gripper_open_val
            if self._stage_progress > 0.9:
                self._done = True

        else:
            q = self._env.mujoco_parser.get_qpos_joints(joint_names=self._joint_names)
            gripper = self._cfg.gripper_open_val

        self._stage_progress = min(self._stage_progress + 0.05, 1.0)

        # Return 7D action: 6 joint angles + 1 gripper value
        return np.concatenate([q, [gripper]], dtype=np.float32)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def stage(self) -> str:
        return self._stage


# Register
ORACLE_REGISTRY.register("pick_place")(PickPlaceOracle)
