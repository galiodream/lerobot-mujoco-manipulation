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
    grasp_body_names: tuple[str, ...] = ()
    pre_grasp_z_offset: float = 0.15
    grasp_z_offset: float = 0.03
    lift_z_offset: float = 0.20
    gripper_close_val: float = 0.0
    gripper_open_val: float = 1.0
    ik_stepsize: float = 1.0
    ik_eps: float = 1e-2
    place_z_offset: float = 0.08
    target_rpy_deg: tuple[float, float, float] = (90.0, 0.0, 90.0)
    pos_tolerance: float = 0.03
    gripper_tolerance: float = 0.05
    max_stage_steps: int = 40
    grasp_hold_steps: int = 10
    release_hold_steps: int = 10


class PickPlaceOracle:
    """Scripted pick-and-place oracle using IK."""

    STAGES = ["pre_grasp", "descend", "close", "lift", "move_target", "open", "retreat"]

    def __init__(self, env, config: PickPlaceOracleConfig | None = None, joint_names: list[str] | None = None):
        self._env = env  # MujocoManipulationEnv
        self._cfg = config or PickPlaceOracleConfig()
        sim_env = getattr(env, "env", None)
        if sim_env is None:
            sim_env = getattr(env, "_env", None)
        self._sim_env = sim_env
        if sim_env is not None:
            self._cfg.ee_body = getattr(sim_env, "_ee_body", self._cfg.ee_body)
            self._cfg.gripper_open_val = getattr(sim_env, "_gripper_open_val", self._cfg.gripper_open_val)
            self._cfg.gripper_close_val = getattr(sim_env, "_gripper_close_val", self._cfg.gripper_close_val)
            self._cfg.object_body = getattr(sim_env, "obj_target", self._cfg.object_body)
            for key, value in sim_env._rp.get("oracle_config", {}).items():
                setattr(self._cfg, key, value)
        self._joint_names = joint_names or ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self._stage = ""
        self._stage_progress = 0.0
        self._stage_steps = 0
        self._done = False
        self._gripper_open = True

    def reset(self, task: str = "") -> None:
        if "red" in task:
            self._cfg.object_body = "body_obj_mug_5"
        elif "blue" in task:
            self._cfg.object_body = "body_obj_mug_6"
        elif self._sim_env is not None:
            self._cfg.object_body = getattr(self._sim_env, "obj_target", self._cfg.object_body)
        self._stage = "pre_grasp"
        self._stage_progress = 0.0
        self._stage_steps = 0
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

    def _read_gripper_raw(self) -> float:
        if self._sim_env is not None and hasattr(self._sim_env, "_read_gripper_raw"):
            return float(self._sim_env._read_gripper_raw())
        return float(self._cfg.gripper_open_val if self._gripper_open else self._cfg.gripper_close_val)

    def _get_target_rpy(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self._cfg.target_rpy_deg, dtype=np.float64))

    def _get_current_grasp_center(self) -> np.ndarray:
        if not self._cfg.grasp_body_names:
            ee_pos, _ = self._get_ee_pose()
            return ee_pos
        points = [self._env.mujoco_parser.get_p_body(body_name) for body_name in self._cfg.grasp_body_names]
        return np.mean(np.asarray(points, dtype=np.float64), axis=0)

    def _tcp_target_from_grasp_center(self, grasp_center: np.ndarray) -> np.ndarray:
        ee_pos, _ = self._get_ee_pose()
        current_grasp_center = self._get_current_grasp_center()
        return ee_pos + (grasp_center - current_grasp_center)

    def _advance_stage(self, stage: str) -> None:
        self._stage = stage
        self._stage_progress = 0.0
        self._stage_steps = 0

    def _stage_ready(self, target_pos: np.ndarray, tolerance: float | None = None) -> bool:
        ee_pos, _ = self._get_ee_pose()
        pos_tol = self._cfg.pos_tolerance if tolerance is None else tolerance
        pos_err = float(np.linalg.norm(ee_pos - target_pos))
        return pos_err <= pos_tol or self._stage_steps >= self._cfg.max_stage_steps

    def _gripper_ready(self, target_val: float) -> bool:
        err = abs(self._read_gripper_raw() - target_val)
        return err <= self._cfg.gripper_tolerance or self._stage_steps >= self._cfg.max_stage_steps

    def _ik_to(self, target_pos: np.ndarray, target_rpy: np.ndarray | None = None) -> np.ndarray:
        from robot_vla_mujoco.mujoco_env.ik import solve_ik
        from robot_vla_mujoco.mujoco_env.transforms import rpy2r

        parser = self._env.mujoco_parser
        q_init = parser.get_qpos_joints(joint_names=self._joint_names)

        if target_rpy is None:
            target_rpy = self._get_target_rpy()

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
            if self._stage_ready(target_pos, tolerance=max(self._cfg.pos_tolerance, 0.04)):
                self._advance_stage("descend")

        elif self._stage == "descend":
            grasp_center = np.array([obj_xy[0], obj_xy[1], obj_z + self._cfg.grasp_z_offset])
            target_pos = self._tcp_target_from_grasp_center(grasp_center)
            q = self._ik_to(target_pos)
            gripper = self._cfg.gripper_open_val
            if self._stage_ready(target_pos, tolerance=max(self._cfg.pos_tolerance, 0.025)):
                self._advance_stage("close")

        elif self._stage == "close":
            current_q = self._env.mujoco_parser.get_qpos_joints(joint_names=self._joint_names)
            q = current_q
            gripper = self._cfg.gripper_close_val
            self._gripper_open = False
            if self._gripper_ready(self._cfg.gripper_close_val) and self._stage_steps >= self._cfg.grasp_hold_steps:
                self._advance_stage("lift")

        elif self._stage == "lift":
            current_ee, _ = self._get_ee_pose()
            grasp_center = np.array([obj_xy[0], obj_xy[1], current_ee[2] + self._cfg.lift_z_offset])
            target_pos = self._tcp_target_from_grasp_center(grasp_center)
            q = self._ik_to(target_pos)
            gripper = self._cfg.gripper_close_val
            if self._stage_ready(target_pos, tolerance=max(self._cfg.pos_tolerance, 0.05)):
                self._advance_stage("move_target")

        elif self._stage == "move_target":
            grasp_center = np.array([target_xy[0], target_xy[1], target_z + self._cfg.place_z_offset])
            target_pos = self._tcp_target_from_grasp_center(grasp_center)
            q = self._ik_to(target_pos)
            gripper = self._cfg.gripper_close_val
            if self._stage_ready(target_pos, tolerance=max(self._cfg.pos_tolerance, 0.035)):
                self._advance_stage("open")

        elif self._stage == "open":
            current_q = self._env.mujoco_parser.get_qpos_joints(joint_names=self._joint_names)
            q = current_q
            gripper = self._cfg.gripper_open_val
            self._gripper_open = True
            if self._gripper_ready(self._cfg.gripper_open_val) and self._stage_steps >= self._cfg.release_hold_steps:
                self._advance_stage("retreat")

        elif self._stage == "retreat":
            current_ee, _ = self._get_ee_pose()
            target_pos = np.array([current_ee[0], current_ee[1], current_ee[2] + 0.1])
            q = self._ik_to(target_pos)
            gripper = self._cfg.gripper_open_val
            if self._stage_ready(target_pos, tolerance=max(self._cfg.pos_tolerance, 0.05)):
                self._done = True

        else:
            q = self._env.mujoco_parser.get_qpos_joints(joint_names=self._joint_names)
            gripper = self._cfg.gripper_open_val

        self._stage_steps += 1
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
