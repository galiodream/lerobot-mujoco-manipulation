"""Pluggable success-condition implementations."""

from typing import Any

import numpy as np

from robot_vla_mujoco.common.registry import Registry

# Global success-condition registry
SUCCESS_CONDITION_REGISTRY = Registry("success_condition")


class PickPlaceCondition:
    """Success condition for pick-and-place tasks.

    Checks:
    1. Object body enters the target zone (XY within distance_threshold).
    2. Object height is stable on target.
    3. Gripper is open (joint below threshold).
    4. End-effector is above a minimum height after release.
    5. Condition must hold for stable_steps consecutive steps.
    """

    def __init__(self):
        self._object_body = ""
        self._target_body = ""
        self._distance_threshold = 0.05
        self._stable_steps = 10
        self._gripper_joint = "rh_r1"
        self._gripper_threshold = 0.1
        self._ee_body = "tcp_link"
        self._ee_release_height = 0.9
        self._stable_count = 0
        self._success = False

    def reset(self, env, success_params: dict, task_context: dict | None = None) -> None:
        self._object_body = success_params.get("object_body", "body_obj_mug_5")
        self._target_body = success_params.get("target_body", "body_obj_plate_11")
        self._distance_threshold = success_params.get("distance_threshold", 0.05)
        self._stable_steps = success_params.get("stable_steps", 10)
        self._gripper_joint = success_params.get("gripper_joint", "rh_r1")
        self._gripper_threshold = success_params.get("gripper_threshold", 0.1)
        self._ee_body = success_params.get("ee_body", "tcp_link")
        self._ee_release_height = success_params.get("ee_release_height", 0.9)
        self._stable_count = 0
        self._success = False

    def update(self, env, obs: dict, action: np.ndarray | None = None) -> None:
        p_obj = env.mujoco_parser.get_p_body(self._object_body)
        p_target = env.mujoco_parser.get_p_body(self._target_body)
        xy_dist = float(np.linalg.norm(p_obj[:2] - p_target[:2]))
        z_diff = float(np.abs(p_obj[2] - p_target[2]))
        # Read gripper: use SimpleEnv2 helper if available (handles both tendon & position)
        if hasattr(env._env, '_read_gripper_raw'):
            gripper = env._env._read_gripper_raw()
        else:
            gripper = float(env.mujoco_parser.get_qpos_joint(self._gripper_joint)[0])
        ee_z = float(env.mujoco_parser.get_p_body(self._ee_body)[2])

        obj_on_target = xy_dist < self._distance_threshold and z_diff < 0.6
        gripper_open = gripper < self._gripper_threshold
        ee_retracted = ee_z > self._ee_release_height

        if obj_on_target and gripper_open and ee_retracted:
            self._stable_count += 1
        else:
            self._stable_count = 0

        if self._stable_count >= self._stable_steps:
            self._success = True

    def is_success(self) -> bool:
        return self._success

    def metrics(self) -> dict[str, Any]:
        return {
            "success": self._success,
            "stable_count": self._stable_count,
            "stable_steps_required": self._stable_steps,
        }


# Register the first success condition
SUCCESS_CONDITION_REGISTRY.register("pick_place")(PickPlaceCondition)
