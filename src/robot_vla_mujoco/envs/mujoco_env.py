"""MujocoManipulationEnv -- unified MuJoCo simulation environment.

Wraps the reference-project SimpleEnv2 with the interface from PROJECT_PLAN:
  - reset / step / render / get_observation / is_success / close
  - step returns (observation, reward, terminated, truncated, info)
  - supports viewer on/off via MODEL_SIM_VIEWER env var
  - supports headless EGL offscreen rendering via init_offscreen_renderer
"""

import os
import time
from typing import Any

import mujoco
import numpy as np

from robot_vla_mujoco.common.seed import set_seed
from robot_vla_mujoco.envs.success_conditions import (
    SUCCESS_CONDITION_REGISTRY,
)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in {"0", "false", "no", "off"}


class MujocoManipulationEnv:
    def __init__(
        self,
        xml_path: str,
        action_type: str = "joint_angle",
        state_type: str = "joint_angle",
        seed: int | None = None,
        initialize_viewer: bool | None = None,
        camera_width: int = 640,
        camera_height: int = 480,
        success_config: dict | None = None,
    ):
        if initialize_viewer is None:
            initialize_viewer = _env_bool("MODEL_SIM_VIEWER", True)

        from robot_vla_mujoco.mujoco_env.y_env2 import SimpleEnv2

        self._xml_path = xml_path
        self._action_type = action_type
        self._state_type = state_type

        self._env = SimpleEnv2(
            xml_path=xml_path,
            action_type=action_type,
            state_type=state_type,
            seed=seed,
            initialize_viewer=initialize_viewer,
        )

        if not initialize_viewer:
            self._env.init_offscreen_renderer(width=camera_width, height=camera_height)

        # Success condition
        sc = success_config or {}
        sc_type = sc.get("type", "pick_place")
        sc_params = sc.get("params", {})
        cls = SUCCESS_CONDITION_REGISTRY.get(sc_type)
        self._success_condition = cls()
        self._success_condition.reset(self, sc_params)

        self._max_steps = 800
        self._step_count = 0
        self._episode_seed = seed

    # -- Gymnasium-style interface -------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._episode_seed = seed
            set_seed(seed)

        self._env.reset(seed=seed)
        self._step_count = 0

        sc_config = (options or {}).get("success_config", {})
        if sc_config:
            cls = SUCCESS_CONDITION_REGISTRY.get(sc_config.get("type", "pick_place"))
            self._success_condition = cls()
            self._success_condition.reset(self, sc_config.get("params", {}))
        else:
            self._success_condition.reset(self, {})

        obs = self.get_observation()
        info = self._build_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._env.step(action)
        sim_dt = self._env.env.model.opt.timestep
        physics_steps = max(1, round(0.15 / sim_dt)) # 0.15 s/step * 800 steps = 120s (sim time)
        self._env.step_env(nstep=physics_steps)
        self._step_count += 1

        self._success_condition.update(self, {}, action)

        obs = self.get_observation()
        reward = 1.0 if self._success_condition.is_success() else 0.0
        terminated = self._success_condition.is_success()
        truncated = self._step_count >= self._max_steps
        info = self._build_info()
        info["success"] = self._success_condition.is_success()
        info["success_metrics"] = self._success_condition.metrics()

        return obs, reward, terminated, truncated, info

    def render(self, fast: bool = False, show_side_view: bool = False, teleop: bool = False, idx: int = 0) -> None:
        self._env.render(fast=fast, show_side_view=show_side_view, teleop=teleop, idx=idx)

    def close(self) -> None:
        self._env.close_offscreen_renderer()
        if hasattr(self._env, "env") and self._env.env.use_mujoco_viewer:
            self._env.env.close_viewer()

    # -- Observation / success -----------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        state = self._env.get_joint_state()
        images: dict[str, np.ndarray] = {}
        try:
            rgb_agent, rgb_ego = self._env.grab_image_fast()
            images["front"] = rgb_agent
            images["wrist"] = rgb_ego
        except Exception:
            pass

        return {
            "observation.images.front": images.get("front"),
            "observation.images.wrist": images.get("wrist"),
            "observation.state": state,
            "task": getattr(self._env, "instruction", ""),
        }

    def is_success(self) -> bool:
        return self._success_condition.is_success()

    def _build_info(self) -> dict[str, Any]:
        return {
            "step_count": self._step_count,
            "sim_time": self._env.env.get_sim_time() if hasattr(self._env, "env") else 0.0,
            "seed": self._episode_seed,
        }

    # -- Viewer control -------------------------------------------------------

    @property
    def viewer_enabled(self) -> bool:
        return self._env.env.use_mujoco_viewer if hasattr(self._env, "env") else False

    def is_viewer_alive(self) -> bool:
        if hasattr(self._env, "env"):
            return self._env.env.is_viewer_alive()
        return True  # headless: always "alive"

    # -- Passthrough for scripts that need SimpleEnv2 internals ----------------

    @property
    def env(self):
        """Access underlying SimpleEnv2 for advanced usage."""
        return self._env

    @property
    def mujoco_parser(self):
        """Access the underlying MuJoCoParserClass (for body/joint queries)."""
        return self._env.env

    @property
    def mujoco_model(self) -> mujoco.MjModel:
        return self._env.env.model

    @property
    def mujoco_data(self) -> mujoco.MjData:
        return self._env.env.data
