"""LeRobot-format wrapper for MujocoManipulationEnv (stub for Milestone 2)."""

from typing import Any

import numpy as np


class LeRobotEnvWrapper:
    """Wraps a MujocoManipulationEnv to produce LeRobot-compatible observations."""

    def __init__(self, env: Any):
        self._env = env

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        obs, _ = self._env.reset(seed=seed)
        return obs

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()
