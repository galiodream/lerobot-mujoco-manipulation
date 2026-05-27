"""Policy adapter base class and DummyPolicy for testing."""

from typing import Any

import numpy as np


class PolicyAdapter:
    """Unified policy interface.

    All policy backends (SmolVLA, pi0, pi0.5, LingBot-VLA) implement this.
    """

    def reset(self, task: str) -> None:
        pass

    def predict_action_trajectory(self, observation: dict[str, Any]) -> np.ndarray:
        """Return action trajectory of shape (horizon, action_dim)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class DummyPolicy(PolicyAdapter):
    """Dummy policy that outputs small random actions. For smoke-testing rollout."""

    def __init__(self, action_dim: int = 7, horizon: int = 1):
        self._action_dim = action_dim
        self._horizon = horizon
        self._task = ""

    def reset(self, task: str) -> None:
        self._task = task

    def predict_action_trajectory(self, observation: dict[str, Any]) -> np.ndarray:
        # Small sinusoidal movement to make the robot wiggle slightly
        t = getattr(self, "_t", 0)
        self._t = t + 1
        actions = np.zeros((self._horizon, self._action_dim), dtype=np.float32)
        actions[0, 0] = 0.01 * np.sin(t * 0.1)  # wiggle joint 1
        actions[0, 1] = 0.01 * np.cos(t * 0.1)  # wiggle joint 2
        # Gripper: alternate open/close
        actions[0, -1] = 0.0  # keep open
        return actions

    def close(self) -> None:
        pass
