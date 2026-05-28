"""Action decoding: handles action trajectories, unnormalization, clipping, gripper mapping."""

from collections import deque

import numpy as np


class ActionChunkBuffer:
    """Buffer that caches policy action chunks and outputs one action per control step.

    Config options (from YAML action_chunk: {...}):
        prediction_horizon: int  -- how many steps the model predicts
        execution_horizon: int   -- how many steps to execute before re-querying
        overlap_strategy: str    -- queue | replace | temporal_ensemble
        action_smoothing: float  -- smoothing factor (0 = none)
    """

    def __init__(
        self,
        prediction_horizon: int = 10,
        execution_horizon: int = 5,
        overlap_strategy: str = "queue",
        action_smoothing: float = 0.0,
    ):
        self._prediction_horizon = prediction_horizon
        self._execution_horizon = execution_horizon
        self._overlap_strategy = overlap_strategy
        self._action_smoothing = action_smoothing
        self._queue: deque = deque()
        self._step_counter = 0
        self._last_action: np.ndarray | None = None

    def add_trajectory(self, trajectory: np.ndarray) -> None:
        """Add a new action trajectory of shape (horizon, action_dim)."""
        if self._overlap_strategy == "replace":
            self._queue.clear()
        for i in range(trajectory.shape[0]):
            self._queue.append(trajectory[i].copy())

    def get_action(self) -> np.ndarray | None:
        """Pop the next action from the buffer. Returns None if empty."""
        self._step_counter += 1
        if not self._queue:
            return None

        action = self._queue.popleft()

        # Action smoothing
        if self._action_smoothing > 0 and self._last_action is not None:
            alpha = self._action_smoothing
            action = alpha * self._last_action + (1.0 - alpha) * action

        self._last_action = action
        return action

    def get_action_or_last(self) -> np.ndarray | None:
        """Return next action, or the last action if buffer is empty, or None if never called."""
        action = self.get_action()
        if action is not None:
            return action
        return self._last_action.copy() if self._last_action is not None else None

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    @property
    def remaining(self) -> int:
        return len(self._queue)


class ActionDecoder:
    """Decodes a raw policy action trajectory into joint-position + gripper commands."""

    def __init__(
        self,
        action_dim: int = 7,
        action_mode: str = "joint_position",
        joint_mins: np.ndarray | None = None,
        joint_maxs: np.ndarray | None = None,
        gripper_open: float = 1.0,
        gripper_close: float = 0.0,
    ):
        self._action_dim = action_dim
        self._action_mode = action_mode
        self._joint_mins = joint_mins
        self._joint_maxs = joint_maxs
        self._gripper_open = gripper_open
        self._gripper_close = gripper_close

    def decode(self, action: np.ndarray) -> np.ndarray:
        """Decode a single action vector into the final control command."""
        # Clip joint positions if bounds are provided
        if self._joint_mins is not None and self._joint_maxs is not None:
            n_joints = min(len(self._joint_mins), self._action_dim - 1)
            action[:n_joints] = np.clip(action[:n_joints], self._joint_mins, self._joint_maxs)

        # Gripper: threshold to binary open/close
        gripper_val = action[-1]
        action = action.copy()
        action[-1] = self._gripper_open if gripper_val > 0.5 else self._gripper_close

        return action
