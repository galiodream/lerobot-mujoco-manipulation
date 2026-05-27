"""Shared types for the project."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Observation:
    images: dict[str, np.ndarray] = field(default_factory=dict)
    state: np.ndarray | None = None
    task: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, img in self.images.items():
            result[f"observation.images.{key}"] = img
        if self.state is not None:
            result["observation.state"] = self.state
        if self.task:
            result["task"] = self.task
        return result


@dataclass
class ActionTrajectory:
    """Output of a policy: shape (horizon, action_dim)."""
    data: np.ndarray

    @property
    def horizon(self) -> int:
        return self.data.shape[0]

    @property
    def action_dim(self) -> int:
        return self.data.shape[1]


@dataclass
class EpisodeMetadata:
    episode_id: str
    task_id: str
    variant_id: str
    object_type: str
    object_initial_pose: list[float]
    target_pose: list[float]
    difficulty: str
    seed: int
    success: bool
    env_config_hash: str
