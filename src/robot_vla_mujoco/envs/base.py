"""Base classes and protocols for environments."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PolicyAdapter(ABC):
    """Unified policy interface."""

    @abstractmethod
    def reset(self, task: str) -> None: ...

    @abstractmethod
    def predict_action_trajectory(self, observation: dict[str, Any]) -> np.ndarray: ...

    @abstractmethod
    def close(self) -> None: ...


class SuccessCondition(ABC):
    """Protocol for pluggable success-condition logic."""

    @abstractmethod
    def reset(self, env: Any, success_params: dict, task_context: dict | None = None) -> None: ...

    @abstractmethod
    def update(self, env: Any, obs: dict, action: np.ndarray | None) -> None: ...

    @abstractmethod
    def is_success(self) -> bool: ...

    @abstractmethod
    def metrics(self) -> dict[str, Any]: ...
