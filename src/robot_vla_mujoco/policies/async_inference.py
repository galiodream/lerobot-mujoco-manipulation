"""Async inference engine: runs policy inference in a background thread.

Decouples the render/physics loop (60 Hz) from the policy inference loop (e.g. 3 Hz),
so that slow VLA models do not block simulation rendering.
"""

import threading
from typing import Any

import numpy as np

from robot_vla_mujoco.policies.base import PolicyAdapter


class AsyncInferenceEngine:
    def __init__(self, policy: PolicyAdapter):
        self._policy = policy

        self._result_traj: np.ndarray | None = None
        self._latest_obs: dict[str, Any] | None = None
        self._obs_updated = False
        self._running = True
        self._inferring = False
        self._lock = threading.Lock()
        self._event = threading.Event()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def reset(self, task: str) -> None:
        self._policy.reset(task)
        with self._lock:
            self._result_traj = None
            self._latest_obs = None

    def is_idle(self) -> bool:
        with self._lock:
            return not self._inferring

    def update_observation(self, observation: dict[str, Any]) -> None:
        with self._lock:
            self._latest_obs = observation
            self._obs_updated = True
        self._event.set()

    def collect_trajectory(self) -> np.ndarray | None:
        with self._lock:
            traj = self._result_traj
            self._result_traj = None
        return traj

    def close(self) -> None:
        self._running = False
        self._event.set()
        self._thread.join(timeout=2.0)

    def _worker(self) -> None:
        while self._running:
            self._event.wait()
            if not self._running:
                break
            self._event.clear()

            while self._running:
                with self._lock:
                    obs = self._latest_obs
                    self._obs_updated = False
                    if obs is None:
                        break
                    self._inferring = True

                try:
                    action_traj = self._policy.predict_action_trajectory(obs)
                finally:
                    with self._lock:
                        self._inferring = False

                with self._lock:
                    self._result_traj = action_traj

                with self._lock:
                    if not self._obs_updated:
                        break
