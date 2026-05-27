"""Closed-loop execution with action chunking and environment interaction.

Provides a higher-level runner that ties together:
  - PolicyAdapter (any VLA policy)
  - ActionChunkBuffer (temporal action management)
  - MujocoManipulationEnv (simulation)
"""

from typing import Any

import numpy as np

from robot_vla_mujoco.policies.action_decoder import ActionChunkBuffer


class ClosedLoopRunner:
    """Runs a policy in closed-loop with action chunking.

    Usage:
        runner = ClosedLoopRunner(env, policy, action_chunk_config)
        result = runner.run_episode(max_steps=400, seed=42)
    """

    def __init__(
        self,
        env: Any,
        policy: Any,
        action_chunk_config: dict | None = None,
    ):
        self._env = env
        self._policy = policy

        cfg = action_chunk_config or {}
        self._chunk_buffer = ActionChunkBuffer(
            prediction_horizon=cfg.get("prediction_horizon", 10),
            execution_horizon=cfg.get("execution_horizon", 5),
            overlap_strategy=cfg.get("overlap_strategy", "queue"),
            action_smoothing=cfg.get("action_smoothing", 0.0),
        )

    def run_episode(
        self,
        max_steps: int = 400,
        seed: int | None = None,
        render: bool = False,
        render_fast: bool = False,
    ) -> dict[str, Any]:
        """Run one episode and return result dict."""
        obs, info = self._env.reset(seed=seed)
        task = obs.get("task", "")
        self._policy.reset(task)
        self._chunk_buffer = ActionChunkBuffer(
            prediction_horizon=self._chunk_buffer._prediction_horizon,
            execution_horizon=self._chunk_buffer._execution_horizon,
        )

        episode_reward = 0.0
        for step in range(max_steps):
            if self._chunk_buffer.is_empty:
                action_traj = self._policy.predict_action_trajectory(obs)
                self._chunk_buffer.add_trajectory(action_traj)

            action = self._chunk_buffer.get_action()
            if action is None:
                action = np.zeros(7, dtype=np.float32)

            obs, reward, terminated, truncated, info = self._env.step(action)
            episode_reward += reward

            if render:
                self._env.render(fast=render_fast)

            if terminated or truncated:
                break

        return {
            "success": info.get("success", False),
            "steps": step + 1,
            "reward": float(episode_reward),
        }
