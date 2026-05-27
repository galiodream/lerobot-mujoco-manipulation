"""Policy rollout: runs a policy in closed-loop on the MuJoCo environment."""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla_mujoco.policies.action_decoder import ActionChunkBuffer


def run_rollout(
    env: Any,
    policy: Any,
    max_steps: int = 400,
    seed: int | None = None,
    render: bool = False,
    render_fast: bool = False,
    action_chunk_config: dict | None = None,
    save_metrics: bool = True,
    output_dir: str | Path | None = None,
    episode_idx: int = 0,
) -> dict[str, Any]:
    """Run a single rollout episode.

    Args:
        env: MujocoManipulationEnv instance.
        policy: PolicyAdapter instance.
        max_steps: Maximum steps per episode.
        seed: RNG seed.
        render: Whether to call env.render() each step.
        render_fast: Fast render (no overlays).
        action_chunk_config: Dict for ActionChunkBuffer.
        save_metrics: Whether to save metrics to disk.
        output_dir: Where to save metrics (if save_metrics).
        episode_idx: Episode index for logging.

    Returns:
        Dict with keys: success, steps, reward, metrics.
    """
    if action_chunk_config is None:
        action_chunk_config = {"prediction_horizon": 1, "execution_horizon": 1}

    chunk_buffer = ActionChunkBuffer(
        prediction_horizon=action_chunk_config.get("prediction_horizon", 10),
        execution_horizon=action_chunk_config.get("execution_horizon", 5),
        overlap_strategy=action_chunk_config.get("overlap_strategy", "queue"),
        action_smoothing=action_chunk_config.get("action_smoothing", 0.0),
    )

    obs, info = env.reset(seed=seed)
    task = obs.get("task", "")
    policy.reset(task)

    episode_reward = 0.0
    episode_metrics: list[dict] = []

    t_start = time.perf_counter()
    for step in range(max_steps):
        # Query policy when buffer is empty
        if chunk_buffer.is_empty:
            action_traj = policy.predict_action_trajectory(obs)
            chunk_buffer.add_trajectory(action_traj)

        action = chunk_buffer.get_action()
        if action is None:
            action = np.zeros(7, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if render:
            env.render(fast=render_fast, idx=episode_idx)

        if terminated or truncated:
            break

    t_end = time.perf_counter()
    episode_time = t_end - t_start

    result = {
        "episode": episode_idx,
        "success": info.get("success", env.is_success()),
        "steps": step + 1,
        "reward": float(episode_reward),
        "time_s": float(episode_time),
        "fps": float((step + 1) / max(episode_time, 0.001)),
        "task": task,
        "seed": seed,
    }

    if save_metrics and output_dir:
        _save_result(result, output_dir)

    return result


def _save_result(result: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "rollout_metrics.jsonl"
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result) + "\n")
