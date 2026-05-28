"""Policy rollout: runs a policy in closed-loop on the MuJoCo environment."""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla_mujoco.policies.action_decoder import ActionChunkBuffer
from robot_vla_mujoco.policies.async_inference import AsyncInferenceEngine


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

    Async inference: policy runs in a background thread continuously, the
    main loop picks up the latest trajectory and feeds new observations.
    """
    if action_chunk_config is None:
        action_chunk_config = {"prediction_horizon": 5, "execution_horizon": 5}

    chunk_buffer = ActionChunkBuffer(
        prediction_horizon=action_chunk_config.get("prediction_horizon", 10),
        execution_horizon=action_chunk_config.get("execution_horizon", 5),
        overlap_strategy=action_chunk_config.get("overlap_strategy", "replace"),
        action_smoothing=action_chunk_config.get("action_smoothing", 0.0),
    )

    async_engine = AsyncInferenceEngine(policy)

    obs, info = env.reset(seed=seed)
    task = obs.get("task", "")
    async_engine.reset(task)

    # Prime the buffer with a synchronous first inference
    new_traj = policy.predict_action_trajectory(obs)
    chunk_buffer.add_trajectory(new_traj)

    async_engine.update_observation(obs)

    episode_reward = 0.0

    t_start = time.perf_counter()

    for step in range(max_steps):
        new_traj = async_engine.collect_trajectory()
        if new_traj is not None:
            chunk_buffer.add_trajectory(new_traj)

        action = chunk_buffer.get_action_or_last()
        if action is None:
            action = np.zeros(7, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        async_engine.update_observation(obs)
        episode_reward += reward

        if render:
            env.render(fast=render_fast, idx=episode_idx)

        if terminated or truncated:
            break

    async_engine.close()

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
