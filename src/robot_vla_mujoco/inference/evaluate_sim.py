"""Batch simulation evaluation across multiple episodes."""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla_mujoco.inference.rollout import run_rollout


def evaluate_policy(
    env: Any,
    policy: Any,
    num_episodes: int = 20,
    max_steps: int = 400,
    seed: int = 42,
    render: bool = False,
    render_fast: bool = False,
    action_chunk_config: dict | None = None,
    output_dir: str | Path | None = None,
    vector_env: Any = None,
) -> dict[str, Any]:
    """Evaluate a policy over multiple episodes and return aggregate metrics.

    Args:
        env: MujocoManipulationEnv (single env) or None if using vector_env.
        policy: PolicyAdapter instance.
        num_episodes: Number of evaluation episodes.
        max_steps: Max steps per episode.
        seed: Base RNG seed.
        render: Whether to render during evaluation.
        render_fast: Fast render mode.
        action_chunk_config: Dict for ActionChunkBuffer.
        output_dir: Where to save per-episode metrics.
        vector_env: Optional SyncVectorEnv for parallel evaluation.

    Returns:
        Dict with keys: success_rate, avg_steps, avg_reward, avg_fps,
                         total_time_s, episodes (list of per-episode results).
    """
    results: list[dict] = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    if vector_env is not None:
        # Parallel evaluation via SyncVectorEnv
        num_envs = vector_env.num_envs
        env_indices = list(range(num_episodes))
        env_idx = 0

        seeds = [seed + i for i in range(num_envs)]
        obs_list, _ = vector_env.reset(seeds=seeds)
        policy.reset("")

        # Simple round-robin approach: create per-env state
        env_done = [False] * num_envs
        env_steps = [0] * num_envs
        env_rewards = [0.0] * num_envs
        env_active = list(range(num_envs))
        chunk_buffers = [
            _make_chunk_buffer(action_chunk_config) for _ in range(num_envs)
        ]

        total_episodes_run = 0
        while total_episodes_run < num_episodes:
            actions = []
            for i in range(num_envs):
                if chunk_buffers[i].is_empty and not env_done[i]:
                    action_traj = policy.predict_action_trajectory(obs_list[i])
                    chunk_buffers[i].add_trajectory(action_traj)
                action = chunk_buffers[i].get_action()
                if action is None:
                    action = np.zeros(7, dtype=np.float32)
                actions.append(action)

            next_obs, rewards, terminateds, truncateds, infos = vector_env.step(actions)

            for i in range(num_envs):
                if env_done[i]:
                    continue
                env_steps[i] += 1
                env_rewards[i] += rewards[i]

                if terminateds[i] or truncateds[i] or env_steps[i] >= max_steps:
                    env_done[i] = True
                    results.append({
                        "episode": total_episodes_run,
                        "success": infos[i].get("success", False),
                        "steps": env_steps[i],
                        "reward": float(env_rewards[i]),
                    })
                    total_episodes_run += 1

                    # Reset this env for next episode
                    if total_episodes_run < num_episodes:
                        new_seed = seed + total_episodes_run
                        obs_list[i], _ = vector_env.reset(seeds=[new_seed])
                        env_done[i] = False
                        env_steps[i] = 0
                        env_rewards[i] = 0.0
                        chunk_buffers[i] = _make_chunk_buffer(action_chunk_config)
                    else:
                        obs_list = next_obs  # keep reference alive
                else:
                    obs_list[i] = next_obs[i]
    else:
        # Sequential evaluation
        for ep in range(num_episodes):
            result = run_rollout(
                env=env,
                policy=policy,
                max_steps=max_steps,
                seed=seed + ep,
                render=render,
                render_fast=render_fast,
                action_chunk_config=action_chunk_config,
                save_metrics=(output_dir is not None),
                output_dir=output_dir,
                episode_idx=ep,
            )
            results.append(result)
            print(f"  Episode {ep + 1}/{num_episodes}: "
                  f"success={result['success']} steps={result['steps']} "
                  f"reward={result['reward']:.2f} fps={result['fps']:.0f}")

    t_end = time.perf_counter()

    successes = [r["success"] for r in results]
    success_rate = float(np.mean(successes))
    avg_steps = float(np.mean([r["steps"] for r in results]))
    avg_reward = float(np.mean([r["reward"] for r in results]))
    avg_fps = float(np.mean([r.get("fps", 0) for r in results]))

    summary = {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "avg_fps": avg_fps,
        "total_time_s": float(t_end - t_start),
        "num_episodes": num_episodes,
        "episodes": results,
    }

    if output_dir:
        with open(Path(output_dir) / "eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return summary


def _make_chunk_buffer(config: dict | None) -> Any:
    from robot_vla_mujoco.policies.action_decoder import ActionChunkBuffer

    cfg = config or {}
    return ActionChunkBuffer(
        prediction_horizon=cfg.get("prediction_horizon", 10),
        execution_horizon=cfg.get("execution_horizon", 5),
        overlap_strategy=cfg.get("overlap_strategy", "queue"),
        action_smoothing=cfg.get("action_smoothing", 0.0),
    )
