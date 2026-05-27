"""Evaluation metrics for policy rollouts."""

from typing import Any

import numpy as np


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute aggregate metrics from a list of per-episode result dicts.

    Each result dict should have keys: success, steps, reward, time_s.
    """
    if not results:
        return {}

    successes = [r.get("success", False) for r in results]
    steps = [r.get("steps", 0) for r in results]
    rewards = [r.get("reward", 0.0) for r in results]
    times = [r.get("time_s", 0.0) for r in results]

    return {
        "success_rate": float(np.mean(successes)),
        "avg_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_time_s": float(np.mean(times)),
        "std_time_s": float(np.std(times)),
        "total_episodes": len(results),
        "total_successes": int(np.sum(successes)),
    }


def compute_action_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Compare predicted actions against ground truth.

    Args:
        predicted: [N, action_dim] predicted actions.
        ground_truth: [N, action_dim] ground-truth actions.

    Returns:
        Dict with l1_error, l2_error, smoothness.
    """
    diff = predicted - ground_truth
    return {
        "action_l1_error": float(np.mean(np.abs(diff))),
        "action_l2_error": float(np.mean(np.linalg.norm(diff, axis=1))),
        "action_smoothness": float(_smoothness(predicted)),
    }


def _smoothness(traj: np.ndarray) -> float:
    """Mean jerk — mean absolute second difference of action trajectory."""
    if len(traj) < 3:
        return 0.0
    second_diff = np.diff(traj, n=2, axis=0)
    return float(np.mean(np.abs(second_diff)))
