"""Episode metadata helpers."""

import json
from pathlib import Path


EPISODE_METADATA_FIELDS = [
    "episode_id",
    "task_id",
    "variant_id",
    "object_type",
    "object_initial_pose",
    "target_pose",
    "difficulty",
    "seed",
    "success",
    "env_config_hash",
]


def build_episode_metadata(
    episode_id: str,
    task_id: str,
    variant_id: str,
    object_type: str,
    object_initial_pose: list[float],
    target_pose: list[float],
    difficulty: str = "easy",
    seed: int = 0,
    success: bool = False,
    env_config_hash: str = "",
) -> dict:
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "variant_id": variant_id,
        "object_type": object_type,
        "object_initial_pose": object_initial_pose,
        "target_pose": target_pose,
        "difficulty": difficulty,
        "seed": seed,
        "success": success,
        "env_config_hash": env_config_hash,
    }


def read_episodes_metadata(dataset_root: str | Path) -> list[dict]:
    meta_path = Path(dataset_root) / "meta" / "episodes.jsonl"
    if not meta_path.exists():
        return []
    records = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_episodes(
    episodes: list[dict],
    keep_success_only: bool = True,
    max_episode_steps: int | None = None,
    difficulty: str | None = None,
) -> list[dict]:
    result = []
    for ep in episodes:
        if keep_success_only and not ep.get("success", False):
            continue
        if max_episode_steps is not None:
            if ep.get("num_frames", 0) > max_episode_steps:
                continue
        if difficulty is not None and ep.get("difficulty") != difficulty:
            continue
        result.append(ep)
    return result
