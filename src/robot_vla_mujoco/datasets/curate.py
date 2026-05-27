"""Dataset curation: filter, truncate, and balance episode manifests.

Produces filtered episode indices without rewriting the raw dataset.
"""

import json
from pathlib import Path


def curate_dataset(
    dataset_root: str | Path,
    keep_success_only: bool = True,
    max_episode_steps: int | None = None,
    balance_by: list[str] | None = None,
    output_manifest: str | Path | None = None,
) -> list[dict]:
    """Filter episodes from a LeRobot dataset and produce a curated manifest.

    Args:
        dataset_root: Path to the LeRobot dataset root.
        keep_success_only: Drop episodes where success != True.
        max_episode_steps: Drop episodes longer than this many frames.
        balance_by: Not implemented yet (placeholder for future balancing).
        output_manifest: If provided, write curated episode list to this JSON file.

    Returns:
        List of curated episode metadata dicts.
    """
    from robot_vla_mujoco.datasets.metadata import filter_episodes, read_episodes_metadata

    episodes = read_episodes_metadata(dataset_root)
    curated = filter_episodes(
        episodes,
        keep_success_only=keep_success_only,
        max_episode_steps=max_episode_steps,
    )

    if output_manifest is not None:
        output_manifest = Path(output_manifest)
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(output_manifest, "w") as f:
            json.dump(curated, f, indent=2)

    return curated
