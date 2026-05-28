"""LeRobot dataset writer: creates and populates datasets in LeRobot v2/v3 format.

Schema (from PROJECT_PLAN):
    observation.images.front  — RGB image (256, 256, 3)
    observation.images.wrist — RGB image (256, 256, 3)
    observation.state        — float32 (7,)
    action                   — float32 (7,)
    task                     — string
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def _make_config_hash(env_config: dict | None = None) -> str:
    raw = json.dumps(env_config or {}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def build_lerobot_features(
    front_shape: tuple[int, int, int] = (256, 256, 3),
    wrist_shape: tuple[int, int, int] = (256, 256, 3),
    state_dim: int = 7,
    action_dim: int = 7,
) -> dict:
    return {
        "observation.images.front": {
            "dtype": "image",
            "shape": list(front_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": list(wrist_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
    }


def ensure_dataset(
    repo_id: str,
    root: str | Path,
    fps: int = 20,
    features: dict | None = None,
) -> Any:
    """Create or open a LeRobot dataset. Returns a LeRobotDataset instance."""
    import shutil
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    meta_dir = root / "meta"

    # Try to re-open existing dataset
    if meta_dir.exists() and (meta_dir / "info.json").exists():
        try:
            return LeRobotDataset(repo_id=repo_id, root=root)
        except Exception:
            # Dataset is incomplete or corrupted — delete and recreate
            shutil.rmtree(root)

    # Clean up any stale directory
    if root.exists():
        shutil.rmtree(root)

    if features is None:
        features = build_lerobot_features()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        use_videos=False,
    )
    return dataset


def write_episode_metadata(
    dataset: Any,
    episode_index: int,
    metadata: dict[str, Any],
) -> None:
    """Write episode-level metadata to the dataset's meta directory."""
    meta_path = Path(dataset.root) / "meta" / "episodes.jsonl"
    record = {"episode_index": episode_index, **metadata}
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "a") as f:
        f.write(json.dumps(record) + "\n")


class DatasetCollector:
    """Convenience wrapper for collecting episodes into a LeRobot dataset."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        fps: int = 20,
        features: dict | None = None,
        env_config: dict | None = None,
    ):
        self._repo_id = repo_id
        self._root = Path(root)
        self._fps = fps
        self._dataset = ensure_dataset(repo_id=repo_id, root=root, fps=fps, features=features)
        self._env_config = env_config
        self._config_hash = _make_config_hash(env_config)

    @property
    def dataset(self):
        return self._dataset

    def start_episode(self, task: str = "") -> None:
        self._task = task
        self._frames: list[dict] = []
        self._t_start = time.time()

    def add_frame(self, obs: dict, action: np.ndarray) -> None:
        import cv2

        timestamp = time.time() - self._t_start

        front = obs.get("observation.images.front")
        wrist = obs.get("observation.images.wrist")

        # Resize images to 256x256 LeRobot format
        target_size = (256, 256)
        if front is not None and front.shape[:2] != target_size:
            front = cv2.resize(front, target_size, interpolation=cv2.INTER_AREA)
        if wrist is not None and wrist.shape[:2] != target_size:
            wrist = cv2.resize(wrist, target_size, interpolation=cv2.INTER_AREA)

        frame = {
            "observation.images.front": front,
            "observation.images.wrist": wrist,
            "observation.state": obs.get("observation.state"),
            "action": action.astype(np.float32),
        }
        self._dataset.add_frame(frame, task=self._task)
        self._frames.append(frame)

    def save_episode(self, metadata: dict | None = None) -> None:
        self._dataset.save_episode()
        if metadata:
            write_episode_metadata(self._dataset, self._dataset.num_episodes - 1, metadata)

    @property
    def num_episodes(self) -> int:
        return self._dataset.num_episodes

    @property
    def num_frames(self) -> int:
        return self._dataset.num_frames
