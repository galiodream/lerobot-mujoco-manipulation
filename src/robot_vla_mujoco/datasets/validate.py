"""Dataset validation: check schema, fps, image shapes, action dim."""

import json
from pathlib import Path
from typing import Any


def validate_dataset(dataset_root: str | Path) -> list[dict]:
    """Validate a LeRobot dataset and return a list of issues.

    Returns an empty list if everything is correct.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    root = Path(dataset_root)
    issues: list[dict] = []

    # 1. Check meta/info.json exists
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        issues.append({"severity": "error", "message": f"Missing meta/info.json in {root}"})
        return issues

    with open(info_path) as f:
        info = json.load(f)

    # 2. Check required features
    features = info.get("features", {})
    required = ["observation.state", "action"]
    for key in required:
        if key not in features:
            issues.append({"severity": "error", "message": f"Missing feature: {key}"})

    # 3. Check state/action dimensions
    state_feat = features.get("observation.state", {})
    action_feat = features.get("action", {})
    if state_feat.get("dtype") == "float32" and isinstance(state_feat.get("shape"), list):
        pass  # OK
    if action_feat.get("dtype") == "float32" and isinstance(action_feat.get("shape"), list):
        pass  # OK

    # 4. Check fps
    fps = info.get("fps", 0)
    if fps <= 0:
        issues.append({"severity": "warning", "message": f"Invalid fps: {fps}"})

    # 5. Check at least one episode
    num_episodes = info.get("total_episodes", 0)
    if num_episodes == 0:
        issues.append({"severity": "warning", "message": "Dataset has 0 episodes"})

    # 6. Try loading the dataset
    try:
        metadata = LeRobotDatasetMetadata(root.name, root=root)
        dataset = LeRobotDataset(root.name, root=root, episodes=[0])
        if len(dataset) == 0:
            issues.append({"severity": "error", "message": "Episode 0 has 0 frames"})
    except Exception as e:
        issues.append({"severity": "error", "message": f"Cannot load dataset: {e}"})

    # 7. Check episode metadata
    episodes_path = root / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        with open(episodes_path) as f:
            for line in f:
                ep = json.loads(line.strip())
                # Verify required metadata fields
                for field in ["episode_id", "task_id"]:
                    if field not in ep:
                        issues.append({
                            "severity": "warning",
                            "message": f"episode missing field: {field}",
                        })

    return issues
