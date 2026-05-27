"""Configuration dataclasses and loading utilities.

Mirrors the configs/ YAML structure described in PROJECT_PLAN section 4.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Per-section configs
# ---------------------------------------------------------------------------

@dataclass
class VectorConfig:
    num_envs: int = 1
    backend: str = "sync"  # sync | async


@dataclass
class SuccessConfig:
    type: str = "pick_place"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraConfig:
    width: int = 256
    height: int = 256


@dataclass
class RandomizationConfig:
    object_pose: bool = True
    lighting: bool = False
    texture: bool = False


@dataclass
class EnvConfig:
    name: str = ""
    scene_xml: str = ""
    sim_hz: int = 500
    control_hz: int = 20
    vector: VectorConfig = field(default_factory=VectorConfig)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    success: SuccessConfig = field(default_factory=SuccessConfig)


@dataclass
class MujocoRobotConfig:
    scene_xml: str = ""
    end_effector_body: str = ""
    gripper_actuator: str = ""


@dataclass
class ControlConfig:
    action_mode: str = "joint_position"
    joint_names: list[str] = field(default_factory=list)
    gripper: dict[str, float] = field(default_factory=dict)


@dataclass
class LeRobotFeatures:
    images: dict[str, str] = field(default_factory=dict)
    state: str = "observation.state"
    action: str = "action"


@dataclass
class RobotConfig:
    name: str = ""
    arm_dof: int = 6
    gripper_dof: int = 1
    action_dim: int = 7
    state_dim: int = 7
    mujoco: MujocoRobotConfig = field(default_factory=MujocoRobotConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    lerobot_features: LeRobotFeatures = field(default_factory=LeRobotFeatures)


@dataclass
class ActionChunkConfig:
    prediction_horizon: int = 10
    execution_horizon: int = 5
    overlap_strategy: str = "queue"
    action_smoothing: float = 0.0


@dataclass
class PolicyConfig:
    type: str = ""
    path: str = ""
    device: str = "cuda"
    chunk_size: int = 10
    action_chunk: ActionChunkConfig = field(default_factory=ActionChunkConfig)


@dataclass
class CurationConfig:
    keep_success_only: bool = True
    max_episode_steps: int = 400
    balance_by: list[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    repo_id: str = ""
    root: str = ""
    fps: int = 20
    task: str = ""
    curation: CurationConfig = field(default_factory=CurationConfig)


@dataclass
class RolloutConfig:
    episodes: int = 20
    max_steps: int = 400
    save_video: bool = True
    output_dir: str = "outputs/rollouts"


@dataclass
class LocalLoggingConfig:
    jsonl: bool = True
    csv: bool = True
    videos: bool = True


@dataclass
class WandbConfig:
    enable: bool = False


@dataclass
class LoggingConfig:
    local: LocalLoggingConfig = field(default_factory=LocalLoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class ExperimentConfig:
    seed: int = 42
    env: EnvConfig = field(default_factory=EnvConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _dict_to_dataclass(cls: type, d: dict | None) -> Any:
    """Recursively convert a dict into a dataclass instance."""
    if d is None:
        return cls()
    import dataclasses
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for k, v in d.items():
        if k in field_types:
            ft = field_types[k]
            if dataclasses.is_dataclass(ft) and isinstance(v, dict):
                kwargs[k] = _dict_to_dataclass(ft, v)
            else:
                kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_dataclass(ExperimentConfig, raw)


def resolve_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Resolve includes / defaults in an experiment config (placeholder for now)."""
    return cfg
