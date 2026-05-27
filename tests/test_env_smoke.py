"""Smoke tests for the environment: reset, step, render."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def xml_path():
    p = Path(__file__).resolve().parent.parent / "assets" / "mujoco" / "tasks" / "pick_place_mug.xml"
    if not p.exists():
        pytest.skip("Task scene XML not found")
    return str(p)


def test_env_creation_headless(xml_path):
    """Environment can be created in headless mode."""
    from robot_vla_mujoco.envs.mujoco_env import MujocoManipulationEnv

    env = MujocoManipulationEnv(xml_path=xml_path, initialize_viewer=False)
    assert env is not None
    env.close()


def test_env_reset(xml_path):
    """reset() returns observation and info dicts."""
    from robot_vla_mujoco.envs.mujoco_env import MujocoManipulationEnv

    env = MujocoManipulationEnv(xml_path=xml_path, initialize_viewer=False)
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    assert "observation.state" in obs
    obs_state = obs["observation.state"]
    assert isinstance(obs_state, np.ndarray)
    assert obs_state.shape == (7,)
    env.close()


def test_env_step(xml_path):
    """step() returns (obs, reward, terminated, truncated, info)."""
    from robot_vla_mujoco.envs.mujoco_env import MujocoManipulationEnv

    env = MujocoManipulationEnv(xml_path=xml_path, initialize_viewer=False)
    env.reset(seed=0)

    action = np.zeros(7, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "step_count" in info
    env.close()


def test_dummy_rollout(xml_path):
    """A full episode with dummy policy completes without crashes."""
    from robot_vla_mujoco.envs.mujoco_env import MujocoManipulationEnv
    from robot_vla_mujoco.policies.base import DummyPolicy

    env = MujocoManipulationEnv(xml_path=xml_path, initialize_viewer=False)
    policy = DummyPolicy(action_dim=7)

    obs, _ = env.reset(seed=0)
    policy.reset(obs.get("task", ""))

    for _ in range(50):
        action = policy.predict_action_trajectory(obs)
        if action.ndim == 2:
            action = action[0]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # No assertion needed on success — just verifying no crashes in headless loop
    env.close()


def test_image_shapes(xml_path):
    """Camera images have correct shapes."""
    from robot_vla_mujoco.envs.mujoco_env import MujocoManipulationEnv

    env = MujocoManipulationEnv(xml_path=xml_path, initialize_viewer=False)
    env.reset(seed=0)

    obs = env.get_observation()
    front = obs.get("observation.images.front")
    wrist = obs.get("observation.images.wrist")
    if front is not None:
        assert front.ndim == 3
        assert front.shape[2] == 3  # RGB
    if wrist is not None:
        assert wrist.ndim == 3
        assert wrist.shape[2] == 3

    env.close()


def test_success_condition_registry():
    """Success condition registry has pick_place."""
    from robot_vla_mujoco.envs.success_conditions import SUCCESS_CONDITION_REGISTRY

    assert "pick_place" in SUCCESS_CONDITION_REGISTRY.keys()
