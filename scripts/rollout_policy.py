#!/usr/bin/env python3
"""Run a policy rollout in the MuJoCo environment.

Usage:
    # Dummy policy (smoke test)
    python scripts/rollout_policy.py --config configs/experiments/omy_dummy_pick_place.yaml --policy dummy

    # With viewer enabled (debug)
    MODEL_SIM_VIEWER=1 python scripts/rollout_policy.py --config configs/experiments/omy_dummy_pick_place.yaml --policy dummy

    # Headless EGL render
    MODEL_SIM_VIEWER=0 MUJOCO_GL=egl python scripts/rollout_policy.py --config configs/experiments/omy_dummy_pick_place.yaml --policy dummy
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in {"0", "false", "no", "off"}


def main():
    parser = argparse.ArgumentParser(description="Run policy rollout")
    parser.add_argument("--config", default="configs/experiments/omy_dummy_pick_place.yaml")
    parser.add_argument("--policy", default="dummy", help="Policy type: dummy, smolvla, pi0, pi05")
    parser.add_argument("--scene", default="omy_pick_place",
                        choices=["omy_pick_place", "ur3e_ag95_pick_place"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--no-viewer", action="store_true", help="Force disable viewer")
    parser.add_argument("--render-fast", action="store_true", help="Fast render mode (no overlays)")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    args = parser.parse_args()

    from robot_vla_mujoco.envs.mujoco_env import MujocoManipulationEnv, resolve_scene
    from robot_vla_mujoco.policies.base import DummyPolicy

    scene = resolve_scene(args.scene)
    xml_path = os.path.abspath(scene["scene_xml"])
    if not os.path.exists(xml_path):
        print(f"ERROR: Scene XML not found: {xml_path}")
        sys.exit(1)

    use_viewer = not args.no_viewer and _env_bool("MODEL_SIM_VIEWER", True)

    print(f"Initializing environment...")
    print(f"  Scene: {args.scene}")
    print(f"  XML: {xml_path}")
    print(f"  Viewer: {use_viewer}")

    env = MujocoManipulationEnv(
        xml_path=xml_path,
        action_type="joint_angle",
        state_type="joint_angle",
        seed=42,
        initialize_viewer=use_viewer,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        robot_profile=scene["robot_profile"],
        success_config=scene.get("success_params"),
    )

    policy = DummyPolicy(action_dim=7)
    policy.reset("pick up the mug and place it on the plate")

    if use_viewer:
        for _ in range(15):
            env.render(fast=False)
            time.sleep(1.0 / 60.0)

    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
        obs, info = env.reset()
        policy.reset(obs.get("task", ""))
        episode_reward = 0.0

        for step in range(args.max_steps):
            action = policy.predict_action_trajectory(obs)
            if action.ndim == 2:
                action = action[0]  # take first step of trajectory

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if use_viewer:
                env.render(fast=args.render_fast)
                time.sleep(1.0 / 60.0)

            if terminated:
                print(f"  Step {step}: SUCCESS! reward={episode_reward:.1f}")
                break
            if truncated:
                print(f"  Step {step}: truncated (max steps). reward={episode_reward:.1f}")
                break

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
