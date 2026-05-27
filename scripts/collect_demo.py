#!/usr/bin/env python3
"""Collect demonstration episodes using scripted oracle or keyboard teleop.

Usage:
    # Scripted oracle (recommended for bulk collection)
    python scripts/collect_demo.py --mode oracle --episodes 10

    # Keyboard teleop (debug)
    python scripts/collect_demo.py --mode teleop

    # With viewer on (debug)
    MODEL_SIM_VIEWER=1 python scripts/collect_demo.py --mode oracle --episodes 3

    # Headless fast collection (recommended for bulk)
    MODEL_SIM_VIEWER=0 MUJOCO_GL=egl python scripts/collect_demo.py --mode oracle --episodes 100
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
    parser = argparse.ArgumentParser(description="Collect demonstration episodes")
    parser.add_argument("--mode", default="oracle", choices=["oracle", "teleop"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--dataset-name", default="omy_pick_place")
    parser.add_argument("--dataset-root", default="data/lerobot/omy_pick_place")
    parser.add_argument("--task", default="pick up the mug and place it on the plate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--success-required", action="store_true", default=True,
                        help="Only save successful episodes")
    args = parser.parse_args()

    use_viewer = not args.no_viewer and _env_bool("MODEL_SIM_VIEWER", True)
    xml_path = os.path.abspath("assets/mujoco/tasks/pick_place_mug.xml")

    from robot_vla_mujoco.envs.mujoco_env import MujocoManipulationEnv
    from robot_vla_mujoco.datasets.lerobot_writer import DatasetCollector

    print(f"Initializing environment (viewer={use_viewer})...")
    env = MujocoManipulationEnv(
        xml_path=xml_path,
        action_type="joint_angle",
        state_type="joint_angle",
        seed=args.seed,
        initialize_viewer=use_viewer,
    )

    collector = DatasetCollector(
        repo_id=args.dataset_name,
        root=args.dataset_root,
        fps=args.fps,
    )

    if use_viewer:
        for _ in range(15):
            env.render(fast=False)
            time.sleep(1.0 / 60.0)

    if args.mode == "teleop":
        from robot_vla_mujoco.datasets.collect_teleop import KeyboardTeleop
        teleop = KeyboardTeleop(env)
        _collect_teleop(env, teleop, collector, args)
    else:
        _collect_oracle(env, collector, args)

    env.close()
    print(f"\nDone. Collected {collector.num_episodes} episodes in {args.dataset_root}")


def _collect_oracle(env, collector, args):
    from robot_vla_mujoco.datasets.collect_scripted import PickPlaceOracle

    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
        obs, info = env.reset(seed=args.seed + ep)
        oracle = PickPlaceOracle(env)
        oracle.reset()

        task = obs.get("task", args.task)
        collector.start_episode(task=task)

        success = False
        for step in range(args.max_steps):
            action = oracle.get_action()
            obs, reward, terminated, truncated, info = env.step(action)
            collector.add_frame(obs, action)

            if env.viewer_enabled:
                env.render(idx=ep)

            if terminated:
                success = True
                print(f"  Step {step}: SUCCESS (stage={oracle.stage})")
                break
            if truncated:
                print(f"  Step {step}: TRUNCATED (stage={oracle.stage})")
                break

            if oracle.done and not terminated:
                print(f"  Step {step}: Oracle finished, checking success...")

        # Check success condition
        if not success:
            success = env.is_success()

        collector.save_episode(metadata={
            "episode_id": f"{ep:06d}",
            "task_id": "pick_place_mug",
            "variant_id": f"mug_red_easy",
            "object_type": "mug",
            "difficulty": "easy",
            "seed": args.seed + ep,
            "success": success,
            "num_frames": step + 1,
        })

        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep}: {status}")


def _collect_teleop(env, teleop, collector, args):
    print("Keyboard teleop controls:")
    print("  W/S: forward/back  A/D: left/right  R/F: up/down")
    print("  Q/E: roll  Arrow keys: pitch/yaw")
    print("  SPACE: toggle gripper  Z: reset episode  ESC: quit")

    ep = 0
    obs, info = env.reset()
    collector.start_episode(task=args.task)
    step = 0

    while True:
        action, reset_ep = teleop.get_action()
        if reset_ep:
            collector.save_episode(metadata={
                "episode_id": f"{ep:06d}",
                "task_id": "pick_place_mug",
                "variant_id": "teleop",
                "object_type": "mug",
                "difficulty": "teleop",
                "seed": args.seed,
                "success": False,
            })
            ep += 1
            print(f"Episode {ep} saved ({step} frames). Starting new episode...")
            obs, info = env.reset()
            collector.start_episode(task=args.task)
            step = 0
            if ep >= args.episodes:
                break
            continue

        obs, reward, terminated, truncated, info = env.step(action)
        collector.add_frame(obs, action)

        if env.viewer_enabled:
            env.render(teleop=True, idx=ep)

        step += 1
        if step >= args.max_steps:
            step = 0
            collector.save_episode(metadata={
                "episode_id": f"{ep:06d}",
                "task_id": "pick_place_mug",
                "variant_id": "teleop",
                "object_type": "mug",
                "difficulty": "teleop",
                "seed": args.seed,
                "success": False,
            })
            ep += 1
            print(f"Episode {ep} truncated. Starting new episode...")
            obs, info = env.reset()
            collector.start_episode(task=args.task)
            if ep >= args.episodes:
                break


if __name__ == "__main__":
    main()
