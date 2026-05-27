#!/usr/bin/env python3
"""Evaluate a policy over multiple episodes and report metrics.

Usage:
    # Dummy policy (smoke test)
    python scripts/eval_policy.py --policy dummy --episodes 20

    # SmolVLA / pi0 (requires GPU and checkpoint)
    python scripts/eval_policy.py --policy smolvla --checkpoint lerobot/smolvla_base --episodes 20

    # With viewer (debug)
    MODEL_SIM_VIEWER=1 python scripts/eval_policy.py --policy dummy --episodes 3

    # Headless multi-env
    MODEL_SIM_VIEWER=0 MUJOCO_GL=egl python scripts/eval_policy.py --policy dummy --episodes 20 --num-envs 4
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


def _make_env_fn(xml_path: str, seed: int, use_viewer: bool = False):
    def _fn():
        os.environ["MODEL_SIM_VIEWER"] = "0"  # worker always headless
        from robot_vla_mujoco.envs.mujoco_env import MujocoManipulationEnv
        return MujocoManipulationEnv(
            xml_path=xml_path,
            action_type="joint_angle",
            state_type="joint_angle",
            seed=seed,
            initialize_viewer=False,
        )
    return _fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy")
    parser.add_argument("--policy", default="dummy", choices=["dummy", "smolvla", "pi0", "pi05"])
    parser.add_argument("--checkpoint", default="", help="Policy checkpoint path")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs")
    parser.add_argument("--device", default="cuda" if _env_bool("MODEL_SIM_HAS_CUDA", False) else "cpu")
    parser.add_argument("--no-metrics", action="store_true", help="Skip saving metrics")
    args = parser.parse_args()

    use_viewer = not args.no_viewer and _env_bool("MODEL_SIM_VIEWER", True)
    xml_path = os.path.abspath("assets/mujoco/tasks/pick_place_mug.xml")

    print(f"Evaluation config:")
    print(f"  policy: {args.policy}")
    print(f"  episodes: {args.episodes}")
    print(f"  num_envs: {args.num_envs}")
    print(f"  viewer: {use_viewer}")
    print(f"  device: {args.device}")

    # Create policy
    if args.policy == "dummy":
        from robot_vla_mujoco.policies.base import DummyPolicy
        policy = DummyPolicy(action_dim=7)
    elif args.policy == "smolvla":
        from robot_vla_mujoco.policies.smolvla_adapter import SmolVLAAdapter
        ckpt = args.checkpoint or "lerobot/smolvla_base"
        policy = SmolVLAAdapter(checkpoint_path=ckpt, device=args.device)
    elif args.policy == "pi0":
        from robot_vla_mujoco.policies.pi0_adapter import PI0Adapter
        ckpt = args.checkpoint or "lerobot/pi0"
        policy = PI0Adapter(checkpoint_path=ckpt, device=args.device)
    elif args.policy == "pi05":
        from robot_vla_mujoco.policies.pi05_adapter import PI05Adapter
        ckpt = args.checkpoint or "lerobot/pi05"
        policy = PI05Adapter(checkpoint_path=ckpt, device=args.device)
    else:
        print(f"Unknown policy: {args.policy}")
        sys.exit(1)

    # Vector env for parallel eval
    vector_env = None
    if args.num_envs > 1:
        from robot_vla_mujoco.envs.vector_env import SyncVectorEnv
        env_fns = [_make_env_fn(xml_path, args.seed + i) for i in range(args.num_envs)]
        vector_env = SyncVectorEnv(env_fns)
        print(f"  Created {args.num_envs} parallel envs")

    if not use_viewer:
        os.environ["MODEL_SIM_VIEWER"] = "0"

    from robot_vla_mujoco.envs.mujoco_env import MujocoManipulationEnv
    from robot_vla_mujoco.inference.evaluate_sim import evaluate_policy

    env = None
    if vector_env is None:
        env = MujocoManipulationEnv(
            xml_path=xml_path,
            action_type="joint_angle",
            state_type="joint_angle",
            seed=args.seed,
            initialize_viewer=use_viewer,
        )

    print("\nStarting evaluation...")
    t_start = time.perf_counter()

    summary = evaluate_policy(
        env=env,
        policy=policy,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        render=use_viewer,
        output_dir=args.output_dir if not args.no_metrics else None,
        vector_env=vector_env,
    )

    t_elapsed = time.perf_counter() - t_start

    print(f"\n{'='*50}")
    print(f"Evaluation complete ({t_elapsed:.1f}s)")
    print(f"  Success rate: {summary['success_rate']:.2%}")
    print(f"  Avg steps: {summary['avg_steps']:.1f}")
    print(f"  Avg reward: {summary['avg_reward']:.2f}")
    print(f"  Avg FPS: {summary['avg_fps']:.1f}")
    print(f"{'='*50}")

    if env is not None:
        env.close()
    if vector_env is not None:
        vector_env.close()
    policy.close()


if __name__ == "__main__":
    main()
