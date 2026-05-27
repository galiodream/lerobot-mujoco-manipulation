#!/usr/bin/env python3
"""Train or fine-tune a policy using LeRobot.

Usage:
    # SmolVLA fine-tune
    python scripts/train_policy.py --config configs/experiments/omy_smolvla_pick_place.yaml

    # Dry-run (validate config only)
    python scripts/train_policy.py --config configs/experiments/omy_smolvla_pick_place.yaml --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Train/fine-tune a policy")
    parser.add_argument("--config", default="configs/experiments/omy_smolvla_pick_place.yaml")
    parser.add_argument("--policy", default=None, help="Override policy type")
    parser.add_argument("--output-dir", default="outputs/train")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only, no training")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from robot_vla_mujoco.common.config import load_config

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    cfg = load_config(config_path)
    print(f"Loaded config: {args.config}")
    print(f"  Robot: {cfg.robot.profile}")
    print(f"  Env: {cfg.env.name}")
    print(f"  Policy: {cfg.policy.type}")
    print(f"  Dataset: {cfg.dataset.repo_id}")
    print(f"  Seed: {cfg.seed}")

    if args.dry_run:
        print("\n[Dry-run] Config validated. No training executed.")
        return

    # Check dataset exists
    dataset_root = Path(cfg.dataset.root)
    if not dataset_root.exists() or not (dataset_root / "meta" / "info.json").exists():
        print(f"\nERROR: Dataset not found at {dataset_root}")
        print("Run scripts/collect_demo.py first to collect training data.")
        sys.exit(1)

    policy_type = args.policy or cfg.policy.type
    print(f"\nTraining {policy_type} on {cfg.dataset.repo_id}...")

    # LeRobot training command
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build lerobot-train command
    cmd_parts = [
        "lerobot-train",
        f"--policy={policy_type}",
        f"--dataset.repo_id={cfg.dataset.repo_id}",
        f"--dataset.root={cfg.dataset.root}",
        f"--output_dir={output_dir}",
        f"--seed={cfg.seed}",
        f"--batch_size=8",
        f"--steps=10000",
    ]

    if args.resume:
        cmd_parts.append("--resume")

    cmd = " ".join(cmd_parts)
    print(f"Running: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    main()
