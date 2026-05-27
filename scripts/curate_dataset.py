#!/usr/bin/env python3
"""Curate a LeRobot dataset: filter failures, truncate long episodes.

Usage:
    python scripts/curate_dataset.py --dataset data/lerobot/omy_pick_place --keep-success-only
    python scripts/curate_dataset.py --dataset data/lerobot/omy_pick_place --max-steps 400
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Curate a LeRobot dataset")
    parser.add_argument("--dataset", required=True, help="Path to LeRobot dataset root")
    parser.add_argument("--keep-success-only", action="store_true", default=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output manifest JSON path")
    args = parser.parse_args()

    from robot_vla_mujoco.datasets.curate import curate_dataset

    curated = curate_dataset(
        dataset_root=args.dataset,
        keep_success_only=args.keep_success_only,
        max_episode_steps=args.max_steps,
        output_manifest=args.output,
    )

    print(f"Curated {len(curated)} episodes from {args.dataset}")
    if args.output:
        print(f"Manifest written to: {args.output}")


if __name__ == "__main__":
    main()
