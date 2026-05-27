#!/usr/bin/env python3
"""Validate a LeRobot dataset schema and integrity.

Usage:
    python scripts/validate_dataset.py --dataset data/lerobot/omy_pick_place
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Validate LeRobot dataset")
    parser.add_argument("--dataset", required=True, help="Path to LeRobot dataset root")
    args = parser.parse_args()

    from robot_vla_mujoco.datasets.validate import validate_dataset

    issues = validate_dataset(args.dataset)
    if not issues:
        print("Dataset validation PASSED.")
        return

    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]

    for w in warnings:
        print(f"  WARNING: {w['message']}")
    for e in errors:
        print(f"  ERROR: {e['message']}")

    print(f"\n{len(errors)} errors, {len(warnings)} warnings.")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
