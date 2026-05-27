#!/usr/bin/env python3
"""Validate all MuJoCo XML assets in the project.

Usage:
    python scripts/validate_assets.py [--asset-root assets/mujoco]
"""

import argparse
import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robot_vla_mujoco.assets.mjcf_loader import validate_asset_root


def main():
    parser = argparse.ArgumentParser(description="Validate MuJoCo XML assets")
    parser.add_argument(
        "--asset-root", default="assets/mujoco",
        help="Root directory for MuJoCo assets (default: assets/mujoco)"
    )
    parser.add_argument(
        "--scene-only", action="store_true",
        help="Only validate the task scene XML"
    )
    parser.add_argument(
        "--task", default="tasks/pick_place_mug.xml",
        help="Task scene to validate when --scene-only (default: tasks/pick_place_mug.xml)"
    )
    args = parser.parse_args()

    asset_root = Path(args.asset_root)

    if args.scene_only:
        from robot_vla_mujoco.assets.mjcf_loader import validate_scene_xml
        xml_path = asset_root / args.task
        print(f"Validating scene: {xml_path}")
        info = validate_scene_xml(xml_path)
        print(f"  model: {info.get('model_name', '?')}")
        print(f"  bodies: {info['n_body']}, joints: {info['n_joint']}, geoms: {info['n_geom']}")
        print(f"  cameras: {info['n_cam']}, controls: {info['n_ctrl']}")
        print(f"  dt: {info['dt']}, integrator: {info['integrator']}")
        body_names = info.get("body_names", [])
        print(f"  body names: {body_names}")
    else:
        results = validate_asset_root(asset_root)
        n_ok = sum(1 for r in results if r["status"] == "OK")
        n_fail = sum(1 for r in results if r["status"] == "FAIL")
        for r in results:
            if r["status"] == "OK":
                print(f'  OK  {r["path"]}  model={r.get("model_name", "?")} '
                      f'bodies={r.get("n_body", "?")} joints={r.get("n_joint", "?")} '
                      f'geoms={r.get("n_geom", "?")} cams={r.get("n_cam", "?")}')
            else:
                print(f'  FAIL {r["path"]}  error={r.get("error", "?")}')
        print(f"\n{n_ok} passed, {n_fail} failed out of {len(results)} XML files.")


if __name__ == "__main__":
    main()
