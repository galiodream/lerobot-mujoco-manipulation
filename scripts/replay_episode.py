#!/usr/bin/env python3
"""Replay a collected episode for visual inspection.

Usage:
    python scripts/replay_episode.py --dataset data/lerobot/omy_pick_place --episode 0
    python scripts/replay_episode.py --dataset data/lerobot/omy_pick_place --episode 0 --save-video
    python scripts/replay_episode.py --dataset data/lerobot/omy_pick_place --episode 0 --print-trajectory
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Replay a collected episode")
    parser.add_argument("--dataset", required=True, help="Path to LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--save-video", action="store_true", help="Save replay as MP4 video")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--fps", type=int, default=20, help="Replay FPS")
    parser.add_argument("--print-trajectory", action="store_true",
                        help="Only print state/action summary (no GUI/video)")
    args = parser.parse_args()

    from robot_vla_mujoco.datasets.replay import replay_episode

    replay_episode(
        dataset_root=args.dataset,
        episode_index=args.episode,
        fps=args.fps,
        save_video=args.save_video,
        output_video_path=args.output,
        print_trajectory=args.print_trajectory,
    )


if __name__ == "__main__":
    main()
