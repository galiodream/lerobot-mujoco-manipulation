"""Episode replay: visualize collected episodes frame-by-frame."""

import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_episode_data(dataset_root: str | Path, episode_index: int) -> list[dict]:
    """Load all frames for a specific episode from a LeRobot dataset.

    Returns a list of frame dicts with keys like:
        observation.images.front, observation.images.wrist,
        observation.state, action, task, timestamp, frame_index.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    root = Path(dataset_root)
    metadata = LeRobotDatasetMetadata(root.name, root=root)
    dataset = LeRobotDataset(root.name, root=root, episodes=[episode_index])

    frames = []
    for i in range(len(dataset)):
        item = dataset[i]
        frames.append(item)
    return frames


def replay_episode(
    dataset_root: str | Path,
    episode_index: int = 0,
    fps: int = 20,
    save_video: bool = False,
    output_video_path: str | Path | None = None,
    print_trajectory: bool = False,
) -> None:
    """Replay a single episode, optionally saving a video.

    Args:
        dataset_root: Path to LeRobot dataset root.
        episode_index: Which episode to replay.
        fps: Frame rate for playback / video.
        save_video: If True, write an MP4 video.
        output_video_path: Output path for the video (auto-generated if None).
        print_trajectory: If True, only print state/action summary (no video).
    """
    frames = load_episode_data(dataset_root, episode_index)
    if not frames:
        print(f"No frames found for episode {episode_index}")
        return

    print(f"Episode {episode_index}: {len(frames)} frames")

    if print_trajectory:
        _print_trajectory_summary(frames)
        return

    video_writer = None
    frame_delay = 1.0 / fps

    if save_video:
        if output_video_path is None:
            output_video_path = Path(dataset_root) / f"replay_ep{episode_index:06d}.mp4"
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine output size from first front image
        first = frames[0]
        sample_img = _get_image(first, "observation.images.front")
        if sample_img is None:
            sample_img = _get_image(first, "observation.image")
        if sample_img is not None:
            h, w = sample_img.shape[:2]
            out_w, out_h = w * 2, h  # side-by-side: front + wrist
        else:
            out_w, out_h = 1280, 480

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (out_w, out_h))

    print("Press Ctrl+C to stop replay.")
    try:
        for i, frame in enumerate(frames):
            front = _get_image(frame, "observation.images.front") or _get_image(frame, "observation.image")
            wrist = _get_image(frame, "observation.images.wrist") or _get_image(frame, "observation.wrist_image")

            state = frame.get("observation.state")
            action = frame.get("action")

            if front is not None and wrist is not None:
                if front.shape[:2] != wrist.shape[:2]:
                    wrist = cv2.resize(wrist, (front.shape[1], front.shape[0]))
                combined = np.hstack([front, wrist])
            elif front is not None:
                combined = front
            else:
                combined = np.zeros((480, 640, 3), dtype=np.uint8)

            # Overlay frame number and state info
            text_lines = [f"Frame {i}/{len(frames)}"]
            if state is not None:
                text_lines.append(f"state: {np.round(state, 2)}")
            if action is not None:
                text_lines.append(f"action: {np.round(action, 2)}")

            y0 = 30
            for line in text_lines:
                cv2.putText(combined, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y0 += 20

            if video_writer is not None:
                video_writer.write(combined)
            else:
                cv2.imshow("Episode Replay (ESC to exit)", combined)
                key = cv2.waitKey(int(frame_delay * 1000)) & 0xFF
                if key == 27:  # ESC
                    break

            time.sleep(frame_delay)
    except KeyboardInterrupt:
        pass

    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {output_video_path}")
    else:
        cv2.destroyAllWindows()


def _get_image(frame: dict, key: str) -> np.ndarray | None:
    img = frame.get(key)
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim == 3 and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))  # C,H,W → H,W,C
        return img
    return None


def _print_trajectory_summary(frames: list[dict]) -> None:
    print(f"{'frame':>6s}  {'state_min':>12s}  {'state_max':>12s}  {'action_min':>12s}  {'action_max':>12s}")
    for i, frame in enumerate(frames):
        state = frame.get("observation.state")
        action = frame.get("action")
        s_str = f"{float(np.min(state)):.4f}~{float(np.max(state)):.4f}" if state is not None else "N/A"
        a_str = f"{float(np.min(action)):.4f}~{float(np.max(action)):.4f}" if action is not None else "N/A"
        if i % max(1, len(frames) // 20) == 0:
            print(f"{i:6d}  {s_str:>12s}  {a_str:>12s}")
