#!/usr/bin/env python3
"""Collect demonstration episodes using automatic oracle control or manual teleop.

Usage:
    # Automatic collection with scripted oracle (recommended for bulk collection)
    python scripts/collect_demo.py --mode auto --episodes 10

    # Manual collection with keyboard teleop
    python scripts/collect_demo.py --mode manual

    # Manual collection with a clickable control panel
    python scripts/collect_demo.py --mode manual_gui

    # With viewer on (debug)
    MODEL_SIM_VIEWER=1 python scripts/collect_demo.py --mode auto --episodes 3

    # Headless fast collection (recommended for bulk)
    MODEL_SIM_VIEWER=0 MUJOCO_GL=egl python scripts/collect_demo.py --mode auto --episodes 100
"""

import argparse
import os
import sys
import time
import tkinter as tk
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
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "manual", "manual_gui", "oracle", "teleop"],
        help="Collection mode: auto/oracle, manual keyboard teleop, or manual_gui button control",
    )
    parser.add_argument("--scene", default="omy_pick_place",
                        choices=["omy_pick_place", "ur3e_ag95_pick_place"],
                        help="Scene to use (determines robot + task XML)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--dataset-name", default=None, help="Override dataset name")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root")
    parser.add_argument("--task", default=None, help="Override task description")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    from robot_vla_mujoco.envs.mujoco_env import MujocoManipulationEnv, resolve_scene
    from robot_vla_mujoco.datasets.lerobot_writer import DatasetCollector

    scene = resolve_scene(args.scene)
    xml_path = os.path.abspath(scene["scene_xml"])
    robot_profile = scene["robot_profile"]
    dataset_name = args.dataset_name or scene["dataset_name"]
    dataset_root = args.dataset_root or scene["dataset_root"]
    task = args.task or scene["task"]
    mode = _normalize_mode(args.mode)

    use_viewer = not args.no_viewer and _env_bool("MODEL_SIM_VIEWER", True)
    if mode in {"manual", "manual_gui"} and not use_viewer:
        raise ValueError("Manual collection requires the MuJoCo viewer. Remove --no-viewer or set MODEL_SIM_VIEWER=1.")

    print(f"Scene: {args.scene}")
    print(f"Robot profile: {robot_profile}")
    print(f"Mode: {mode}")
    # Manual control uses EE pose delta; automatic collection uses joint angle targets.
    action_type = "eef_pose" if mode in {"manual", "manual_gui"} else "joint_angle"
    print(f"Preparing environment (viewer={use_viewer})...")
    env = MujocoManipulationEnv(
        xml_path=xml_path,
        action_type=action_type,
        state_type="joint_angle",
        seed=args.seed,
        initialize_viewer=use_viewer,
        robot_profile=robot_profile,
        success_config=scene.get("success_params"),
    )
    print("Environment setup complete.")

    collector = DatasetCollector(
        repo_id=dataset_name,
        root=dataset_root,
        fps=args.fps,
    )

    if use_viewer:
        print("Viewer ready. Warming up initial frames...")
        for _ in range(15):
            env.render(fast=False)
            time.sleep(1.0 / 60.0)

    if mode == "manual":
        _collect_manual(env, collector, task, args)
    elif mode == "manual_gui":
        _collect_manual_gui(env, collector, task, args)
    else:
        _collect_auto(env, collector, task, args)

    env.close()
    print(f"\nDone. Collected {collector.num_episodes} episodes in {dataset_root}")


def _normalize_mode(mode: str) -> str:
    return {"oracle": "auto", "teleop": "manual"}.get(mode, mode)


class _ButtonTeleopPanel:
    def __init__(self):
        try:
            self._root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError(
                "Failed to create the manual control panel window. "
                "Please make sure a desktop display is available for Tk."
            ) from exc
        self._root.title("MuJoCo Manual Control")
        self._root.geometry("460x520")
        self._root.minsize(420, 480)
        self._root.resizable(True, True)

        self._dpos_scale = 0.007
        self._drot_scale = 0.03
        self._gripper_open = True
        self._pending_action = np.zeros(7, dtype=np.float32)
        self._save_requested = False
        self._quit_requested = False

        self._build_ui()
        self._root.protocol("WM_DELETE_WINDOW", self._request_quit)

    def _queue_motion(self, dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0):
        self._pending_action = np.array([dx, dy, dz, droll, dpitch, dyaw, float(self._gripper_open)], dtype=np.float32)

    def _toggle_gripper(self):
        self._gripper_open = not self._gripper_open
        self._pending_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(self._gripper_open)], dtype=np.float32)
        self._gripper_label_var.set(f"Gripper: {'Open' if self._gripper_open else 'Closed'}")

    def _request_save(self):
        self._save_requested = True

    def _request_quit(self):
        self._quit_requested = True

    def _build_ui(self):
        pad = {"padx": 6, "pady": 6}

        self._root.grid_columnconfigure(0, weight=1)
        self._root.grid_columnconfigure(1, weight=1)

        title = tk.Label(self._root, text="Click To Move The Robot", font=("Arial", 14, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(10, 4))

        move_frame = tk.LabelFrame(self._root, text="Position", padx=8, pady=8)
        move_frame.grid(row=1, column=0, padx=10, pady=6, sticky="nsew")
        tk.Button(move_frame, text="Forward", width=10, command=lambda: self._queue_motion(dx=-self._dpos_scale)).grid(row=0, column=1, **pad)
        tk.Button(move_frame, text="Left", width=10, command=lambda: self._queue_motion(dy=-self._dpos_scale)).grid(row=1, column=0, **pad)
        tk.Button(move_frame, text="Right", width=10, command=lambda: self._queue_motion(dy=self._dpos_scale)).grid(row=1, column=2, **pad)
        tk.Button(move_frame, text="Back", width=10, command=lambda: self._queue_motion(dx=self._dpos_scale)).grid(row=2, column=1, **pad)
        tk.Button(move_frame, text="Up", width=10, command=lambda: self._queue_motion(dz=self._dpos_scale)).grid(row=0, column=3, **pad)
        tk.Button(move_frame, text="Down", width=10, command=lambda: self._queue_motion(dz=-self._dpos_scale)).grid(row=2, column=3, **pad)

        rot_frame = tk.LabelFrame(self._root, text="Rotation", padx=8, pady=8)
        rot_frame.grid(row=1, column=1, padx=10, pady=6, sticky="nsew")
        tk.Button(rot_frame, text="Roll +", width=10, command=lambda: self._queue_motion(droll=self._drot_scale)).grid(row=0, column=0, **pad)
        tk.Button(rot_frame, text="Roll -", width=10, command=lambda: self._queue_motion(droll=-self._drot_scale)).grid(row=0, column=1, **pad)
        tk.Button(rot_frame, text="Pitch +", width=10, command=lambda: self._queue_motion(dpitch=self._drot_scale)).grid(row=1, column=0, **pad)
        tk.Button(rot_frame, text="Pitch -", width=10, command=lambda: self._queue_motion(dpitch=-self._drot_scale)).grid(row=1, column=1, **pad)
        tk.Button(rot_frame, text="Yaw +", width=10, command=lambda: self._queue_motion(dyaw=self._drot_scale)).grid(row=2, column=0, **pad)
        tk.Button(rot_frame, text="Yaw -", width=10, command=lambda: self._queue_motion(dyaw=-self._drot_scale)).grid(row=2, column=1, **pad)

        action_frame = tk.LabelFrame(self._root, text="Episode", padx=8, pady=8)
        action_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=6, sticky="nsew")
        tk.Button(action_frame, text="Toggle Gripper", width=14, command=self._toggle_gripper).grid(row=0, column=0, **pad)
        tk.Button(action_frame, text="Save / Reset", width=14, command=self._request_save).grid(row=0, column=1, **pad)
        tk.Button(action_frame, text="Quit", width=14, command=self._request_quit).grid(row=1, column=0, columnspan=2, **pad)

        self._gripper_label_var = tk.StringVar(value="Gripper: Open")
        tk.Label(self._root, textvariable=self._gripper_label_var, font=("Arial", 11)).grid(row=3, column=0, columnspan=2, pady=(8, 4))
        tk.Label(
            self._root,
            text="Each click applies one small step.\nUse Save / Reset to finish an episode.",
            justify="center",
        ).grid(row=4, column=0, columnspan=2, pady=(0, 10))

    def poll(self):
        self._root.update_idletasks()
        self._root.update()

        action = self._pending_action.copy()
        self._pending_action[:] = 0.0
        action[-1] = float(self._gripper_open)

        save_requested = self._save_requested
        self._save_requested = False
        return action, save_requested, self._quit_requested

    def close(self):
        if self._root is not None:
            self._root.destroy()
            self._root = None


def _variant_id_from_task(task: str) -> str:
    task_lower = task.lower()
    if "blue" in task_lower:
        return "mug_blue"
    if "red" in task_lower:
        return "mug_red"
    return "mug_unspecified"


def _collect_auto(env, collector, task_str, args):
    from robot_vla_mujoco.datasets.collect_scripted import PickPlaceOracle

    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
        obs, info = env.reset(seed=args.seed + ep)
        oracle = PickPlaceOracle(env, joint_names=env._env.joint_names)

        task = obs.get("task", task_str)
        oracle.reset(task)
        collector.start_episode(task=task)
        variant_id = _variant_id_from_task(task)

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
            "variant_id": variant_id,
            "object_type": "mug",
            "difficulty": "auto",
            "seed": args.seed + ep,
            "success": success,
            "num_frames": step + 1,
        })

        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep}: {status}")


def _collect_manual(env, collector, task_str, args):
    print("Keyboard teleop controls:")
    print("  W/S: forward/back  A/D: left/right  R/F: up/down")
    print("  Q/E: roll  Arrow keys: pitch/yaw")
    print("  SPACE: toggle gripper  Z: save/reset episode  close viewer: quit")
    print("\nMake sure the MuJoCo viewer window is focused to receive key events.")

    ep = 0
    obs, info = env.reset(seed=args.seed)
    current_task = obs.get("task", task_str)
    collector.start_episode(task=current_task)
    step = 0

    while ep < args.episodes and env.is_viewer_alive():
        # Use SimpleEnv2's built-in teleop (requires viewer)
        action, reset_ep = env._env.teleop_robot()

        if reset_ep:
            success = env.is_success()
            collector.save_episode(metadata={
                "episode_id": f"{ep:06d}",
                "task_id": "pick_place_mug",
                "variant_id": _variant_id_from_task(current_task),
                "object_type": "mug",
                "difficulty": "manual",
                "seed": args.seed + ep,
                "success": success,
                "num_frames": step,
            })
            ep += 1
            print(f"Episode {ep} saved ({step} frames, success={success}). Starting new episode...")
            if ep >= args.episodes:
                break
            obs, info = env.reset(seed=args.seed + ep)
            current_task = obs.get("task", task_str)
            collector.start_episode(task=current_task)
            step = 0
            continue

        obs, reward, terminated, truncated, info = env.step(action)
        collector.add_frame(obs, action)

        if env.viewer_enabled:
            env.render(teleop=True, idx=ep)

        step += 1
        if step >= args.max_steps:
            collector.save_episode(metadata={
                "episode_id": f"{ep:06d}",
                "task_id": "pick_place_mug",
                "variant_id": _variant_id_from_task(current_task),
                "object_type": "mug",
                "difficulty": "manual",
                "seed": args.seed + ep,
                "success": env.is_success(),
                "num_frames": step,
            })
            ep += 1
            print(f"Episode {ep} truncated. Starting new episode...")
            if ep >= args.episodes:
                break
            obs, info = env.reset(seed=args.seed + ep)
            current_task = obs.get("task", task_str)
            collector.start_episode(task=current_task)
            step = 0

    if not env.is_viewer_alive():
        print("Viewer closed. Stopping manual collection.")


def _collect_manual_gui(env, collector, task_str, args):
    print("Button-panel teleop:")
    print("  Use the control panel window to move the robot one step per click.")
    print("  Toggle Gripper changes open/close state.")
    print("  Save / Reset stores the current episode and starts the next one.")
    print("  Quit closes manual collection.")

    panel = _ButtonTeleopPanel()
    ep = 0
    obs, info = env.reset(seed=args.seed)
    current_task = obs.get("task", task_str)
    collector.start_episode(task=current_task)
    step = 0

    try:
        while ep < args.episodes and env.is_viewer_alive():
            action, save_requested, quit_requested = panel.poll()
            if quit_requested:
                print("Control panel closed. Stopping manual collection.")
                break

            if save_requested:
                success = env.is_success()
                collector.save_episode(metadata={
                    "episode_id": f"{ep:06d}",
                    "task_id": "pick_place_mug",
                    "variant_id": _variant_id_from_task(current_task),
                    "object_type": "mug",
                    "difficulty": "manual_gui",
                    "seed": args.seed + ep,
                    "success": success,
                    "num_frames": step,
                })
                ep += 1
                print(f"Episode {ep} saved ({step} frames, success={success}). Starting new episode...")
                if ep >= args.episodes:
                    break
                obs, info = env.reset(seed=args.seed + ep)
                current_task = obs.get("task", task_str)
                collector.start_episode(task=current_task)
                step = 0
                continue

            moved = bool(np.any(np.abs(action[:6]) > 1e-9)) or step == 0
            if moved:
                obs, reward, terminated, truncated, info = env.step(action)
                collector.add_frame(obs, action)
                step += 1

            if env.viewer_enabled:
                env.render(teleop=True, idx=ep)
                time.sleep(1.0 / 60.0)

            if step >= args.max_steps:
                collector.save_episode(metadata={
                    "episode_id": f"{ep:06d}",
                    "task_id": "pick_place_mug",
                    "variant_id": _variant_id_from_task(current_task),
                    "object_type": "mug",
                    "difficulty": "manual_gui",
                    "seed": args.seed + ep,
                    "success": env.is_success(),
                    "num_frames": step,
                })
                ep += 1
                print(f"Episode {ep} truncated. Starting new episode...")
                if ep >= args.episodes:
                    break
                obs, info = env.reset(seed=args.seed + ep)
                current_task = obs.get("task", task_str)
                collector.start_episode(task=current_task)
                step = 0
    finally:
        panel.close()

    if not env.is_viewer_alive():
        print("Viewer closed. Stopping manual collection.")


if __name__ == "__main__":
    main()
