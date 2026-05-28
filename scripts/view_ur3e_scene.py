#!/usr/bin/env python3
"""Visual smoke test: load the UR3e+AG95 scene with the MuJoCo viewer.

Usage:
    python scripts/view_ur3e_scene.py          # viewer on (default)
    MODEL_SIM_VIEWER=0 python scripts/view_ur3e_scene.py  # headless smoke test only
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in {"0", "false", "no", "off"}


def main():
    use_viewer = _env_bool("MODEL_SIM_VIEWER", True)
    xml_path = os.path.abspath("assets/mujoco/tasks/pick_place_mug_ur3e.xml")

    print(f"Loading: {xml_path}")
    print(f"Viewer: {use_viewer}")

    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print(f"Model loaded: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")
    print(f"Cameras: {model.ncam}")

    # List actuator names and their target joints
    act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    for i, name in enumerate(act_names):
        print(f"  actuator[{i}]: {name}, ctrlrange={model.actuator_ctrlrange[i]}")

    if not use_viewer:
        # Headless smoke: take a few steps and render to offscreen
        renderer = mujoco.Renderer(model, height=480, width=640)
        for i in range(10):
            mujoco.mj_step(model, data)
            renderer.update_scene(data, camera="agentview")
            img = renderer.render()
            if i == 0:
                print(f"  Agentview image shape: {img.shape}, mean={img.mean():.1f}")
        renderer.close()
        print("Headless smoke test PASSED")
        return

    # Viewer mode
    print("\nOpening viewer... Press ESC to quit, SPACE to pause/resume.")
    print("Use mouse to rotate/zoom/pan the view.")

    from robot_vla_mujoco.mujoco_env.mujoco_parser import MuJoCoParserClass

    # Use MuJoCoParserClass for the viewer (handles GLFW context)
    env_parser = MuJoCoParserClass(name="UR3eScene", rel_xml_path=xml_path)
    env_parser.init_viewer(
        width=1280,
        height=900,
        distance=3.0,
        elevation=-25,
        azimuth=140,
        lookat=[0.5, 0.0, 0.5],
    )

    # Simulate with zero control (arm stays at initial pose)
    print("\nRunning simulation with gravity only...")
    step_count = 0
    while env_parser.is_viewer_alive():
        # Step physics
        mujoco.mj_step(env_parser.model, env_parser.data)
        step_count += 1

        # Optional: apply tiny jitter to shoulder pan to show movement
        # env_parser.data.ctrl[0] = 0.1 * np.sin(step_count * 0.01)

        # Text overlay
        env_parser.viewer_text_overlay(
            text1="UR3e + AG95",
            text2=f"Step: {step_count} | Bodies: {model.nbody} | Joints: {model.njnt}",
        )
        env_parser.render()

        # Throttle to ~60fps
        if step_count % 3 == 0:
            time.sleep(0.001)

    env_parser.close_viewer()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
