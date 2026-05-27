# Verification & Testing Guide

All commands assume the `lerobot-mujoco-manipulation` conda environment is activated, or prefixed with `conda run -n lerobot-mujoco-manipulation`.

Environment variables used throughout:
- `MODEL_SIM_VIEWER=1` — enable MuJoCo GLFW viewer (debug / visual inspection)
- `MODEL_SIM_VIEWER=0` — headless mode (policy evaluation, data collection, CI)
- `MUJOCO_GL=egl` — use EGL offscreen rendering (required for headless GPU render)
- `DISPLAY=:0` — X11 display target (only needed when viewer is on)

---

## 1. Asset Validation (Milestone 1)

Validate all MuJoCo XML assets can be loaded by the physics engine.

```bash
# Scan entire asset tree
MUJOCO_GL=egl DISPLAY=:0 conda run -n lerobot-mujoco-manipulation \
  python scripts/validate_assets.py --asset-root assets/mujoco

# Validate only the task scene
MUJOCO_GL=egl DISPLAY=:0 conda run -n lerobot-mujoco-manipulation \
  python scripts/validate_assets.py --scene-only
```

Expected: all XMLs under `robots/`, `objects/`, `tasks/` pass. Include-only fragments under `arenas/object/obj_*.xml` are skipped.

---

## 2. Smoke Tests (pytest)

Full CPU/headless test suite.

```bash
MUJOCO_GL=egl MODEL_SIM_VIEWER=0 conda run -n lerobot-mujoco-manipulation \
  python -m pytest tests/ -v --tb=short
```

Covers:
- `test_mjcf_load.py` — every standalone XML loads via MuJoCo
- `test_env_smoke.py` — env creation, reset, step, dummy rollout (50 steps), image shapes, success-condition registry

---

## 3. Dummy Policy Rollout (Environment Smoke)

Verify the simulation loop runs end-to-end with a no-op policy.

```bash
# Headless (fast, no display needed)
MODEL_SIM_VIEWER=0 MUJOCO_GL=egl conda run -n lerobot-mujoco-manipulation \
  python scripts/rollout_policy.py --policy dummy --episodes 3 --max-steps 50

# With viewer (visual debug, press ESC to quit)
conda run -n lerobot-mujoco-manipulation \
  python scripts/rollout_policy.py --policy dummy --episodes 1 --max-steps 200
```

---

## 4. Render Performance Benchmark

Separately measure physics, viewer, camera, and EGL render paths.

```bash
# Physics-only (no rendering)
MUJOCO_GL=egl DISPLAY=:0 conda run -n lerobot-mujoco-manipulation \
  python scripts/mujoco_render_benchmark.py --mode none --loops 50 --physics-steps 8

# EGL offscreen render (fast path for policy images)
MUJOCO_GL=egl DISPLAY=:0 conda run -n lerobot-mujoco-manipulation \
  python scripts/mujoco_render_benchmark.py --mode renderer --loops 50 --physics-steps 8 --camera-every 1

# Viewer basic render (GLFW path — may be slow on remote/headless hosts)
DISPLAY=:0 conda run -n lerobot-mujoco-manipulation \
  python scripts/mujoco_render_benchmark.py --mode viewer_fast --loops 80 --physics-steps 8

# Viewer with offscreen camera (simulates model_sim.py overlay path)
DISPLAY=:0 conda run -n lerobot-mujoco-manipulation \
  python scripts/mujoco_render_benchmark.py --mode camera --loops 80 --physics-steps 8 --camera-every 1
```

Expected: `renderer` (EGL) mode should achieve ~3 ms/frame camera render. `viewer_fast` may be slow on machines without local GPU OpenGL.

---

## 5. Data Collection (Milestone 2)

### Scripted Oracle (bulk collection)

```bash
# Headless fast collection
MODEL_SIM_VIEWER=0 MUJOCO_GL=egl conda run -n lerobot-mujoco-manipulation \
  python scripts/collect_demo.py --mode oracle --episodes 10 --max-steps 400 \
    --dataset-name omy_pick_place --dataset-root data/lerobot/omy_pick_place

# With viewer (watch the robot move)
conda run -n lerobot-mujoco-manipulation \
  python scripts/collect_demo.py --mode oracle --episodes 1 --max-steps 200
```

### Keyboard Teleop (manual debugging)

```bash
conda run -n lerobot-mujoco-manipulation \
  python scripts/collect_demo.py --mode teleop --episodes 1 --max-steps 500
```

Controls: `W/S` forward/back, `A/D` left/right, `R/F` up/down, `Q/E` roll, arrow keys pitch/yaw, `SPACE` toggle gripper, `Z` reset episode, `ESC` quit.

---

## 6. Dataset Inspection (Milestone 2)

### Replay an episode

```bash
# Print state/action summary (no GUI needed)
conda run -n lerobot-mujoco-manipulation \
  python scripts/replay_episode.py \
    --dataset data/lerobot/omy_pick_place --episode 0 --print-trajectory

# Save side-by-side video (front + wrist)
conda run -n lerobot-mujoco-manipulation \
  python scripts/replay_episode.py \
    --dataset data/lerobot/omy_pick_place --episode 0 --save-video
```

### Validate dataset schema

```bash
conda run -n lerobot-mujoco-manipulation \
  python scripts/validate_dataset.py --dataset data/lerobot/omy_pick_place
```

### Curate (filter failed episodes)

```bash
conda run -n lerobot-mujoco-manipulation \
  python scripts/curate_dataset.py \
    --dataset data/lerobot/omy_pick_place --keep-success-only --max-steps 400
```

---

## 7. Policy Evaluation (Milestone 3)

### Dummy policy (smoke test, no GPU needed)

```bash
# Single env
MODEL_SIM_VIEWER=0 MUJOCO_GL=egl conda run -n lerobot-mujoco-manipulation \
  python scripts/eval_policy.py --policy dummy --episodes 20 --max-steps 200

# Multi-env parallel (4 workers)
MODEL_SIM_VIEWER=0 MUJOCO_GL=egl conda run -n lerobot-mujoco-manipulation \
  python scripts/eval_policy.py --policy dummy --episodes 20 --num-envs 4
```

### SmolVLA / pi0 / pi0.5 (requires GPU + checkpoint)

```bash
# SmolVLA
python scripts/eval_policy.py --policy smolvla \
  --checkpoint lerobot/smolvla_base --episodes 20 --device cuda

# pi0
python scripts/eval_policy.py --policy pi0 \
  --checkpoint lerobot/pi0 --episodes 20 --device cuda

# pi0.5
python scripts/eval_policy.py --policy pi05 \
  --checkpoint lerobot/pi05 --episodes 20 --device cuda

# Save per-episode metrics to outputs/
python scripts/eval_policy.py --policy dummy --episodes 20 \
  --output-dir outputs/eval/omy_dummy
```

Expected output: success rate, avg steps, avg reward, avg FPS printed to stdout and saved to `eval_summary.json`.

---

## 8. Training (Milestone 4, dry-run only)

```bash
# Validate training config (no GPU needed)
conda run -n lerobot-mujoco-manipulation \
  python scripts/train_policy.py \
    --config configs/experiments/omy_smolvla_pick_place.yaml --dry-run

# Actual training (requires GPU + collected dataset)
python scripts/train_policy.py \
  --config configs/experiments/omy_smolvla_pick_place.yaml
```

---

## Quick-Verify Script (All-in-One)

```bash
#!/bin/bash
# Run all headless smoke checks. Exits non-zero on first failure.
set -euo pipefail
export MUJOCO_GL=egl
export MODEL_SIM_VIEWER=0
export DISPLAY=:0

ENV="conda run -n lerobot-mujoco-manipulation"

echo "=== 1. Asset validation ==="
$ENV python scripts/validate_assets.py --asset-root assets/mujoco

echo "=== 2. Pytest suite ==="
$ENV python -m pytest tests/ -v --tb=short

echo "=== 3. Dummy rollout (headless) ==="
$ENV python scripts/rollout_policy.py --policy dummy --episodes 2 --max-steps 30

echo "=== 4. Dummy eval ==="
$ENV python scripts/eval_policy.py --policy dummy --episodes 2 --max-steps 20 --no-metrics

echo "=== ALL CHECKS PASSED ==="
```
