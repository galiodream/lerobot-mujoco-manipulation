# pi0 Evaluation Performance Analysis & Fixes

Date: 2026-05-28
Tags: `pi0`, `eval_policy`, `AsyncInferenceEngine`, `model_sim.py`, `sim_time`, `wall_time`, `viewer lag`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Issue 1: `ValueError: All image features are missing from the batch`](#2-issue-1-valueerror-all-image-features-are-missing-from-the-batch)
3. [Issue 2: `RuntimeError: The size of tensor a (7) must match the size of tensor b (6)`](#3-issue-2-runtimeerror-the-size-of-tensor-a-7-must-match-the-size-of-tensor-b-6)
4. [Issue 3: Simulation runs slowly — sim_time vs wall_time mismatch](#4-issue-3-simulation-runs-slowly--sim_time-vs-wall_time-mismatch)
5. [Issue 4: Viewer stuttering — policy inference blocks the render loop](#5-issue-4-viewer-stuttering--policy-inference-blocks-the-render-loop)
6. [Summary of All Changes](#6-summary-of-all-changes)
7. [Key Takeaway](#7-key-takeaway)

---

## 1. Overview

We encountered several issues when running `eval_policy.py` with the `pi0` policy and a local fine-tuned checkpoint. The symptoms ranged from immediate crashes to severe runtime performance degradation when the viewer was enabled.

The reference file `model_sim.py` (in the companion project `lerobot-mujuco`) did **not** exhibit these problems with the same checkpoint and scene, making it an ideal baseline for comparison.

---

## 2. Issue 1: `ValueError: All image features are missing from the batch`

### Error

```
ValueError: All image features are missing from the batch.
At least one expected.
(batch: dict_keys(['observation.state', 'observation.image', 'observation.wrist_image', 'task']))
(image_features:{})
```

### Root Cause

In `pi0_adapter.py`, the config was constructed from scratch:

```python
cfg = PI0Config(chunk_size=50, n_action_steps=5)
self._policy = PI0Policy.from_pretrained(checkpoint_path, config=cfg)
```

The default `PI0Config()` has an empty `input_features` dictionary (`{}`), so `cfg.image_features` (which filters for `FeatureType.VISUAL`) was empty. When `prepare_images()` checked `self.config.image_features`, it found nothing — but the batch did contain image keys — hence the error.

### Fix

Let `from_pretrained` load the checkpoint's own `config.json` (which has the correct `input_features`), then override only the necessary fields:

```python
self._policy = PI0Policy.from_pretrained(checkpoint_path)
self._policy.config.chunk_size = chunk_size
self._policy.config.n_action_steps = n_action_steps
```

This matches how `model_sim.py` works — it does not pass `config=cfg`, so the checkpoint's config is loaded automatically.

### Files Changed

- `src/robot_vla_mujoco/policies/pi0_adapter.py`

---

## 3. Issue 2: `RuntimeError: The size of tensor a (7) must match the size of tensor b (6)`

### Error

```
RuntimeError: The size of tensor a (7) must match the size of tensor b (6)
at non-singleton dimension 1
```

### Root Cause

The environment's `get_joint_state()` returns a 7-dimensional vector (6 joint angles + 1 gripper state):

```python
# y_env2.py:281-291
def get_joint_state(self):
    qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
    gripper = self.env.get_qpos_joint('rh_r1')
    gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
    return np.concatenate([qpos, [gripper_cmd]], dtype=np.float32)
```

However, the checkpoint's `config.json` defines `observation.state` with shape `[6]`:

```json
"input_features": {
    "observation.state": {"type": "STATE", "shape": [6]},
    ...
}
```

The normalization layer tries to apply 6-dimensional `mean`/`std` to a 7-dimensional input, causing the dimension mismatch.

### Fix

Truncate the state to match the expected dimension from the checkpoint config:

```python
state = observation.get("observation.state")
if state is None:
    state = np.zeros(self._action_dim, dtype=np.float32)
elif state.shape[-1] > self._policy.config.input_features["observation.state"].shape[0]:
    state = state[..., :self._policy.config.input_features["observation.state"].shape[0]]
```

`model_sim.py` already did this manually: `state = PnPEnv.get_joint_state()[:6]`.

### Files Changed

- `src/robot_vla_mujoco/policies/pi0_adapter.py`

---

## 4. Issue 3: Simulation runs slowly — sim_time vs wall_time mismatch

### Symptom

Without the viewer (headless mode), the simulation was slow in terms of physical time progression.

### Root Cause

`mujoco_env.py` called `step_env()` with the default `nstep=1`:

```python
def step(self, action):
    self._env.step(action)
    self._env.step_env()       # default nstep=1 → 0.002s per step
    self._step_count += 1
```

MuJoCo's `dt=0.002s`, so each step advanced simulation time by only 2ms. To advance by 1 frame (16.7ms, the standard 60Hz interval), 8 steps were needed.

`model_sim.py` computed multi-step physics:

```python
SIM_DT = PnPEnv.env.dt             # 0.002s
RENDER_HZ = 60
FRAME_INTERVAL = 1.0 / RENDER_HZ   # ~0.0167s
PHYSICS_STEPS = max(1, round(FRAME_INTERVAL / SIM_DT))  # ~8 steps
PnPEnv.step_env(nstep=PHYSICS_STEPS)
```

### Fix

Adopt the same logic in `mujoco_env.py`:

```python
def step(self, action):
    self._env.step(action)
    sim_dt = self._env.env.model.opt.timestep
    physics_steps = max(1, round((1.0 / 60.0) / sim_dt))
    self._env.step_env(nstep=physics_steps)
    self._step_count += 1
```

### Verification (headless)

```
60 steps: wall_time=0.239s, sim_time=0.960s
Sim time per step: 0.0160s   # ≈ 1/60s ✓
Wall time per step: 0.0040s
Sim/Wall ratio: 4.02x
```

### Files Changed

- `src/robot_vla_mujoco/envs/mujoco_env.py`

---

## 5. Issue 4: Viewer stuttering — policy inference blocks the render loop

### Symptom

With the viewer enabled, the simulation was extremely choppy — sim_time updates every ~2 seconds, and the viewer was unresponsive. This occurred even after fixing the physics step count.

### Root Cause

The rollout loop was **synchronous**: every time the `ActionChunkBuffer` was empty, it called `policy.predict_action_trajectory(obs)` inline, which blocks for ~300ms (pi0 inference on GPU). With `prediction_horizon=1`, the buffer was empty every single step.

```python
# rollot.py (before fix)
for step in range(max_steps):
    if chunk_buffer.is_empty:
        action_traj = policy.predict_action_trajectory(obs)  # ← blocks 300ms!
        chunk_buffer.add_trajectory(action_traj)
    action = chunk_buffer.get_action()
    obs, reward, terminated, truncated, info = env.step(action)
    if render:
        env.render(...)
```

This meant: 300ms inference → 16ms physics → 300ms inference → 16ms physics → ..., yielding approximately **3 FPS** in the viewer.

`model_sim.py` avoids this by running inference in a **background thread** (3Hz inference, 60Hz render loop).

### Fix

Two orthogonal changes:

#### 5a. Increase `prediction_horizon` default

Changed default from 1 to 5, matching pi0's `chunk_size`. This gives the async engine ~800ms to finish inference before the buffer runs out.

#### 5b. New `AsyncInferenceEngine` class

```python
# async_inference.py
class AsyncInferenceEngine:
    """Runs policy inference in a background thread.
    Continuously produces new trajectories from the latest observation."""
```

Key design:
- Background thread runs `policy.predict_action_trajectory(obs)` in a loop
- `update_observation(obs)` feeds the latest observation to the worker
- `collect_trajectory()` returns a new trajectory if one is ready
- Continuous mode: if new observations arrive while the worker is still processing, it restarts inference immediately with the latest data

#### 5c. Rewritten rollout loop

```python
# rollot.py (after fix)
async_engine = AsyncInferenceEngine(policy)
async_engine.reset(task)
async_engine.update_observation(obs)

# Prime buffer with synchronous first inference
new_traj = policy.predict_action_trajectory(obs)
chunk_buffer.add_trajectory(new_traj)

for step in range(max_steps):
    new_traj = async_engine.collect_trajectory()
    if new_traj is not None:
        chunk_buffer.add_trajectory(new_traj)  # "replace" strategy
    action = chunk_buffer.get_action_or_last()
    obs, reward, terminated, truncated, info = env.step(action)
    async_engine.update_observation(obs)
    if render:
        env.render(...)
```

#### 5d. `get_action_or_last()` in buffer

Added a new method to `ActionChunkBuffer` that returns the last known action when the buffer is empty, instead of returning `None` (which caused zeroing out):

```python
def get_action_or_last(self) -> np.ndarray | None:
    action = self.get_action()
    if action is not None:
        return action
    return self._last_action.copy() if self._last_action is not None else None
```

### Files Changed / Created

- `src/robot_vla_mujoco/policies/async_inference.py` **(new)**
- `src/robot_vla_mujoco/policies/action_decoder.py`
- `src/robot_vla_mujoco/inference/rollout.py`

---

## 6. Summary of All Changes

| File | Type | Change |
|------|------|--------|
| `src/robot_vla_mujoco/policies/pi0_adapter.py` | Modified | Load config from checkpoint instead of creating default; truncate state to match expected dim |
| `src/robot_vla_mujoco/envs/mujoco_env.py` | Modified | Advance physics by ~8 steps per policy step (was 1) |
| `src/robot_vla_mujoco/policies/async_inference.py` | **New** | Background thread for async policy inference |
| `src/robot_vla_mujoco/policies/action_decoder.py` | Modified | Added `get_action_or_last()`; default overlap to `replace` |
| `src/robot_vla_mujoco/inference/rollout.py` | Modified | Use AsyncInferenceEngine, default prediction_horizon=5 |
| `src/robot_vla_mujoco/inference/evaluate_sim.py` | Modified | Removed `inference_hz` parameter (no longer needed) |
| `scripts/eval_policy.py` | Modified | Removed `--inference-hz` flag |

---

## 7. Key Takeaway

The root cause of all performance problems was **synchronous blocking** of the main loop by pi0 inference (~300ms per call). When combined with `prediction_horizon=1`, every step blocked. The fix decouples inference into a background thread, allowing physics and rendering to run at full speed while inference completes asynchronously.

The other issues (missing `input_features`, state dimension mismatch, insufficient physics steps) were secondary problems exposed by the fact that the evaluation path had diverged from the proven `model_sim.py` implementation patterns.
