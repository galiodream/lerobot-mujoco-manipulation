# 基于 LeRobot + MuJoCo + LingBot-VLA 的机器人多模态策略后训练与动作建模项目计划

## 1. 项目定位

本项目目标是搭建一个可扩展的机器人多模态策略后训练与仿真推理框架，核心技术栈为 MuJoCo、LeRobot、SmolVLA、pi0/pi0.5 与 LingBot-VLA。

建议第一阶段采用“仿真闭环 MVP + 可扩展后训练框架”的路线：

- 先跑通 `UR3e + AG95` 单臂桌面抓取与放置任务。
- 使用 MuJoCo 构建可复用仿真环境、机械臂资产、桌面场景和任务物体。
- 使用 LeRobot 数据格式作为数据采集、训练和模型推理之间的统一接口。
- 首期优先支持 SmolVLA 和 pi0/pi0.5 推理闭环。
- 后续接入 LingBot-VLA 后训练，按其要求准备 LeRobot v3 数据、robot config 和 norm stats。

推荐模型路线：

1. `SmolVLA`：首个 VLA baseline，轻量、官方 LeRobot 流程清晰，适合快速验证。
2. `pi0 / pi0.5`：作为更强的 VLA/action flow baseline。
3. `LingBot-VLA`：作为最终的高能力后训练路线，独立管理重依赖和训练配置。

参考项目：

- [galiodream/lerobot-mujuco](https://github.com/galiodream/lerobot-mujuco)：MuJoCo + LeRobot 数据采集、训练、pi0/SmolVLA 推理示例。
- [hangtingLiu/UR3e_AG95_Grasp_public](https://github.com/hangtingLiu/UR3e_AG95_Grasp_public)：UR3e + AG95 资产、环境初始化、RGB/depth 渲染和抓取任务参考。
- [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)：高质量 MuJoCo 机器人资产组织方式参考。
- [mikedh/trimesh](https://github.com/mikedh/trimesh)：mesh 加载、检查、转换、惯量估算和资产预处理。
- [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)：SO-ARM100/SO-101 低成本机械臂资产与 LeRobot 生态参考。
- [huggingface/lerobot](https://github.com/huggingface/lerobot)：模型加载、数据格式、训练和 rollout 主框架。
- [Robbyant/lingbot-vla](https://github.com/Robbyant/lingbot-vla)：LingBot-VLA 后训练和部署参考。

## 2. 推荐目录结构

```text
lerobot-mujoco-manipulation/
  README.md
  PROJECT_PLAN.md
  pyproject.toml
  constraints/
    core.txt
    lerobot.txt
  requirements/
    base.txt
    dev.txt
    lerobot.txt
    lingbot.txt

  configs/
    envs/
      ur3e_ag95_pick_place.yaml
      so_arm100_pick_place.yaml
    robots/
      ur3e_ag95.yaml
      so_arm100.yaml
    policies/
      smolvla.yaml
      pi0.yaml
      pi05.yaml
      lingbot_vla.yaml
    experiments/
      ur3e_ag95_smolvla_pick_place.yaml
      ur3e_ag95_pi0_pick_place.yaml
      ur3e_ag95_lingbot_posttrain.yaml

  assets/
    mujoco/
      robots/
        ur3e_ag95/
          scene.xml
          robot.xml
          gripper.xml
          meshes/
        so_arm100/
          scene.xml
          robot.xml
          meshes/
      arenas/
        table_basic.xml
        tabletop_cameras.xml
      objects/
        mugs/
        blocks/
        plates/
      tasks/
        pick_place_mug.xml
        stack_blocks.xml
    manifests/
      asset_sources.yaml

  src/
    robot_vla_mujoco/
      common/
        config.py
        registry.py
        logging.py
        seed.py
        types.py

      assets/
        mjcf_loader.py
        mesh_tools.py
        scene_builder.py

      robots/
        base.py
        ur3e_ag95.py
        so_arm100.py

      envs/
        base.py
        mujoco_env.py
        vector_env.py
        task_registry.py
        randomization.py
        success_conditions.py
        wrappers/
          lerobot_wrapper.py

      datasets/
        collect_teleop.py
        collect_scripted.py
        lerobot_writer.py
        metadata.py
        curate.py
        transforms.py
        replay.py
        convert.py
        validate.py

      policies/
        base.py
        lerobot_policy.py
        smolvla_adapter.py
        pi0_adapter.py
        pi05_adapter.py
        lingbot_adapter.py
        action_decoder.py

      inference/
        rollout.py
        closed_loop.py
        evaluate_sim.py

      training/
        train_lerobot.py
        train_lingbot.py
        compute_norm_stats.py

      eval/
        metrics.py
        open_loop.py
        benchmark.py
        regression.py

  scripts/
    collect_demo.py
    replay_episode.py
    train_policy.py
    rollout_policy.py
    eval_policy.py
    validate_assets.py
    validate_dataset.py
    curate_dataset.py

  data/
    .gitkeep

  outputs/
    .gitkeep

  tests/
    test_mjcf_load.py
    test_env_smoke.py
    test_lerobot_schema.py
    test_action_adapter.py
    test_policy_dummy.py

  .github/
    workflows/
      ci.yml
```

## 3. 架构设计

### 3.1 MuJoCo 资产层

MuJoCo 资产层负责维护机器人、夹爪、桌面、相机、任务物体和组合场景。

设计原则：

- `assets/mujoco/robots/` 保存机械臂、夹爪、mesh、actuator、joint、camera。
- `assets/mujoco/arenas/` 保存桌面、灯光、相机布局和物理参数。
- `assets/mujoco/objects/` 保存杯子、方块、盘子等任务物体。
- `assets/mujoco/tasks/` 保存由 robot、arena、objects 组合出的任务级 scene。
- XML 使用 `include` 组合，避免每个任务复制完整机械臂和桌面。
- 每个 robot 目录保留资产来源、license 和 mesh 缩放说明。

第一版建议：

- 默认实现 `UR3e + AG95`。
- 后续加入 `SO-ARM100/SO-101`。
- 先支持单臂桌面任务，再扩展双臂、移动底盘或复杂长程任务。

`mesh_tools.py` 后续职责：

- 检查 mesh 是否存在、路径是否正确。
- 使用 `trimesh` 加载 STL/OBJ/GLB。
- 做 mesh 缩放、坐标轴修正、简化和惯量估算。
- 输出 MuJoCo 友好的 mesh manifest。

`assets/manifests/asset_sources.yaml` 从 Milestone 1 开始就需要定义，避免资产来源、license、缩放和转换记录在后续扩展时失控。建议格式：

```yaml
assets:
  - name: ur3e_ag95_robot
    type: robot
    source_url: https://github.com/hangtingLiu/UR3e_AG95_Grasp_public
    license: "check upstream repository before redistribution"
    version: "main@<commit-sha>"
    local_path: assets/mujoco/robots/ur3e_ag95
    scale: 1.0
    orientation_fix: null
    conversion_required: false
    notes: "UR3e + AG95 MuJoCo XML and meshes used as first baseline asset."
  - name: mug_set
    type: object
    source_url: "<source-url>"
    license: "<license>"
    version: "<version-or-commit>"
    local_path: assets/mujoco/objects/mugs
    scale: 1.0
    orientation_fix: "z-up if source asset is y-up"
    conversion_required: true
    notes: "Converted with trimesh; keep original mesh and converted mesh paths."
```

每个 mesh 或 XML 资产至少记录 `source_url`、`license`、`version`、`local_path`、`scale`、`orientation_fix`、`conversion_required` 和 `notes`。如果 license 不明确，资产可以用于本地实验，但不能作为可再分发资源提交或发布。

### 3.2 环境层

环境层对外提供统一的仿真接口，屏蔽 MuJoCo 原始 API、不同 robot profile 和不同 action mode 的差异。

核心类建议为 `MujocoManipulationEnv`：

```python
class MujocoManipulationEnv:
    def reset(self, seed: int | None = None, options: dict | None = None) -> dict: ...
    def step(self, action) -> tuple[dict, float, bool, bool, dict]: ...
    def render(self, camera_names: list[str] | None = None) -> dict: ...
    def get_observation(self) -> dict: ...
    def is_success(self) -> bool: ...
    def close(self) -> None: ...
```

`step()` 采用 Gymnasium 风格返回 `(observation, reward, terminated, truncated, info)`：

- `terminated` 表示任务自然结束，例如成功、明确失败或不可恢复状态。
- `truncated` 表示被外部条件截断，例如达到 `max_steps`、安全限幅触发或用户中断。
- `info` 至少包含 `success`、`episode_metrics`、`success_metrics`、`step_count`、`sim_time`、`task_id`、`variant_id`、`seed` 和 `truncation_reason`。
- `episode_metrics` 记录可累计指标，例如距离目标的最小值、碰撞次数、动作平滑度和 episode length。
- `success_metrics` 来自当前 `SuccessCondition.metrics()`，用于解释成功条件每个子项是否满足。

第一版 observation：

```text
observation.images.front      RGB image
observation.images.wrist      RGB image
observation.state             robot proprio/state
task                          language instruction
```

第一版 action：

```text
joint_position + gripper
```

UR3e + AG95 默认 action 维度：

```text
6 arm joints + 1 gripper = 7
```

后续 action mode：

- `joint_position`
- `joint_velocity`
- `delta_joint`
- `ee_delta_pose`
- `hybrid_ee_gripper`

并行环境设计：

- 单实例 `MujocoManipulationEnv` 仍是核心接口，所有 wrapper 都围绕它组合。
- `mujoco.viewer` 只用于单实例交互调试，不参与并行采集和批量评测。
- 并行数据采集和评测默认使用 headless 多进程，优先实现 `SyncVectorEnv`，后续再扩展 `AsyncVectorEnv`。
- `SyncVectorEnv` 用于可复现 benchmark 和小规模并行 rollout；`AsyncVectorEnv` 用于大规模数据采集或慢环境混合。
- 每个子进程独立持有 MuJoCo model/data/renderer，避免跨线程共享 MuJoCo viewer 或 GL context。

task config 中需要预留并行和成功条件字段：

```yaml
vector:
  num_envs: 1
  backend: sync  # sync | async

success:
  type: pick_place
  params:
    object_body: mug
    target_body: plate
    distance_threshold: 0.05
    stable_steps: 10
```

成功条件使用 `SuccessCondition` 协议和 registry，不把 pick-place 逻辑写死在环境主类里：

```python
class SuccessCondition:
    def reset(self, env, success_params: dict, task_context: dict | None = None) -> None: ...
    def update(self, env, obs: dict, action) -> None: ...
    def is_success(self) -> bool: ...
    def metrics(self) -> dict: ...
```

任务 YAML 通过 `success.type: pick_place | stack | push | insert` 选择成功条件。`success_params` 明确来自实验配置中的 `env.success.params`，经过 config resolver 展开后传入；不要把完整 experiment config 传给 condition。`task_context` 是可选的 episode 级上下文，例如 `task_id`、`variant_id`、object alias 和采样后的初始位姿。第一版只实现 `pick_place`，但 `stack`、`push`、`insert` 等任务后续只需要新增 condition 类并注册。

`pick_place` 第一版成功条件：

- 物体中心进入目标区域。
- 物体高度稳定在目标容器或盘子上。
- 夹爪已打开。
- 末端执行器离开物体一定距离。
- 持续满足若干仿真步，避免瞬时误判。

### 3.3 数据层

数据层负责演示采集、回放、转换、校验和 LeRobot dataset 写入。

核心模块：

- `collect_scripted.py`：脚本专家策略生成首批可训练 demo，优先实现 oracle pick-place。
- `collect_teleop.py`：键盘 teleop 用于 debug，SpaceMouse 或 GUI slider 作为可选增强。
- `lerobot_writer.py`：写入 LeRobot 数据格式。
- `metadata.py`：写入和读取 episode/task variant metadata。
- `curate.py`：过滤失败 episode、截断过长 episode、按 metadata 做采样均衡。
- `transforms.py`：训练时 data transform/augmentation 配置入口。
- `replay.py`：复放 episode，对齐画面、状态和动作。
- `validate.py`：检查数据 schema、fps、图像 shape、action dim。

命令入口约定：

- 用户直接运行的工作流入口统一放在 `scripts/` 下，例如 `scripts/collect_demo.py`、`scripts/replay_episode.py`、`scripts/curate_dataset.py`。
- `src/robot_vla_mujoco/datasets/*.py` 保留为可导入模块，供脚本、测试和训练代码复用。
- 验收命令优先使用 `python scripts/*.py`，只有开发调试或库内部测试才使用 `python -m robot_vla_mujoco...`。

`scripts/replay_episode.py` 用于人工检查 dataset 质量：默认逐帧复放指定 episode，显示或保存 front/wrist 图像、状态和动作轨迹；加 `--save-video` 时输出对齐视频，加 `--print-trajectory` 时只打印 state/action summary，适合无 GUI/headless 环境。

第一版 LeRobot dataset schema 建议：

```python
features = {
    "observation.images.front": {
        "dtype": "image",
        "shape": (256, 256, 3),
    },
    "observation.images.wrist": {
        "dtype": "image",
        "shape": (256, 256, 3),
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
    },
    "task": {
        "dtype": "string",
    },
}
```

episode metadata 不放进每一帧 observation，而是作为 episode 级元数据写入 LeRobot metadata，供训练过滤、采样均衡和 benchmark 切片使用。第一版需要记录：

```python
episode_metadata = {
    "episode_id": "000001",
    "task_id": "pick_place_mug",
    "variant_id": "mug_red_plate_white_easy",
    "object_type": "mug",
    "object_initial_pose": [0.42, 0.05, 0.02, 1.0, 0.0, 0.0, 0.0],
    "target_pose": [0.55, -0.10, 0.02, 1.0, 0.0, 0.0, 0.0],
    "difficulty": "easy",
    "seed": 42,
    "success": True,
    "env_config_hash": "<hash-of-env-robot-task-config>",
}
```

数据后处理策略：

- 原始采集数据保持不可变，不把图像增强离线写回 dataset。
- 图像增强默认在训练 transform 层完成，例如 random crop、color jitter、噪声、轻微模糊。
- `curate.py` 只产出过滤后的 episode index 或轻量 manifest，用于丢弃失败 episode、截断异常长 episode、平衡不同物体/初始位置/难度分布。
- 训练入口根据 metadata 支持过滤，例如只训练 `object_type=mug` 或只评测 `difficulty=hard`。

需要保证：

- 采集数据和训练数据共用同一套 robot/action config。
- 图像键名、state 键名、action 维度能被 SmolVLA、pi0/pi0.5 和 LingBot adapter 复用。
- metadata key 的命名稳定，后续新增字段只能向后兼容。
- dataset root 默认放在 `data/lerobot/<dataset_name>`。
- 大数据不提交到 git。

### 3.4 策略层

策略层负责统一不同模型的加载、预处理、推理、后处理和动作解码。

统一接口建议：

```python
class PolicyAdapter:
    def reset(self, task: str) -> None: ...
    def predict_action_trajectory(self, observation: dict) -> np.ndarray: ...
    def close(self) -> None: ...
```

约定 `predict_action_trajectory()` 始终返回标准化 action trajectory，shape 为 `(horizon, action_dim)`。如果模型只输出单步 action，则返回 `(1, action_dim)`；如果模型输出 action chunk 或 flow matching trajectory，则保留完整 horizon。最终由 `ActionDecoder` 和 `ActionChunkBuffer` 决定每个 control step 实际执行哪一步。

各 adapter 职责：

- `smolvla_adapter.py`：加载 `lerobot/smolvla_base` 或本地 fine-tuned checkpoint。
- `pi0_adapter.py`：加载 LeRobot pi0 policy。
- `pi05_adapter.py`：加载 LeRobot pi0.5 policy。
- `lingbot_adapter.py`：加载 LingBot-VLA post-trained checkpoint。
- `action_decoder.py`：处理 action trajectory、反归一化、限幅、action mode 转换和 gripper 映射。
- `ActionChunkBuffer`：缓存模型输出的 action chunk，统一处理单步输出、chunk overlap 和 temporal ensemble。

`ActionChunkBuffer` 配置项：

```yaml
action_chunk:
  prediction_horizon: 10
  execution_horizon: 5
  overlap_strategy: queue  # replace | queue | temporal_ensemble
  action_smoothing: 0.0
```

不同模型输出差异：

- SmolVLA 可能输出单步 action 或较短 action chunk，适合作为第一版闭环 baseline。
- pi0/pi0.5 更可能输出 horizon 大于 1 的 action trajectory，需要明确 `prediction_horizon` 和 `execution_horizon`。
- LingBot-VLA adapter 也只产出标准 action trajectory，不直接操作 MuJoCo control。
- `temporal_ensemble` 后续用于多个重叠 chunk 的加权融合；第一版默认使用 `queue`，减少实现复杂度。

第一版优先实现 SmolVLA adapter：

- SmolVLA 输入天然包含多视角图像、机器人状态和语言指令。
- 输出为连续动作，适合和 MuJoCo joint action 对接。
- 官方 LeRobot 训练命令和 checkpoint 加载方式较稳定。

LingBot-VLA 独立管理原因：

- 依赖更重。
- 训练环境要求更高。
- 需要额外准备 robot config、norm stats、Qwen2.5-VL 和 depth 相关权重。
- 不应阻塞基础 MuJoCo + LeRobot 闭环开发。

### 3.5 训练层

训练层分为 LeRobot 训练和 LingBot 后训练两条线。

LeRobot 训练：

- `train_lerobot.py` 封装 `lerobot-train`。
- 支持 SmolVLA、pi0、pi0.5。
- 训练参数从 `configs/experiments/*.yaml` 读取。
- 默认输出到 `outputs/train/<run_name>`。

训练 transform 和数据增强：

- 默认在训练 dataloader/transform 层做图像增强，不改写原始 LeRobot dataset。
- 第一版只预留 transform 配置入口，实际增强可以从 resize、center crop 和 normalization 开始。
- random crop、color jitter、noise 等增强作为训练配置开关，不作为数据采集或 dataset 写入职责。

日志和指标：

- 默认使用本地 `JSONL + CSV + rollout videos`，保证离线环境也能复现。
- WandB 作为可选项，通过 `logging.wandb.enable: true/false` 控制。
- 第一版不同时接入 TensorBoard/MLflow，避免日志系统过早复杂化。
- 每次训练和评测需要记录 config snapshot、git commit 或 dirty-state 标记、dataset metadata summary 和 metrics summary。

LingBot 后训练：

- `compute_norm_stats.py` 计算 norm stats。
- `train_lingbot.py` 封装 LingBot-VLA 官方训练入口。
- `configs/policies/lingbot_vla.yaml` 保存模型路径、depth 开关、norm stats 路径。
- `configs/robots/<robot>.yaml` 同时服务 LeRobot 和 LingBot feature mapping。

LingBot 数据准备流程：

1. 准备 LeRobot v3 格式数据。
2. 定义 robot config，映射 images、states、actions。
3. 计算 norm stats。
4. 使用 LingBot-VLA 配置进行 post-training。
5. 导出 checkpoint 给 `lingbot_adapter.py` 做仿真闭环评测。

依赖版本策略：

- 推荐核心仿真环境和 LingBot 后训练环境分离，LingBot 依赖不阻塞基础 MuJoCo + LeRobot 闭环。
- `requirements/*.txt` 是直接安装列表，描述需要安装哪些包。
- `constraints/*.txt` 是 pip `-c` 使用的版本约束文件，固定或限制核心依赖版本，避免 MuJoCo、numpy、Torch、LeRobot 之间漂移。
- `requirements/base.txt` 安装时使用 `pip install -r requirements/base.txt -c constraints/core.txt`。
- `requirements/lerobot.txt` 安装时使用 `pip install -r requirements/lerobot.txt -c constraints/core.txt -c constraints/lerobot.txt`。
- `constraints/core.txt` 固定 Python 兼容范围说明、MuJoCo、numpy、opencv、trimesh 等核心依赖版本。
- `constraints/lerobot.txt` 固定 LeRobot、Torch、Transformers、Datasets 等训练推理依赖。
- `requirements/lingbot.txt` 独立管理 LingBot-VLA、Qwen、depth 相关重依赖，安装时使用 LingBot 专属 constraints 或官方 lock 方案，不和核心环境强行合并。
- README 或 constraints 中必须明确 Python、CUDA、Torch、MuJoCo 和 LeRobot 版本组合；遇到 numpy/Torch/MuJoCo 冲突时，以可跑通核心仿真闭环的组合为基准。

### 3.6 推理与评测层

推理模块：

- `rollout.py`：闭环 rollout。
- `closed_loop.py`：action chunk 执行和环境交互。
- `evaluate_sim.py`：批量 episode 评测。
- 批量评测优先使用 headless `SyncVectorEnv`，viewer 只用于单 episode 可视化 debug。

评测模块：

- `open_loop.py`：dataset 上预测 action，与 ground-truth action 对齐。
- `benchmark.py`：批量模型、批量任务、批量种子评测。
- `regression.py`：后续 benchmark regression，对固定 seed 的 N 个 episode 生成对比报告。
- `metrics.py`：成功率、轨迹长度、碰撞数、动作平滑度、推理延迟等。

核心指标：

- `success_rate`
- `average_progress`
- `episode_length`
- `collision_count`
- `action_l1_error`
- `action_l2_error`
- `action_smoothness`
- `inference_latency_ms`

benchmark regression 分阶段实施：

- 第一阶段只记录固定 seed、固定任务配置下的指标报告，不阻塞开发。
- 第二阶段为 dummy policy、scripted oracle 和轻量 checkpoint 设置 smoke threshold。
- 第三阶段再为真实 VLA checkpoint 设置 success rate 回归阈值，GPU/VLA 用例标记为 optional 或 nightly。

## 4. 关键配置示例

```yaml
# configs/experiments/ur3e_ag95_smolvla_pick_place.yaml
seed: 42

env:
  name: ur3e_ag95_pick_place
  scene_xml: assets/mujoco/tasks/pick_place_mug.xml
  sim_hz: 500
  control_hz: 20
  vector:
    num_envs: 1
    backend: sync
  cameras:
    front:
      width: 256
      height: 256
    wrist:
      width: 256
      height: 256
  randomization:
    object_pose: true
    lighting: false
    texture: false
  success:
    type: pick_place
    params:
      object_body: mug
      target_body: plate
      distance_threshold: 0.05
      stable_steps: 10

robot:
  profile: ur3e_ag95
  action_mode: joint_position
  action_dim: 7
  state_dim: 7

dataset:
  repo_id: local/ur3e_ag95_pick_place
  root: data/lerobot/ur3e_ag95_pick_place
  fps: 20
  task: "pick up the mug and place it on the plate"
  curation:
    keep_success_only: true
    max_episode_steps: 400
    balance_by:
      - object_type
      - difficulty

policy:
  type: smolvla
  path: lerobot/smolvla_base
  device: cuda
  chunk_size: 10
  action_chunk:
    prediction_horizon: 10
    execution_horizon: 5
    overlap_strategy: queue
    action_smoothing: 0.0

rollout:
  episodes: 20
  max_steps: 400
  save_video: true
  output_dir: outputs/rollouts/ur3e_ag95_smolvla

logging:
  local:
    jsonl: true
    csv: true
    videos: true
  wandb:
    enable: false
```

Robot profile 示例：

```yaml
# configs/robots/ur3e_ag95.yaml
name: ur3e_ag95
arm_dof: 6
gripper_dof: 1
action_dim: 7
state_dim: 7

mujoco:
  scene_xml: assets/mujoco/robots/ur3e_ag95/scene.xml
  end_effector_body: tool0
  gripper_actuator: gripper

control:
  action_mode: joint_position
  joint_names:
    - shoulder_pan_joint
    - shoulder_lift_joint
    - elbow_joint
    - wrist_1_joint
    - wrist_2_joint
    - wrist_3_joint
  gripper:
    open_value: 1.0
    close_value: 0.0

lerobot_features:
  images:
    front: observation.images.front
    wrist: observation.images.wrist
  state: observation.state
  action: action
```

## 5. 实现路线

### Milestone 1：项目骨架与仿真环境

目标：能加载 MuJoCo 场景，完成 reset、step、render。

交付：

- 创建基础目录结构。
- 创建 `pyproject.toml`、`requirements/` 和 `constraints/`。
- 定义 `assets/manifests/asset_sources.yaml` 格式。
- 实现 MJCF 加载器。
- 实现 `MujocoManipulationEnv`。
- 实现 `SuccessCondition` registry，并接入 `pick_place` 成功条件。
- 接入 UR3e + AG95 资产。
- 实现 `scripts/validate_assets.py`。
- 实现 `scripts/rollout_policy.py` 的 dummy policy 模式。
- 实现 headless smoke test；viewer 仅作为单实例 debug 模式。

验收：

```bash
python scripts/validate_assets.py --asset-root assets/mujoco
python scripts/rollout_policy.py --config configs/experiments/ur3e_ag95_smolvla_pick_place.yaml --policy dummy
pytest tests/test_mjcf_load.py tests/test_env_smoke.py
```

### Milestone 2：数据采集与 LeRobot 写入

目标：采集和回放可训练数据。

交付：

- 实现 scripted oracle pick-place 初始策略。
- oracle policy 阶段包括 pre-grasp、descend、close、lift、move-to-target、open、retreat。
- 实现 keyboard teleop debug 模式；SpaceMouse 或 GUI slider 只作为可选增强。
- 实现 LeRobot dataset writer。
- 实现 episode metadata 写入，包括 task variant、object pose、difficulty、seed、success 和 env config hash。
- 实现 `curate.py`，支持失败 episode 过滤、过长 episode 截断和 metadata 均衡采样。
- 实现 episode replay。
- 实现 dataset schema validate。
- 实现 `scripts/curate_dataset.py`，作为 `robot_vla_mujoco.datasets.curate` 的用户侧 CLI。

验收：

```bash
python scripts/collect_demo.py --config configs/experiments/ur3e_ag95_smolvla_pick_place.yaml --episodes 10
python scripts/replay_episode.py --dataset data/lerobot/ur3e_ag95_pick_place --episode 0 --save-video
python scripts/validate_dataset.py --dataset data/lerobot/ur3e_ag95_pick_place
python scripts/curate_dataset.py --dataset data/lerobot/ur3e_ag95_pick_place --keep-success-only
```

### Milestone 3：SmolVLA 与 pi0/pi0.5 推理闭环

目标：模型能读取 MuJoCo observation 并输出 action，完成仿真闭环。

交付：

- 实现 `PolicyAdapter`。
- 实现 `SmolVLAAdapter`。
- 实现 `Pi0Adapter` 和 `Pi05Adapter`。
- 实现 `ActionChunkBuffer` 与 `ActionDecoder`，统一单步 action、action chunk 和 horizon trajectory。
- 实现本地 JSONL/CSV metrics logger，WandB 只作为可选配置。
- 实现 rollout 视频保存。
- 实现 `SyncVectorEnv` 的 headless 批量评测入口；`AsyncVectorEnv` 只保留接口或后续任务。

验收：

```bash
python scripts/rollout_policy.py --config configs/experiments/ur3e_ag95_smolvla_pick_place.yaml
python scripts/eval_policy.py --config configs/experiments/ur3e_ag95_smolvla_pick_place.yaml --episodes 20
python scripts/eval_policy.py --config configs/experiments/ur3e_ag95_smolvla_pick_place.yaml --episodes 20 --num-envs 4
```

### Milestone 4：训练与微调

目标：使用采集数据训练或微调策略。

交付：

- `train_lerobot.py` 封装 LeRobot 训练。
- 支持 SmolVLA fine-tune。
- 支持 pi0/pi0.5 fine-tune 配置。
- 支持训练日志、checkpoint、resume。
- 支持训练时 transforms 配置，图像增强不离线改写原始 dataset。
- 支持按 episode metadata 过滤训练集和验证集。

验收：

```bash
python scripts/train_policy.py --config configs/experiments/ur3e_ag95_smolvla_pick_place.yaml
python scripts/eval_policy.py --policy outputs/train/ur3e_ag95_smolvla/checkpoints/latest
```

### Milestone 5：LingBot-VLA 后训练

目标：接入 LingBot-VLA 后训练和仿真评测。

交付：

- `requirements/lingbot.txt` 独立管理 LingBot 依赖。
- 推荐使用独立 LingBot 环境，不要求基础 MuJoCo/LeRobot 环境安装 LingBot 重依赖。
- 实现 LingBot robot config 导出。
- 实现 norm stats 计算。
- 实现 LingBot training wrapper。
- 实现 LingBot inference adapter。
- 实现 benchmark regression 报告；GPU/VLA 回归测试标记为 optional 或 nightly。

验收：

```bash
python -m robot_vla_mujoco.training.compute_norm_stats --dataset data/lerobot/ur3e_ag95_pick_place
python scripts/train_policy.py --config configs/experiments/ur3e_ag95_lingbot_posttrain.yaml
python scripts/eval_policy.py --config configs/experiments/ur3e_ag95_lingbot_posttrain.yaml
```

## 6. 测试计划

单元测试：

- `test_mjcf_load.py`：所有 scene XML 可被 `mujoco.MjModel.from_xml_path` 加载。
- `test_env_smoke.py`：环境可 reset、step、render，图像 shape 正确。
- `test_lerobot_schema.py`：采集 1 个短 episode 后，dataset features 与配置一致。
- `test_lerobot_schema.py` 同时检查 episode metadata 必填字段存在。
- `test_action_adapter.py`：模型输出能正确限幅并映射到 MuJoCo actuator。
- `test_action_adapter.py` 同时覆盖单步 action、action chunk 和 horizon trajectory。
- `test_policy_dummy.py`：dummy policy 能完成 rollout，不依赖 GPU。

集成测试：

- `collect_demo -> replay_episode -> validate_dataset`
- `collect_scripted -> curate --keep-success-only -> validate_dataset`
- `train_policy --dry-run`
- `rollout_policy --policy dummy`
- `eval_policy --episodes 3`
- `eval_policy --episodes 3 --num-envs 2`

人工验收：

- viewer 模式能看到机械臂、桌子、目标物体、相机视角。
- headless 模式能正常渲染 RGB 图像。
- replay 的图像、状态、动作时间对齐。
- 模型输出不会导致关节爆限或仿真不稳定。
- rollout 视频能保存到 `outputs/rollouts/`。

CI 计划：

- 首期 CI 使用 `.github/workflows/ci.yml` 跑静态检查、pytest、MJCF headless load 和 dummy rollout smoke test。
- 无 GPU 环境默认跳过真实 VLA 推理、训练和 LingBot 测试。
- GPU/VLA 用例使用 pytest marker 或独立 workflow 标记为 optional/nightly。
- benchmark regression 第一阶段只生成报告，不设置阻塞阈值；后续再逐步对固定 seed 的 success rate 设置阈值。

## 7. 可行性与风险

### 可行性

- MuJoCo 资产和环境初始化有可参考项目。
- LeRobot 已提供成熟的数据格式、训练入口和 VLA policy 加载方式。
- SmolVLA 是较轻量的首期 VLA baseline，适合快速完成端到端闭环。
- LingBot-VLA 官方后训练流程清晰，但适合放在第二阶段后半段或第三阶段。

### 主要风险

- 不同模型对 observation key、action dim、归一化统计的要求不同。
- 不同模型输出单步 action、action chunk 或 horizon trajectory，若不统一会导致推理闭环语义混乱。
- MuJoCo 资产的坐标系、惯量、碰撞体和 actuator 参数会显著影响训练质量。
- 资产来源、license、缩放和 mesh 转换记录如果缺失，后续难以复现和发布。
- pi0/pi0.5 和 LingBot-VLA 对 GPU 显存、依赖版本和 checkpoint 格式要求更高。
- LeRobot、MuJoCo、Torch、numpy 和 LingBot 依赖可能发生版本冲突。
- 仿真里成功的策略不一定能直接迁移到真实机械臂，后续需要 domain randomization 和 sim2real 校准。

### 降低风险的策略

- 先固定一个 robot profile：`UR3e + AG95`。
- 先固定一个任务：桌面 pick-place。
- 先固定一个 action mode：`joint_position + gripper`。
- 统一 observation/action schema，所有模型通过 adapter 对接。
- 所有模型输出先进入 `ActionChunkBuffer` 和 `ActionDecoder`，再映射到 MuJoCo control。
- 资产必须登记到 `asset_sources.yaml`，license 不清晰的资产只用于本地实验。
- 训练前强制运行 dataset schema validate。
- 采集数据保留原始版本，训练增强只在 transform 层完成。
- CI 首期只覆盖 CPU/headless smoke test，GPU/VLA 测试设为 optional。
- LingBot 相关依赖隔离到 `requirements/lingbot.txt`。

## 8. 默认假设

- 第一版默认机械臂为 `UR3e + AG95`。
- 第一版默认任务为单臂桌面 pick-place。
- 第一版默认模型路线为 `SmolVLA -> pi0/pi0.5 -> LingBot-VLA`。
- 第一版 observation 使用 front camera、wrist camera、robot state 和 language instruction。
- 第一版 action 使用 7 维 `joint_position + gripper`。
- 第一版并行评测优先使用 headless `SyncVectorEnv`，`AsyncVectorEnv` 后续实现。
- 第一版数据采集优先使用 scripted oracle，键盘 teleop 只用于 debug，SpaceMouse/GUI slider 不作为阻塞项。
- 第一版日志默认本地 JSONL/CSV/video，WandB 可选，TensorBoard/MLflow 暂不纳入首期。
- 第一版图像增强只在训练 transform 中启用，不离线重写 dataset。
- 训练数据、模型权重、rollout 视频不提交到 git。
- 后续如果加入 SO-ARM100/SO-101，应通过新增 robot profile 接入，而不是改动环境主接口。

## 9. 建议第一批文件落地顺序

1. `pyproject.toml`
2. `requirements/base.txt`
3. `constraints/core.txt`
4. `assets/manifests/asset_sources.yaml`
5. `configs/robots/ur3e_ag95.yaml`
6. `configs/envs/ur3e_ag95_pick_place.yaml`
7. `configs/policies/smolvla.yaml`
8. `configs/experiments/ur3e_ag95_smolvla_pick_place.yaml`
9. `src/robot_vla_mujoco/envs/base.py`
10. `src/robot_vla_mujoco/envs/mujoco_env.py`
11. `src/robot_vla_mujoco/envs/success_conditions.py`
12. `src/robot_vla_mujoco/policies/base.py`
13. `src/robot_vla_mujoco/policies/action_decoder.py`
14. `scripts/validate_assets.py`
15. `scripts/rollout_policy.py`
16. `scripts/curate_dataset.py`
17. `tests/test_env_smoke.py`

这个顺序能最快得到一个可运行的最小闭环，然后再向数据、训练和 LingBot 后训练扩展。
