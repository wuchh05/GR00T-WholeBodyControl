# GEAR-SONIC MuJoCo/mjlab 迁移更新记录

更新时间：2026-07-14

## 目标

第一阶段目标是先绕过 Isaac Lab/Isaac Sim，使用 NVIDIA GPU + MuJoCo/mjlab 跑通 GEAR-SONIC 的最小训练闭环。

当前不处理晟腾 NPU，也不追求官方大规模并行训练复现。成功标准先定义为：

- 可以在 `sonic` conda 环境中同时 import SONIC trainer 和 mjlab。
- 可以用 mjlab G1 flat tracking env 创建环境、reset、step。
- 可以通过 `gear_sonic/train_agent_trl.py` 进入 PPO rollout/learning。
- 可以保存 checkpoint。

## 已完成

### 训练入口解耦

- `gear_sonic/train_agent_trl.py` 已支持 `sim_type: mjlab`。
- mjlab 路径会跳过 Isaac AppLauncher / Isaac Lab 顶层 import。
- 保留 SONIC 的 TRL/PPO trainer、actor/critic、配置体系。

### mjlab 环境接入

- 新增 `gear_sonic/envs/mjlab_env.py`：
  - 使用 mjlab 自带 `unitree_g1_flat_tracking_env_cfg()` 作为第一阶段 G1 flat tracking 基底。
  - 支持从配置传入 `mjlab_env.motion_file`。
  - 支持 `mjlab_env.source_path` 指向本地 `/home/wuchenghui/mjlab/src`。

### SONIC wrapper 适配

- 新增/更新 `gear_sonic/envs/wrapper/mjlab_sonic_env_wrapper.py`：
  - 把 mjlab observation group `actor`/`critic` 转成 SONIC 期望的 `actor_obs`/`critic_obs`。
  - 兼容 mjlab 的 `(obs, reward, terminated, truncated, extras)` step 返回。
  - 补齐 SONIC trainer 需要的 `infos["episode"]`、`infos["time_outs"]`、`infos["to_log"]`。
  - 支持 TensorDict/mapping 风格的 policy output。
  - 对 `num_envs=1` 时的 1D action 自动补 batch 维。
  - 对 action shape 做显式校验，避免错误一路传到 mjlab 内部。

### 最小配置

- 新增 `gear_sonic/config/exp/mjlab/sonic_mjlab_minimal.yaml`：
  - `sim_type: mjlab`
  - 默认小规模 smoke 配置。
  - 第一阶段只保留 G1 flat tracking、actor/critic 基础观测。

### Smoke 数据工具

- 新增 `gear_sonic/data_process/pack_reference_motion_to_mjlab_npz.py`：
  - 将 `gear_sonic_deploy/reference/example/<motion>/` 中的 deploy reference CSV 打包成 mjlab tracking `.npz`。
  - 支持 `--body-count` padding，用于 smoke test。

注意：这个工具和生成的 padded NPZ 只是为了验证训练链路，不是最终训练数据转换方案。正式方案仍需要从 Bones-SEED CSV 经 mjlab/MuJoCo FK 生成完整 body motion。

## 已验证

### 环境与依赖

- `sonic` 环境已具备 SONIC trainer 依赖：Hydra/OmegaConf/TRL/Accelerate/Torch 等。
- `sonic` 环境已补齐 mjlab 关键依赖：
  - `mujoco-warp`
  - `mjviser`
  - `rsl-rl-lib==5.4.0`
- `sonic` 环境可以 import 本地 mjlab：
  - `sys.path.insert(0, "/home/wuchenghui/mjlab/src"); import mjlab`

### 数据

- 从正在下载的 Bones-SEED 压缩包中抽出过一个小 CSV：
  - `data/mjlab_smoke/bones_csv/warm_up_neck_001__A360_M.csv`
- 基于仓库已有 deploy reference 生成过 smoke NPZ：
  - `data/mjlab_smoke/motions/macarena_001__A545_M_padded128.npz`
- NPZ key/shape 检查通过：
  - `joint_pos`: `(1375, 29)`
  - `joint_vel`: `(1375, 29)`
  - `body_pos_w`: `(1375, 128, 3)`
  - `body_quat_w`: `(1375, 128, 4)`
  - `body_lin_vel_w`: `(1375, 128, 3)`
  - `body_ang_vel_w`: `(1375, 128, 3)`

### mjlab raw env

- 在 `mujoco` conda 环境中，raw mjlab env + padded NPZ 支持：
  - `num_envs=2`
  - CUDA device
  - reset
  - zero-action step
  - reward finite

### SONIC wrapper

- 在 `sonic` conda 环境中，mjlab wrapper 支持：
  - `num_envs=1`
  - `reset_all()`
  - `step({"actions": ...})`
  - `actor_obs`: `(1, 160)`
  - `critic_obs`: `(1, 286)`
  - reward finite

### 训练 smoke

已通过最小 PPO smoke：

```bash
conda run -n sonic python gear_sonic/train_agent_trl.py \
  +exp=mjlab/sonic_mjlab_minimal \
  num_envs=1 \
  mjlab_env.motion_file=/home/wuchenghui/GR00T-WholeBodyControl/data/mjlab_smoke/motions/macarena_001__A545_M_padded128.npz \
  algo.config.num_steps_per_env=2 \
  algo.config.num_learning_iterations=1 \
  algo.config.num_learning_epochs=1 \
  algo.config.num_mini_batches=1
```

结果：

- 进入 `TRLPPOTrainer.train()`。
- 完成 rollout 和 learning。
- 打印 mjlab reward/termination/metric 日志。
- 保存 checkpoint：
  - `logs_rl/TRL_G1_MjLab/sonic_mjlab_minimal-20260714_205316/last.pt`


多环境 smoke 已新增验证：

```bash
conda run -n sonic python gear_sonic/train_agent_trl.py \
  +exp=mjlab/sonic_mjlab_minimal \
  num_envs=16 \
  mjlab_env.motion_file=/home/wuchenghui/GR00T-WholeBodyControl/data/mjlab_smoke/motions/macarena_001__A545_M_padded128.npz \
  algo.config.num_steps_per_env=2 \
  algo.config.num_learning_iterations=1 \
  algo.config.num_learning_epochs=1 \
  algo.config.num_mini_batches=1
```

结果：通过，并保存 checkpoint 到 `logs_rl/TRL_G1_MjLab/sonic_mjlab_minimal-20260714_212506/last.pt`。

## 已知问题

### 1. `sonic` 环境中 joint limit buffer broadcast 差异

已定位到 `sonic` 环境中 mjlab 多环境初始化存在一个 buffer broadcast 差异：

```text
soft_joint_pos_limits: (1, 29, 2)
joint_pos_limits: (1, 29, 2)
```

而 `num_envs=2` 时 reset 会用 env ids `[0, 1]` 索引这些 limit buffer，导致越界。`mujoco` 环境中对应 shape 是 `(2, 29, 2)`。

当前已在 `MjlabSonicEnvWrapper` 初始化时做兼容修复：如果这些 joint limit buffer 第一维为 1 且 `num_envs > 1`，则 repeat 到 `num_envs`。

已验证：

- `sonic` 环境 `num_envs=2` wrapper reset/step 通过。
- `sonic` 环境 `num_envs=2` training smoke 通过。
- `sonic` 环境 `num_envs=16` training smoke 通过。

这个修复是兼容层，不是 mjlab 上游根因修复。后续仍建议确认是否由 PyTorch/mujoco-warp 版本组合导致。

### 2. 当前 smoke motion 不是正式训练数据

`macarena_001__A545_M_padded128.npz` 是为了避开 mjlab body index 越界而构造的临时数据：

- 前 14 个 body 来自 deploy reference。
- 其余 body 用 body 0 padding。

这足够验证代码链路，但不适合真实训练。

正式训练数据需要从 Bones-SEED G1 CSV 生成完整 mjlab motion NPZ，包括完整 robot body positions/quats/velocities。

### 3. seed 逻辑暂时保守

`create_mjlab_env()` 当前只有显式设置 `mjlab_env.seed` 时才传给 mjlab。原因是 smoke 阶段先避免额外触发 reset/randomization 分支。

等多环境和正式 motion 转换稳定后，需要重新恢复和验证可复现 seed。

## 下一步

### P0：实现正式 Bones CSV -> mjlab NPZ 转换

目标：

- 输入 Bones-SEED G1 CSV。
- 输出 mjlab tracking motion NPZ。
- body arrays 来自 MuJoCo/mjlab FK，而不是 padding。

输出必须包含：

- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

并增加断言：

- 29 DOF。
- mjlab G1 joint order 对齐。
- tracking body names 与 `unitree_g1_flat_tracking_env_cfg()` 一致。

### P1：扩大并行训练 smoke

已完成：

- `num_envs=2`，1 iteration 通过。
- `num_envs=16`，1 iteration 通过。

下一档目标：

- `num_envs=64`，10 iterations 不 NaN。
- checkpoint reload 后可以继续 rollout。
- 在正式 motion NPZ 上重复上述 smoke。

### P2：逐步恢复 SONIC 特性

在基础 PPO/mjlab 链路稳定后，再逐步加入：

- multi-future command observations
- tokenizer observation group
- universal-token actor / aux loss
- adaptive sampling
- hand/object/camera/VLA 相关能力

## 当前可复现命令

生成 smoke NPZ：

```bash
python gear_sonic/data_process/pack_reference_motion_to_mjlab_npz.py \
  --input-dir gear_sonic_deploy/reference/example/macarena_001__A545_M \
  --output data/mjlab_smoke/motions/macarena_001__A545_M_padded128.npz \
  --fps 50 \
  --body-count 128
```

单环境训练 smoke：

```bash
conda run -n sonic python gear_sonic/train_agent_trl.py \
  +exp=mjlab/sonic_mjlab_minimal \
  num_envs=1 \
  mjlab_env.motion_file=/home/wuchenghui/GR00T-WholeBodyControl/data/mjlab_smoke/motions/macarena_001__A545_M_padded128.npz \
  algo.config.num_steps_per_env=2 \
  algo.config.num_learning_iterations=1 \
  algo.config.num_learning_epochs=1 \
  algo.config.num_mini_batches=1
```
