# GEAR-SONIC MuJoCo/mjlab 迁移更新记录

更新时间：2026-07-15

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

### Smoke 与 Bones 数据工具

- 新增 `gear_sonic/data_process/pack_reference_motion_to_mjlab_npz.py`：
  - 将 `gear_sonic_deploy/reference/example/<motion>/` 中的 deploy reference CSV 打包成 mjlab tracking `.npz`。
  - 支持 `--body-count` padding，用于 smoke test。
- 新增 `gear_sonic/data_process/convert_bones_csv_to_mjlab_npz.py`：
  - 输入单个 Bones-SEED G1 CSV。
  - 解析 root translation/rotation 和 29 DOF。
  - 通过 mjlab/MuJoCo forward kinematics 生成完整 body motion arrays。
  - 输出 mjlab tracking `.npz`。

注意：padded NPZ 只是为了验证训练链路，不是最终训练数据；Bones CSV 转换器才是后续正式数据路径的起点。


### 数据格式判断：官方 motion_lib PKL vs mjlab NPZ

- Bones-SEED G1 CSV 是源数据：包含 root translation/rotation 和 29 DOF retargeted robot joint motion。
- 官方 `convert_soma_csv_to_motion_lib.py` 的输出是 SONIC motion_lib PKL，服务于原 Isaac/SONIC motion library 路径。
- mjlab tracking env 不直接读取 motion_lib PKL；当前 mjlab loader 需要 `.npz`，至少包含 `joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`。
- 因此官方转换脚本仍应保留给 Isaac/原 SONIC 路径，但 mjlab 路径需要独立的 Bones CSV -> mjlab tracking NPZ 转换。
- smoke 记录需要明确数据来源：早期 padded smoke 使用 deploy reference CSV，不是官方 Bones motion_lib；后续 full-body smoke 使用一个 Bones CSV 经 mjlab FK 转成 NPZ。


## 当前数据处理任务

已按官方训练指南启动一个顺序任务，不停止现有 Hugging Face 下载进程：

1. 解压 `bones-seed/g1.tar.gz` 到 `bones-seed/extracted/g1/csv`。
2. 执行官方 Step1：`convert_soma_csv_to_motion_lib.py`，输出到 `data/motion_lib_bones_seed/robot`。
3. 执行官方 Step2：`filter_and_copy_bones_data.py`，输出到 `data/motion_lib_bones_seed/robot_filtered`。

启动命令：

```bash
mkdir -p bones-seed/extracted data/motion_lib_bones_seed
tar -xzf bones-seed/g1.tar.gz -C bones-seed/extracted
conda run -n sonic python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
  --input bones-seed/extracted/g1/csv \
  --output data/motion_lib_bones_seed/robot \
  --fps 30 --fps_source 120 --individual --num_workers 16
conda run -n sonic python gear_sonic/data_process/filter_and_copy_bones_data.py \
  --source data/motion_lib_bones_seed/robot \
  --dest data/motion_lib_bones_seed/robot_filtered \
  --workers 16
```

截至记录时，全量任务仍处于解压阶段，尚未进入官方 Step1；局部验证目录已用当前已解压 CSV 跑通官方 Step1/Step2。

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

- 新增第一层 mjlab 对齐检查脚本：
  - `gear_sonic/scripts/check_mjlab_alignment.py`
  - 不依赖 Isaac Lab，可在 `sonic` 环境运行。
  - 检查 Bones CSV 必需列、IsaacLab->MuJoCo DOF 映射、mjlab 29 joint lookup、14 tracking body lookup、anchor body、NPZ shape/finite。
- 对齐检查已通过：
  - Bones CSV: `data/mjlab_smoke/bones_csv/warm_up_neck_001__A360_M.csv`
  - mjlab NPZ: `data/mjlab_smoke/motions_batch/warm_up_neck_001__A360_M.npz`
  - mjlab joint indexes: `0..28`
  - tracking bodies: 14 个，anchor body 为 `torso_link`。
- 局部官方数据处理已跑通：
  - 从 `bones-seed/extracted/g1/csv` 复制 32 个已解压 CSV 到 `data/partial_bones_seed/g1_csv`。
  - 官方 Step1 输出：`data/partial_bones_seed/motion_lib_robot`，结果 `32/32 converted, 0 failed`。
  - 官方 Step2 输出：`data/partial_bones_seed/motion_lib_robot_filtered`，结果 `32/32 copied, 0 filtered out`。
- 局部 mjlab FK NPZ 已生成：
  - 输入：`data/partial_bones_seed/g1_csv`。
  - 输出：`data/partial_bones_seed/mjlab_motions_100f`。
  - 参数：`--limit 4 --max-output-frames 100`。
  - 结果：4 条 NPZ，每条 `frames=100 joints=29 bodies=30`。
- 局部对齐检查已通过：
  - CSV: `data/partial_bones_seed/g1_csv/220714/change_idle_left_to_idle_001__A025.csv`。
  - NPZ: `data/partial_bones_seed/mjlab_motions_100f/220714/change_idle_left_to_idle_001__A025.npz`。
  - mjlab joint indexes: `0..28`，tracking bodies: 14 个，anchor body: `torso_link`。
- 从正在下载的 Bones-SEED 压缩包中抽出过一个小 CSV：
  - `data/mjlab_smoke/bones_csv/warm_up_neck_001__A360_M.csv`
- 基于仓库已有 deploy reference 生成过 smoke NPZ：
  - `data/mjlab_smoke/motions/macarena_001__A545_M_padded128.npz`
- 基于已抽样 Bones CSV 生成过 full-body FK NPZ：
  - `data/mjlab_smoke/motions/warm_up_neck_001__A360_M_mjlab_fk_100f.npz`
  - shape: `joint_pos (100, 29)`, `body_pos_w (100, 30, 3)`
  - finite 检查通过。
- 批量目录模式已通过 smoke：
  - 输入目录：`data/mjlab_smoke/bones_csv`
  - 输出：`data/mjlab_smoke/motions_batch/warm_up_neck_001__A360_M.npz`
  - 参数：`--limit 1 --max-output-frames 10 --skip-existing`
  - shape: `joint_pos (10, 29)`, `body_pos_w (10, 30, 3)`
  - finite 检查通过。
- padded NPZ key/shape 检查通过：
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

正式 FK NPZ 也已通过 `num_envs=16` smoke：

```bash
conda run -n sonic python gear_sonic/train_agent_trl.py \
  +exp=mjlab/sonic_mjlab_minimal \
  num_envs=16 \
  mjlab_env.motion_file=/home/wuchenghui/GR00T-WholeBodyControl/data/mjlab_smoke/motions/warm_up_neck_001__A360_M_mjlab_fk_100f.npz \
  algo.config.num_steps_per_env=2 \
  algo.config.num_learning_iterations=1 \
  algo.config.num_learning_epochs=1 \
  algo.config.num_mini_batches=1
```

结果：通过，并保存 checkpoint 到 `logs_rl/TRL_G1_MjLab/sonic_mjlab_minimal-20260714_222448/last.pt`。由于 rollout 只有 2 step，episode 日志 buffer 可能为空，`Mean length` 可出现 `nan`；后续应使用更长 rollout 做稳定性验证。

### 短 rollout sanity

新增 `gear_sonic/scripts/check_mjlab_rollout_sanity.py`，已在局部 Bones/mjlab NPZ 上验证：

```bash
conda run -n sonic python gear_sonic/scripts/check_mjlab_rollout_sanity.py \
  --motion-npz data/partial_bones_seed/mjlab_motions_100f/220714/change_idle_left_to_idle_001__A025.npz \
  --mjlab-source-path /home/wuchenghui/mjlab/src \
  --device cuda:0 \
  --num-envs 16 \
  --steps 3
```

结果：

- reset: `actor_obs (16, 160)`, `critic_obs (16, 286)`。
- zero action 3 step: reward finite，`done_sum=0`。
- random action 3 step: reward finite，`done_sum=0`。

### 小训练趋势验证

局部 Bones/mjlab NPZ 已通过 `num_envs=64`、`num_steps_per_env=8`、`10 iterations`：

```bash
conda run -n sonic python gear_sonic/train_agent_trl.py \
  +exp=mjlab/sonic_mjlab_minimal \
  num_envs=64 \
  mjlab_env.motion_file=/home/wuchenghui/GR00T-WholeBodyControl/data/partial_bones_seed/mjlab_motions_100f/220714/change_idle_left_to_idle_001__A025.npz \
  algo.config.num_steps_per_env=8 \
  algo.config.num_learning_iterations=10 \
  algo.config.num_learning_epochs=1 \
  algo.config.num_mini_batches=1 \
  use_wandb=false
```

结果：

- 跑满 10 iterations。
- 保存 checkpoint：`logs_rl/TRL_G1_MjLab/sonic_mjlab_minimal-20260715_161345/model_step_000010.pt` 和 `last.pt`。
- reward/metrics/loss 未出现 NaN；mean rewards 从约 `0.61` 增至约 `1.94`。
- 仍可见短 rollout 初期 `Mean length: nan` / empty-slice warning，这是 episode buffer 初期为空导致的日志问题，不阻断训练。

## 已解决问题

### 1. `sonic` 环境中 joint limit buffer broadcast 差异

现象：`sonic` 环境里 mjlab 多环境初始化时，部分 joint limit buffer 第一维为 1，例如：

```text
soft_joint_pos_limits: (1, 29, 2)
joint_pos_limits: (1, 29, 2)
```

`num_envs > 1` reset 时会按 env ids 索引这些 buffer，导致越界。

处理：`MjlabSonicEnvWrapper` 初始化时，如果这些 joint limit buffer 第一维为 1 且 `num_envs > 1`，则 repeat 到 `num_envs`。

验证：

- `num_envs=2` wrapper reset/step 通过。
- `num_envs=16` training smoke 通过。
- `num_envs=64`、10 iterations 小训练趋势通过。

### 2. padded smoke motion 不是正式训练数据

早期 `macarena_001__A545_M_padded128.npz` 只用于验证链路，body 维度经过 padding，不适合真实训练。

处理：已新增 Bones CSV -> mjlab FK NPZ 转换器，并用局部 Bones CSV 生成 full-body NPZ。

验证：

- 单条 full-body FK NPZ：`frames=100 joints=29 bodies=30`。
- 局部 4 条 full-body FK NPZ 已生成。
- 局部 full-body NPZ 已通过对齐检查、短 rollout、`num_envs=64` 小训练趋势。

### 3. episode info keys 不一致导致 trainer 聚合崩溃

现象：`num_envs=64`、10 iterations 初次运行时，训练完成前几轮后在 `process_ep_infos()` 里因为部分 episode 缺少 `Episode_Reward/...` key 而触发 `KeyError`。

处理：`process_ep_infos()` 改为对所有 episode info 的 key 取并集，单个 episode 缺 key 时跳过该 key，而不是中断训练。

验证：修复后同一命令跑满 10 iterations 并保存 checkpoint。

## 已知问题

### 1. Isaac-vs-mjlab 单帧 FK 对齐尚未完成

当前已经完成 mjlab 内部的 joint/body/order/NPZ shape 检查，但还没有在 Isaac Lab 中对同一 Bones frame 做 forward kinematics 并与 mjlab body pose 数值对比。

这是判断 Isaac 训练效果能否迁移到 mjlab 的关键验证项之一。

### 2. seed 逻辑暂时保守

`create_mjlab_env()` 当前只有显式设置 `mjlab_env.seed` 时才传给 mjlab。原因是 smoke 阶段先避免额外触发 reset/randomization 分支。

等多环境和正式 motion 转换稳定后，需要重新恢复和验证可复现 seed。

### 3. mjlab 当前只接单个 NPZ motion file

`mjlab.tasks.tracking.mdp.MotionLoader` 当前通过 `np.load(motion_file)` 读取单个 NPZ。局部小训练趋势使用单条 motion；后续如果要多 motion 采样，需要扩展 loader 或在外层合并/采样 motion。

## 下一步

### P0：完成 Isaac-vs-mjlab 单帧 FK 对齐

当前已完成 mjlab 内部顺序检查，但还需要在 Isaac Lab 可运行环境下做同一帧对比：

- 输入同一 Bones CSV frame 的 root pose + 29 DOF。
- Isaac 和 mjlab 分别 forward kinematics。
- 对比 `torso_link`、左右 ankle、左右 wrist、head/upper body 等关键 body pose。
- 输出 position/orientation error，并设定可接受阈值。

### P1：扩大局部数据和训练验证

已完成：

- 局部 32 CSV 官方 Step1/Step2。
- 局部 4 条 mjlab full-body NPZ。
- `num_envs=64`、10 iterations 单 motion 小训练趋势。

下一档目标：

- 转换 50-100 条局部 mjlab NPZ。
- 支持多 motion 采样或先逐条跑 smoke。
- checkpoint reload 后继续 rollout。
- 比较不同 motion 的 reward/error/termination 分布，确认不是只在单条 idle motion 上可跑。

### P2：实现 mjlab eval 指标

对齐官方 training guide 的 eval 口径：

- success rate。
- local/global MPJPE。
- render video。

初期不要求达到官方收敛指标，但要能稳定计算并用于 Isaac-vs-mjlab 横向比较。

### P3：逐步恢复 SONIC 特性

在基础 PPO/mjlab 链路和正确性验证稳定后，再逐步加入：

- multi-future command observations。
- tokenizer observation group。
- universal-token actor / aux loss。
- adaptive sampling。
- hand/object/camera/VLA 相关能力。

## 当前可复现命令

生成 padded smoke NPZ：

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


运行短 rollout sanity：

```bash
conda run -n sonic python gear_sonic/scripts/check_mjlab_rollout_sanity.py \
  --motion-npz data/partial_bones_seed/mjlab_motions_100f/220714/change_idle_left_to_idle_001__A025.npz \
  --mjlab-source-path /home/wuchenghui/mjlab/src \
  --device cuda:0 \
  --num-envs 16 \
  --steps 3
```

运行局部小训练趋势：

```bash
conda run -n sonic python gear_sonic/train_agent_trl.py \
  +exp=mjlab/sonic_mjlab_minimal \
  num_envs=64 \
  mjlab_env.motion_file=/home/wuchenghui/GR00T-WholeBodyControl/data/partial_bones_seed/mjlab_motions_100f/220714/change_idle_left_to_idle_001__A025.npz \
  algo.config.num_steps_per_env=8 \
  algo.config.num_learning_iterations=10 \
  algo.config.num_learning_epochs=1 \
  algo.config.num_mini_batches=1 \
  use_wandb=false
```

运行第一层 mjlab 对齐检查：

```bash
conda run -n sonic python gear_sonic/scripts/check_mjlab_alignment.py \
  --bones-csv data/mjlab_smoke/bones_csv/warm_up_neck_001__A360_M.csv \
  --motion-npz data/mjlab_smoke/motions_batch/warm_up_neck_001__A360_M.npz \
  --mjlab-source-path /home/wuchenghui/mjlab/src \
  --device cuda:0
```

生成 Bones full-body FK NPZ：

```bash
conda run -n sonic python gear_sonic/data_process/convert_bones_csv_to_mjlab_npz.py \
  --input data/mjlab_smoke/bones_csv/warm_up_neck_001__A360_M.csv \
  --output data/mjlab_smoke/motions/warm_up_neck_001__A360_M_mjlab_fk_100f.npz \
  --input-fps 120 \
  --output-fps 50 \
  --device cuda:0 \
  --max-output-frames 100 \
  --mjlab-source-path /home/wuchenghui/mjlab/src
```
