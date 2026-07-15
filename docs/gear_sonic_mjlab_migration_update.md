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
- 新增 `gear_sonic/config/exp/manager/sonic_isaac_minimal.yaml`：
  - 用 Isaac Lab 原环境跑一个和 mjlab minimal 尽量接近的 MLP PPO 基线。
  - actor obs `160`、critic obs `286`、action `29`，便于和 mjlab minimal 做小数据训练趋势对比。
  - 这不是官方 `sonic_release` universal-token 全量配置，只用于低成本横向 smoke。

### Smoke 与 Bones 数据工具

- 新增 `gear_sonic/scripts/summarize_training_compare.py`：
  - 从 Isaac/mjlab 训练日志中提取 iteration 和 Mean rewards。
  - 输出 JSON，便于后续多 motion 训练对比复用同一套汇总口径。
- 新增 `gear_sonic/data_process/pack_reference_motion_to_mjlab_npz.py`：
  - 将 `gear_sonic_deploy/reference/example/<motion>/` 中的 deploy reference CSV 打包成 mjlab tracking `.npz`。
  - 支持 `--body-count` padding，用于 smoke test。
- 新增 `gear_sonic/data_process/convert_bones_csv_to_mjlab_npz.py`：
  - 输入单个 Bones-SEED G1 CSV。
  - 解析 root translation/rotation 和 29 DOF。
  - 通过 mjlab/MuJoCo forward kinematics 生成完整 body motion arrays。
  - 输出 mjlab tracking `.npz`。

注意：padded NPZ 只是为了验证训练链路，不是最终训练数据；Bones CSV 转换器才是后续正式数据路径的起点。


- 新增 `gear_sonic/envs/mjlab_multi_motion.py`：
  - 提供 `MultiMotionLoader` / `MultiMotionCommand`。
  - 每个 env reset 时抽取一条 motion 和局部帧，并记录该 env 当前 motion 的结束帧，避免简单拼接 NPZ 后跨 motion 边界。
  - `mjlab_env.motion_dir` / `motion_files` / `max_motions` 会自动启用多 motion 路径。

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

当前审计结果：

- `bones-seed/g1.tar.gz` 存在，大小约 22G。
- tar 包内 CSV 数：`142220`。
- `bones-seed/extracted/g1/csv` 已解压 CSV 数：`142220`。
- 0 字节 CSV 数：`0`。
- 因此 G1 CSV 下载和解压可以判定为完整。

全量官方 Step1/Step2 已完成。日志文件：

```bash
logs/data_processing/full_bones_processing_20260715_203300.log
```

最终结果：

- 官方 Step1 输出：`data/motion_lib_bones_seed/robot`，`142220` 个 PKL。
- 官方 Step2 输出：`data/motion_lib_bones_seed/robot_filtered`，`129785` 个 PKL。
- `bones_full_processing` tmux session 已结束。

局部验证目录和全量目录都已跑通官方 Step1/Step2。

## 已验证

### 环境与依赖

- `sonic` 环境已具备 SONIC trainer 依赖：Hydra/OmegaConf/TRL/Accelerate/Torch 等。
- `sonic` 环境已补齐 mjlab 关键依赖：
  - `mujoco-warp`
  - `mjviser`
  - `rsl-rl-lib==5.4.0`
- `sonic` 环境可以 import 本地 mjlab：
  - `sys.path.insert(0, "/home/wuchenghui/mjlab/src"); import mjlab`
- 官方 universal-token 依赖 `vector_quantize_pytorch` 已补齐：
  - PyPI 包名：`vector-quantize-pytorch`。
  - 代码 import 名：`vector_quantize_pytorch`。
  - 官方 `sonic_release` / `sonic_bones_seed` 配置通过 `gear_sonic/config/actor_critic/quantizers/fsq.yaml` 使用 `_target_: vector_quantize_pytorch.FSQ`，因此这是 training guide 官方脚本的必需依赖。
  - 已加入 `gear_sonic[training]` 依赖声明。

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
- 128 条对比数据已生成：
  - 输入 CSV：`data/mjlab_compare/bones_128_csv`。
  - mjlab NPZ：`data/mjlab_compare/bones_128_npz_100f`，结果 `128/128`，每条 `frames=100 joints=29 bodies=30`。
  - Isaac motion_lib PKL：`data/mjlab_compare/bones_128_motion_lib_robot_filtered`，结果 `128/128 converted`，过滤后 `128`。
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

### Isaac-vs-mjlab 单帧 FK 对齐

FK 是 forward kinematics，即给定 root 位姿和 29 DOF 关节角后，直接计算各个 body/link 的世界坐标和姿态。这个检查不涉及 policy、reward 或训练，只验证同一帧动作在 Isaac G1 和 mjlab/MuJoCo G1 上的几何结果是否一致。

当前没有手写新的 G1 XML：Isaac 路径使用仓库内 `gear_sonic/data/assets/robot_description/urdf/g1/main.urdf`，mjlab 路径使用 `/home/wuchenghui/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml`。两者来自不同格式和导入路径，Isaac importer 还会把若干 fixed/inertial links merge 到父 link。当前厘米级位置误差和小角度姿态误差更符合资产/导入器/body frame 差异；不是 joint order、单位或 body index 这类代码错误的典型表现。

新增 `gear_sonic/scripts/check_isaac_mjlab_fk_alignment.py`，已用局部 Bones partial 数据验证：

```bash
conda run -n sonic python gear_sonic/scripts/check_isaac_mjlab_fk_alignment.py \
  --bones-csv data/partial_bones_seed/g1_csv/220714/change_idle_left_to_idle_001__A025.csv \
  --mjlab-npz data/partial_bones_seed/mjlab_motions_100f/220714/change_idle_left_to_idle_001__A025.npz \
  --frame 0 \
  --device cpu \
  --headless \
  --output-json logs/data_processing/isaac_mjlab_fk_frame0.json
```

结果：

- 14 个 tracking body 全部按 body name 对齐。
- 最大位置误差：`0.01097 m`。
- 平均位置误差：`0.00461 m`。
- 最大姿态误差：`0.22117 rad`。
- 平均姿态误差：`0.10250 rad`。
- 状态：`pass`，低于当前临时阈值 `0.05 m` / `0.25 rad`。

第一次运行时曾出现约 `1 m` 级误差，原因不是引擎差异，而是检查脚本误用了 body mapping 方向。修正为按 mjlab 实际 body name 顺序索引后，对齐通过。

注意：GPU Isaac/PhysX 运行曾因当前机器显存分配失败而无法创建 PhysicsScene；CPU headless 模式可完成该单帧检查。

批量 FK 扫描也已完成，新增 `gear_sonic/scripts/check_isaac_mjlab_fk_batch.py`：

```bash
conda run -n sonic python gear_sonic/scripts/check_isaac_mjlab_fk_batch.py \
  --csv-root data/partial_bones_seed/g1_csv \
  --npz-root data/partial_bones_seed/mjlab_motions_100f \
  --limit 4 \
  --frames 0,middle,last \
  --device cpu \
  --headless \
  --output-json logs/data_processing/isaac_mjlab_fk_batch_partial.json \
  --output-csv logs/data_processing/isaac_mjlab_fk_batch_partial.csv
```

结果：

- 4 条局部 motion，每条 3 帧，共 12 个 FK checks。
- 最大位置误差：`0.02153 m`，平均位置误差均值：`0.00553 m`。
- 最大姿态误差：`0.26676 rad`，平均姿态误差均值：`0.08915 rad`。
- 位置误差全部低于 `0.05 m`；4/12 个姿态 check 略高于临时 `0.25 rad` 阈值。
- 结论：没有看到 joint order/unit/body index 错误会导致的几十厘米到米级偏差；剩余姿态差异需要按 body frame/URDF importer/MJCF 固有差异继续收紧阈值和解释。

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

4 条局部 mjlab NPZ 已逐条重跑短 rollout sanity，全部通过：

- `change_idle_left_to_idle_001__A025.npz`: zero `0.19157`，random `0.16945`，`done_sum=0`。
- `change_idle_left_to_idle_001__A025_M.npz`: zero `0.19195`，random `0.17048`，`done_sum=0`。
- `change_idle_left_to_idle_001__A026.npz`: zero `0.19819`，random `0.18380`，`done_sum=0`。
- `change_idle_left_to_idle_001__A026_M.npz`: zero `0.20016`，random `0.18913`，`done_sum=0`。

### checkpoint reload 验证

已用 `logs_rl/TRL_G1_MjLab/sonic_mjlab_minimal-20260715_161345/last.pt` 验证 checkpoint reload：

```bash
conda run -n sonic python gear_sonic/train_agent_trl.py \
  +exp=mjlab/sonic_mjlab_minimal \
  checkpoint=/home/wuchenghui/GR00T-WholeBodyControl/logs_rl/TRL_G1_MjLab/sonic_mjlab_minimal-20260715_161345/last.pt \
  num_envs=16 \
  mjlab_env.motion_file=/home/wuchenghui/GR00T-WholeBodyControl/data/partial_bones_seed/mjlab_motions_100f/220714/change_idle_left_to_idle_001__A025.npz \
  algo.config.num_steps_per_env=4 \
  algo.config.num_learning_iterations=1 \
  algo.config.num_learning_epochs=1 \
  algo.config.num_mini_batches=1 \
  use_wandb=false
```

结果：

- 日志显示 `Loaded checkpoint from step 10`。
- 完成 16 env x 4 step x 1 iteration。
- 保存新 checkpoint：`logs_rl/TRL_G1_MjLab/sonic_mjlab_minimal-20260715_192154/last.pt`。

### mjlab eval plumbing

新增 `gear_sonic/scripts/eval_mjlab_plumbing.py`，用于先验证 mjlab checkpoint eval 管线：创建 mjlab env、加载 policy/value checkpoint、用 deterministic action mean rollout、输出 reward 和 mjlab MotionCommand tracking metrics。

```bash
conda run -n sonic python gear_sonic/scripts/eval_mjlab_plumbing.py \
  --checkpoint logs_rl/TRL_G1_MjLab/sonic_mjlab_minimal-20260715_161345/last.pt \
  --motion-npz data/partial_bones_seed/mjlab_motions_100f/220714/change_idle_left_to_idle_001__A025.npz \
  --mjlab-source-path /home/wuchenghui/mjlab/src \
  --num-envs 16 \
  --steps 16 \
  --device cuda:0 \
  --output-json logs/data_processing/mjlab_eval_plumbing_partial.json
```

结果：

- checkpoint global step: `10`。
- `reward_finite: true`，`done_count: 0`。
- reward mean/min/max: `0.06203 / 0.05790 / 0.06755`。
- command metrics mean: anchor pos `0.08117`，body pos `0.06630`，joint pos `1.81204`。

这一步只是 eval plumbing，不代表官方 `success_rate/mpjpe_l/mpjpe_g` 已经完全对齐。后续需要把 mjlab metrics 输出格式进一步对齐 official eval callback。

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


### mjlab 多 motion smoke

已用 `data/partial_bones_seed/mjlab_motions_100f` 中 4 条 NPZ 验证多 motion 路径。结果：CommandManager 显示 `MultiMotionCommand`，actor obs `(160,)`、critic obs `(286,)`、action `(29,)`，完成训练并保存 checkpoint。

### Isaac-vs-mjlab 小训练对比

已完成一组单 motion、同预算、低成本 paired comparison。为了避免官方 universal-token 配置的额外依赖和特性混入，这里使用两个 minimal MLP 配置：

- Isaac: `gear_sonic/config/exp/manager/sonic_isaac_minimal.yaml`，motion 使用官方 Step1/Step2 生成的 PKL。
- mjlab: `gear_sonic/config/exp/mjlab/sonic_mjlab_minimal.yaml`，motion 使用同一个 Bones CSV 转成的 mjlab NPZ。
- motion: `220714/change_idle_left_to_idle_001__A025`。
- 预算：`num_envs=64`、`num_steps_per_env=8`、`num_learning_iterations=10`、`num_learning_epochs=1`、`num_mini_batches=1`。

运行结果：

| backend | log dir | Mean rewards trend | final Mean rewards |
| --- | --- | --- | --- |
| Isaac minimal | `logs_rl/TRL_G1_Isaac_Minimal/sonic_isaac_minimal-20260715_195714` | `0.00000 -> 1.78443` | `1.78443` |
| mjlab minimal | `logs_rl/TRL_G1_MjLab/sonic_mjlab_minimal-20260715_195923` | `0.00000 -> 1.93785` | `1.93785` |

汇总 JSON：`logs/data_processing/minimal_training_compare_64x8x10.json`。解析命令：

```bash
python gear_sonic/scripts/summarize_training_compare.py \
  --isaac-log logs/data_processing/isaac_minimal_compare_64x8x10.log \
  --mjlab-log logs/data_processing/mjlab_minimal_compare_64x8x10.log \
  --output-json logs/data_processing/minimal_training_compare_64x8x10.json
```

结论：两边在同一条 partial Bones motion 上都能完成训练，obs/action 维度一致，reward 均呈正向趋势，且没有 NaN/崩溃。这能证明 mjlab pipeline 不是只会跑空 rollout，而是能进入可学习闭环。它还不能证明和 Isaac 官方训练最终效果等价，因为当前只用了单 motion、10 iterations、minimal MLP 子集，且两边 reward/termination 实现仍有细节差异。

官方 `sonic_release` 全量配置的 Isaac 对比暂未完成：首次尝试时暴露了 `open3d` 和 `vector_quantize_pytorch` 依赖问题。`open3d` 已改为可选依赖；`vector_quantize_pytorch` 已安装并加入 training 依赖，后续可以继续验证 universal-token 路径。

### Isaac-vs-mjlab 128 motion 训练对比

已完成 128 条 motion、同预算 paired comparison：

- Isaac: `gear_sonic/config/exp/manager/sonic_isaac_minimal.yaml`，motion 使用 `data/mjlab_compare/bones_128_motion_lib_robot_filtered`。
- mjlab: `gear_sonic/config/exp/mjlab/sonic_mjlab_minimal.yaml`，motion 使用 `data/mjlab_compare/bones_128_npz_100f`，通过 `MultiMotionCommand` 采样。
- 预算：`num_envs=64`、`num_steps_per_env=8`、`num_learning_iterations=10`、`num_learning_epochs=1`、`num_mini_batches=1`。

| backend | log dir | Mean rewards trend | final Mean rewards |
| --- | --- | --- | --- |
| Isaac minimal | `logs_rl/TRL_G1_Isaac_Minimal/sonic_isaac_minimal-20260715_204443` | `0.34267 -> 1.34873` | `1.34873` |
| mjlab minimal | `logs_rl/TRL_G1_MjLab/sonic_mjlab_minimal-20260715_204329` | `0.28502 -> 1.35795` | `1.35795` |

汇总 JSON：`logs/data_processing/training_compare_128_64x8x10.json`。

结论：扩到 128 条 motion 后，两边仍然都能学习，最终 reward 非常接近，mjlab final - Isaac final 约 `0.00922`。这比单 motion smoke 更有说服力，但仍属于短训练、minimal MLP、低预算趋势对比；还不是官方 4096 env / universal-token / 长训复现。

### 官方 `sonic_release` tiny smoke

`vector_quantize_pytorch` 安装后，已运行 training guide 默认 `sonic_release` 配置的极小 smoke：

```bash
conda run -n sonic python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  num_envs=4 headless=True use_wandb=false \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/wuchenghui/GR00T-WholeBodyControl/data/mjlab_compare/bones_128_motion_lib_robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/home/wuchenghui/GR00T-WholeBodyControl/data/smpl_filtered \
  ++algo.config.num_steps_per_env=2 \
  ++algo.config.num_learning_iterations=1 \
  ++algo.config.num_learning_epochs=1 \
  ++algo.config.num_mini_batches=1
```

日志：`logs/data_processing/sonic_release_smoke_after_vector_quantize.log`。结果：

- universal-token module 初始化成功。
- FSQ quantizer 初始化成功，embedding dim `64`。
- g1 / teleop / smpl encoders 初始化成功。
- g1_dyn / g1_kin decoders 初始化成功。
- 进入 `Learning iteration 1`，完成 4 env x 2 step tiny PPO smoke。

这说明 training guide 官方配置的依赖阻塞已解除；但这仍只是 tiny smoke，不是官方收敛复现。


### 官方 `sonic_release` reload / eval / mjlab universal-token smoke

已完成完整 release 网络的第一轮 smoke 验证：

- `sonic_release/last.pt` reload smoke 已通过：日志显示 `Loaded checkpoint from step 41550`，进入 `Learning iteration 1`，无 key mismatch / size mismatch / RuntimeError。
  - 日志：`logs/data_processing/sonic_release_reload_smoke_128.log`
- `eval_agent_trl.py` metrics smoke 已通过，并显式落盘 metrics JSON。
  - 输出：`logs_eval/sonic_release_metrics_smoke/metrics_eval.json`
  - 结果：`motion_count=4`，包含 `eval/all_metrics_dict`、`mpjpe_l`、`mpjpe_g`、`accel_dist` 等字段。
  - 日志：`logs/data_processing/sonic_release_eval_metrics_smoke_with_json.log`
- mjlab universal-token finetune smoke 已通过。
  - 配置：`gear_sonic/config/exp/mjlab/sonic_mjlab_universal_smoke.yaml`
  - checkpoint：`sonic_release/last.pt`
  - 数据：`data/mjlab_compare/bones_128_npz_100f`，`max_motions=8`
  - 规模：`num_envs=4`、`num_steps_per_env=4`、`num_learning_iterations=1`
  - 网络：`UniversalTokenModule`、FSQ、G1/teleop/SMPL encoders、`g1_dyn` action decoder、`g1_kin` motion decoder 均初始化成功。
  - 当前 v1 语义：G1 tokenizer 使用真实 future motion；teleop/SMPL tokenizer 字段为 checkpoint-compatible 占位；aux loss 仅启用 `g1_recon`。
  - 输出：`logs_rl/TRL_G1_MjLab/sonic_mjlab_universal_smoke-20260715_225918/last.pt`
  - 本地训练记录：`logs_rl/TRL_G1_MjLab/sonic_mjlab_universal_smoke-20260715_225918/metrics.jsonl`
  - 记录字段包括 `loss/policy_avg`、`loss/value_avg`、`loss/entropy_avg`、`loss/aux_g1_recon_avg`、`loss/total_aux_loss_avg`。

### Isaac-vs-mjlab 128 motion 完整网络小训练对比

已完成 128 条 motion、同预算、完整 release 网络结构的小训练对比。该对比的目的不是宣称 mjlab 已和 Isaac 完全等价，而是验证完整网络能在两边加载、rollout、backward、保存，并暴露下一步语义差异。

- Isaac：`+exp=manager/universal_token/all_modes/sonic_release`，checkpoint 使用 `sonic_release/last.pt`，motion 使用 `data/mjlab_compare/bones_128_motion_lib_robot_filtered`。
- mjlab：`+exp=mjlab/sonic_mjlab_universal_smoke`，checkpoint 使用 `sonic_release/last.pt`，motion 使用 `data/mjlab_compare/bones_128_npz_100f`。
- 预算：`num_envs=16`、`num_steps_per_env=8`、`num_learning_iterations=10`、`num_learning_epochs=1`、`num_mini_batches=1`。

| backend | run dir | records | checkpoint | key observation |
| --- | --- | ---: | --- | --- |
| Isaac full-network | `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_isaac_full_128_16x8x10-20260715_230143` | 10 | `last.pt` | 10 iterations 完成，loss finite，termination 计数为 0 |
| mjlab full-network v1 | `logs_rl/TRL_G1_MjLab/sonic_mjlab_universal_smoke_mjlab_full_128_16x8x10-20260715_230631` | 10 | `last.pt`, `model_step_000010.pt` | 10 iterations 完成，loss finite，但 value/weighted PPO loss 更高，`ee_body_pos` termination 较多 |

关键指标摘要：

| backend | step | policy loss | value loss | entropy | aux g1 recon | total aux | motion/body or termination signal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Isaac | 1 | `0.00277` | `0.05317` | `13.12242` | `0.33074` | `0.01139` | `anchor_pos=0`, `ee_body_pos=0` |
| Isaac | 10 | `0.02734` | `0.00477` | `13.12160` | `0.41616` | `0.02054` | `anchor_pos=0`, `ee_body_pos=0` |
| mjlab | 1 | `-0.01246` | `0.29966` | `13.12242` | `0.52253` | `0.00523` | `error_body_pos=0.09350`, `ee_body_pos=2.66667` |
| mjlab | 10 | `0.06141` | `2.30206` | `13.12333` | `0.50630` | `0.00506` | `error_body_pos=0.17854`, `ee_body_pos=1.57143` |

汇总 JSON：`logs/data_processing/full_network_compare_128_16x8x10.json`。

结论：完整网络训练链路已能在 mjlab 上跑通，但数值趋势还不能视为和 Isaac 等价。当前 mjlab v1 的差异主要来自：critic privileged obs 仍是 padding 兼容、teleop/SMPL tokenizer 为占位、reward/termination 语义与 Isaac 尚未完全对齐。

### 完整网络 eval 对比 smoke

- Isaac finetuned checkpoint eval 已通过：
  - checkpoint：`logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_isaac_full_128_16x8x10-20260715_230143/last.pt`
  - 输出：`logs_eval/isaac_full_128_16x8x10_metrics/metrics_eval.json`
  - `motion_count=4`
  - `eval/all/mpjpe_l=28.46325`，`eval/all/mpjpe_g=589.28462`，`eval/all/accel_dist=1.40326`
- mjlab finetuned checkpoint eval plumbing 已通过：
  - checkpoint：`logs_rl/TRL_G1_MjLab/sonic_mjlab_universal_smoke_mjlab_full_128_16x8x10-20260715_230631/last.pt`
  - 输出：`logs_eval/mjlab_full_128_16x8x10_metrics.json`
  - `steps=32`，`num_envs=16`，`reward_finite=true`，`done_count=48`
  - tracking metrics mean：anchor pos `0.07692`，body pos `0.13079`，joint pos `2.60516`

这两套 eval 口径目前还不是同一指标空间：Isaac 是官方 MPJPE/accel callback；mjlab 是 MotionCommand tracking metrics。后续要把 mjlab eval 输出进一步对齐 official eval callback 的 metric schema。

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

### 4. Isaac-vs-mjlab FK 检查脚本 body mapping 方向错误

现象：首次 Isaac-vs-mjlab FK 对齐出现约 `1 m` 级位置误差，但 pelvis 和部分末端 body 又接近正常，提示更像是检查脚本的 body index 取错，而不是整体坐标系错误。

处理：改为按 mjlab 实际 `robot.body_names` 顺序索引 NPZ 中的 body arrays，避免使用方向容易混淆的 body mapping 常量。

验证：同一 frame 重跑后通过，最大位置误差 `0.01097 m`，最大姿态误差 `0.22117 rad`。批量 4 motion x 3 frame 扫描的最大位置误差为 `0.02153 m`，没有出现大尺度错位。

### 5. 官方 Isaac minimal 对比路径缺少 `open3d`

现象：尝试跑 Isaac 侧 minimal/official 路径时，`torch_humanoid_batch.py` 顶层 import `open3d`，但当前 `sonic` 环境未安装该包。该训练路径不需要 mesh FK，因此不应阻断。

处理：把 `open3d` 改为可选依赖；只有调用 `mesh_fk()` 时才要求安装 `open3d`。

验证：Isaac minimal 训练已能继续越过该 import，并完成 64 env x 8 step x 10 iterations 的对比训练。

### 6. Isaac-vs-mjlab 小训练对比已完成第一版

现象：此前只有 mjlab 单边训练趋势，无法回答“和 Isaac 原环境相比是不是至少同方向可学”。

处理：新增 Isaac minimal 配置，使用同一条 partial Bones motion 的官方 PKL，与对应 mjlab NPZ 做同预算 10-iteration 对比。

验证：Isaac Mean rewards `0.00000 -> 1.78443`，mjlab Mean rewards `0.00000 -> 1.93785`；两边都正向、无 NaN、保存 checkpoint。

### 7. `vector_quantize_pytorch` 缺失

现象：官方 `sonic_release`/`sonic_bones_seed` universal-token 配置需要 `vector_quantize_pytorch.FSQ`，但当前 `sonic` 环境缺少该包。

处理：执行 `conda run -n sonic python -m pip install vector-quantize-pytorch`，并将 `vector-quantize-pytorch` 加入 `gear_sonic[training]` 依赖。

验证：`from vector_quantize_pytorch import FSQ; FSQ(levels=[8,5,5,5])` 通过。

### 8. mjlab 多 motion 采样

现象：mjlab 原生 `MotionLoader` 只读取单个 NPZ，不能直接覆盖 100+ motion 对比。

处理：新增 SONIC 侧 `MultiMotionLoader` / `MultiMotionCommand`，并在 `create_mjlab_env()` 中支持 `mjlab_env.motion_dir`、`motion_files`、`max_motions`。

验证：4 条 NPZ smoke 通过；128 条 NPZ、64 env x 8 step x 10 iterations 训练通过。


### 9. 官方 `sonic_release` checkpoint reload 和 eval smoke

现象：此前只完成了 tiny train smoke，未确认 release checkpoint 能完整加载，也未确认官方 eval callback 能落盘 metrics。

处理：使用 `sonic_release/last.pt`、128 motion 子集和 `data/smpl_filtered` 分别跑 checkpoint reload smoke 与 `eval_agent_trl.py` metrics smoke。

验证：checkpoint reload 显示 `Loaded checkpoint from step 41550`；eval 输出 `logs_eval/sonic_release_metrics_smoke/metrics_eval.json`，包含 `eval/all_metrics_dict`。

### 10. mjlab universal-token 完整网络 smoke

现象：mjlab 之前只验证了 minimal MLP，不能说明官方 universal-token 网络、FSQ、encoder/decoder 和 PPO backward 能在 mjlab wrapper 上运行。

处理：新增 mjlab universal-token smoke 配置和 G1-only aux loss 配置；wrapper 增加 checkpoint-compatible tokenizer schema 和 actor/critic dim 兼容。

验证：`sonic_mjlab_universal_smoke-20260715_225918` 完成 1 iteration，保存 `last.pt`，生成 `metrics.jsonl`，包含 PPO loss 和 `loss/aux_g1_recon_avg`。

### 11. 无 wandb 时缺少本地 loss 记录

现象：`use_wandb=false` 时 stdout 只显示 pretty training table，PPO/aux loss 没有稳定落盘。

处理：在 `TRLPPOTrainer.log()` 中追加本地 `metrics.jsonl` 写入，不依赖 wandb/API。

验证：mjlab universal-token smoke 和 Isaac/mjlab full-network 128 motion 对比均生成 `metrics.jsonl`，包含 `loss/policy_avg`、`loss/value_avg`、`loss/entropy_avg`、`loss/aux_g1_recon_avg`、`loss/total_aux_loss_avg`。

## 已知问题

### 1. seed 逻辑暂时保守

`create_mjlab_env()` 当前只有显式设置 `mjlab_env.seed` 时才传给 mjlab。原因是 smoke 阶段先避免额外触发 reset/randomization 分支。

等多环境和正式 motion 转换稳定后，需要重新恢复和验证可复现 seed。

### 2. mjlab full-network v1 仍未和 Isaac 语义等价

完整网络已经能在 mjlab 上加载、训练、保存和 eval plumbing，但当前 v1 仍有三类语义差异：critic privileged obs 仍是 padding 兼容，teleop/SMPL tokenizer 字段仍是占位，reward/termination/eval metric 口径尚未完全对齐 Isaac。128 motion full-network 小训练中，mjlab 的 value/weighted PPO loss 明显高于 Isaac，且 `ee_body_pos` termination 较多。

## 下一步

### P0：修正 mjlab full-network 语义差异

当前主线已经从“能否跑通”转为“是否和 Isaac Lab 训练行为一致”。下一步优先修正完整网络 v1 的已知差异：

- 把 mjlab `critic_obs` 从 padding 兼容改成等价 `privileged_mf_hist`，覆盖 command、anchor、body pos/ori、base vel、joint pos/vel、actions history。
- 把 teleop/SMPL tokenizer 占位替换为真实数据源；在这之前，禁止把当前 v1 的 full-network 结果解释为官方 SONIC 等价训练。
- 在 G1 tokenizer 里补齐 anchor orientation、root height、future frame dt 语义，而不只做 shape-compatible。
- 重新启用官方全量 aux loss：`g1_smpl_latent`、`g1_teleop_latent`、`teleop_smpl_latent`、`reencoded_smpl_g1_latent`，并观察 `metrics.jsonl` 中各项 loss 是否 finite。

### P1：过程级 Isaac-vs-mjlab 对齐

- 对同一批 100+ motion 抽样，逐项对比 Isaac 和 mjlab 的 obs shape、tokenizer slice、action dim、reward term、termination term。
- 对 `ee_body_pos` termination 较高的 mjlab episodes，记录 motion key、frame、anchor/body/joint error，定位是 motion conversion、body order、reward scale 还是 termination 阈值差异。
- 扩大 FK/body pose 检查，覆盖手臂、腰、脚踝大幅动作，并把姿态误差分解到具体 body frame。

### P2：扩大 full-network 训练对比

- 先保持 128 motion，跑 `num_envs=64`、`num_steps_per_env=24`、`num_learning_iterations=50-100`。
- 每次只改一个语义模块，跑 smoke + 10 iteration + 50 iteration，避免越改越偏。
- 对比 `metrics.jsonl`：reward、episode length、done count、policy loss、value loss、entropy、aux loss。
- Isaac 侧继续用 `eval_agent_trl.py` 产出 `metrics_eval.json`；mjlab 侧逐步把 eval plumbing 对齐到相同 metric schema。

### P3：文档和命令固化

- 把 full-network smoke、128 motion 对比、eval JSON 路径整理成固定命令块。
- 为 mjlab v1 明确标注“G1 tokenizer 真实，teleop/SMPL 占位”的限制。
- 后续只有当 critic/tokenizer/reward/eval 全部对齐后，才进入长训和官方效果复现阶段。

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


运行 Isaac-vs-mjlab 单帧 FK 对齐：

```bash
conda run -n sonic python gear_sonic/scripts/check_isaac_mjlab_fk_alignment.py \
  --bones-csv data/partial_bones_seed/g1_csv/220714/change_idle_left_to_idle_001__A025.csv \
  --mjlab-npz data/partial_bones_seed/mjlab_motions_100f/220714/change_idle_left_to_idle_001__A025.npz \
  --frame 0 \
  --device cpu \
  --headless \
  --output-json logs/data_processing/isaac_mjlab_fk_frame0.json
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
