# RWM-U Training Setup

This repo uses the stable engineering path: keep heavy RWM-U training separate
from SONIC/Isaac when scaling up. For the current integration smoke,
`rwm_env.backend=rwmu` directly imports `rsl_rl_rwm` to load the local dynamics
checkpoint. For production fine-tuning, prefer exporting the trained RWM-U model
as an inference-only artifact and loading that artifact from SONIC/Isaac.

Current status:

- Upstream RWM-U bundled-data smoke training runs.
- SONIC `+exp=rwm/sonic_release` smoke runs without Isaac using placeholder dynamics.
- `rwm_env.backend=rwmu` now loads SONIC RWM-U dynamics checkpoints produced by
  `train_sonic_rwmu_dynamics.py`; this is an integration loader, not yet a
  quality-guaranteed simulator.
- Policy rollout export supports RWM smoke, mjlab, and Isaac AppLauncher startup;
  real Isaac export must be run inside an Isaac Lab environment and pass strict
  physical-field validation.

## 1. Clone

```bash
git clone <your-repo-url> GR00T-WholeBodyControl
cd GR00T-WholeBodyControl

git clone https://github.com/leggedrobotics/robotic_world_model.git external_dependencies/robotic_world_model
git clone https://github.com/leggedrobotics/rsl_rl_rwm.git external_dependencies/rsl_rl_rwm
```

## 2. SONIC / Isaac Environment

Use this environment for original SONIC training, Isaac rollout export, policy
evaluation, and final SONIC policy fine-tuning.

```bash
conda create -n sonic-isaac python=3.11 -y
conda activate sonic-isaac

# Install Isaac Lab / Isaac Sim first, following the SONIC/Isaac Lab guide.

pip install -e "gear_sonic[training,rwm]"
```

Verify SONIC-side config and RWM-U field coverage:

```bash
python gear_sonic/scripts/validate_rwm_config_parity.py \
  --num-envs 4096 --checkpoint sonic_release/last.pt

python gear_sonic/scripts/validate_sonic_rwmu_fields.py
```

Optional SONIC no-Isaac smoke. This uses placeholder dynamics, not a real RWM-U
checkpoint:

```bash
python gear_sonic/train_agent_trl.py \
  +exp=rwm/sonic_release \
  +checkpoint=sonic_release/last.pt \
  num_envs=2 headless=True \
  algo.config.num_steps_per_env=2 \
  algo.config.num_learning_iterations=1 \
  algo.config.num_learning_epochs=1 \
  algo.config.num_mini_batches=1 \
  use_wandb=false
```

## 3. RWM-U Environment

Use this environment for RWM-U dynamics/offline imagination training and later
RWM-U inference export. This environment owns `rsl_rl_rwm`.

```bash
conda create -n rwmu python=3.11 -y
conda activate rwmu

# Pick the PyTorch/CUDA build that matches the machine. Example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -e "gear_sonic[rwm]"
pip install -e external_dependencies/robotic_world_model/source/mbrl
pip install -e external_dependencies/rsl_rl_rwm
```

Verify upstream RWM-U bundled-data smoke:

```bash
python gear_sonic/scripts/run_rwmu_offline_smoke.py \
  --device cuda \
  --num-envs 2 \
  --num-steps-per-env 2 \
  --max-iterations 1 \
  --max-episode-length 8 \
  --output-dir /tmp/rwmu_offline_smoke
```

Expected output:

```text
RWM-U offline smoke OK: /tmp/rwmu_offline_smoke/policy_0.pt
```

The bundled RWM-U data is upstream ANYmal-D data, not SONIC humanoid data. It is
only for checking that the RWM-U training stack is installed correctly.

## 4. Data Format In Plain Terms

RWM-U training data is a simulator diary. Each row is one rollout time step:

```text
current robot state | action applied | reward/extra labels | contact labels | failure labels
```

For the upstream ANYmal-D smoke data, one row is 45 state numbers, 12 action
numbers, 0 extension numbers, 8 contact numbers, and 1 termination number.

For SONIC, the row is wider. Example: `joint_pos` has 29 numbers because the
release policy controls 29 body joints; `body_pos_w` has `14 x 3` numbers because
SONIC tracks 14 body links and each link has xyz world position.

Do not train RWM-U to predict noisy `actor_obs` directly. RWM-U should predict
clean robot state/contact/failure. The SONIC adapter then rebuilds `actor_obs`,
`critic_obs`, reward, and done from those predictions.

## 5. Real Training Path

The intended stable path is:

1. In `sonic-isaac`, export real SONIC/Isaac transitions.
2. In `rwmu`, train RWM-U on the exported SONIC dataset.
3. In `rwmu`, export the trained dynamics model as an inference-only artifact,
   for example TorchScript, ONNX, or a small PyTorch module plus state dict.
4. In `sonic-isaac`, load that artifact from `gear_sonic/envs/rwm_env.py` and run
   SONIC PPO with `+exp=rwm/sonic_release rwm_env.backend=rwmu`.

Step 4 is now wired for smoke/integration testing. The remaining research gate is
model quality: train on real SONIC/Isaac rollouts, validate holdout n-step error,
and only then use the backend for meaningful fine-tuning.


## Isaac/PhysX-Only RWM-U Contract

The RWM-U dataset now separates the physics model from the SONIC MDP managers.
RWM-U learns only this transition:

```text
physics_state_t + isaac_joint_position_action_t -> physics_state_t_plus_1 + contact_t_plus_1
```

Reward, done, observations, tokenizer future reference, proprio history, motion
time cursor, and adaptive sampling remain SONIC/Isaac manager logic. The exporter
still stores Isaac reward/done as diagnostics so parity can be checked, but they
are not RWM-U primary training targets.

Code sources:

- Policy experiment and future reference settings: `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`.
- Isaac action term: `gear_sonic/config/manager_env/actions/terms/joint_pos.yaml`, a `JointPositionActionCfg` over all robot joints.
- Wrapper action boundary: `gear_sonic/envs/wrapper/manager_env_wrapper.py`; `_last_env_actions_to_sim` is the exact decoded/clipped 29-D action handed to `self.env.step(env_actions)`.
- Motion reset/reference sampling: `gear_sonic/envs/manager_env/mdp/commands.py`; `_resample_command` samples `motion_ids` and `motion_start_time_steps`, reads motion-lib root/body/joint state, applies reset randomization/clipping, then writes root/joint state to Isaac.
- Physics fields exported before and after step: `gear_sonic/scripts/export_rwmu_transitions.py` reads `scene["robot"].data`; `gear_sonic/scripts/collect_sonic_rwmu_dataset.py` flattens those fields into `rwm_u.state` and `rwm_u.next_state`.

Current flattened physics state terms are:

```text
root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w,
root_lin_vel_b, root_ang_vel_b, projected_gravity_b,
joint_pos, joint_vel,
body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
```

For the real SONIC G1 release Isaac env, validation currently reports 470
physics-state dimensions: root fields 22 + joint fields 58 + 30 robot bodies
with position/quaternion/linear-velocity/angular-velocity fields. The action is
29 dimensions, matching the Isaac Lab `joint_pos` action term.

`collect_sonic_rwmu_dataset.py` deliberately skips `motion_state` by default.
RWM-U does not train on future motion/reference internals, and some command
properties are expensive or unsafe to materialize during Isaac export. Use
`--record-motion-state` only for targeted debugging.

A local real-Isaac smoke was verified with the release path, a 6-motion subset,
`num_envs=1`, and random actions. The exported dataset validated as:

```text
state:       (2, 1, 470)
next_state:  (2, 1, 470)
action:      (2, 1, 29)
contact:     (2, 1, 30)
extension:   (2, 1, 0)
termination: (2, 1, 0)
```

Sampling remains rollout-based because Isaac needs a valid simulator state before
each transition. Training/validation can treat every exported row as an
independent transition sample `(s_t, a_t, s_t+1)`, while n-step evaluation uses
consecutive rows from the same rollout to measure accumulated model error.

## 6. SONIC Policy Rollout Export And Dynamics Smoke

Collect rollout data with one or more SONIC policy checkpoints. Repeat
`--checkpoint` or pass `--checkpoint-list` to mix policies for data diversity.
The smoke command below uses the RWM placeholder environment only to verify the
policy sampling and dataset format path:

```bash
python gear_sonic/scripts/collect_sonic_rwmu_dataset.py \
  --output /tmp/sonic_rwmu_policy_smoke.pt \
  --steps 16 \
  --device cuda \
  --action-source policy \
  --checkpoint sonic_release/last.pt \
  --checkpoint sonic_release/last.pt \
  --policy-selection step_cycle \
  -- +exp=rwm/sonic_release num_envs=2 headless=True use_wandb=false
```

Validate the exported RWM-U tensors:

```bash
python gear_sonic/scripts/validate_sonic_rwmu_dataset.py \
  /tmp/sonic_rwmu_policy_smoke.pt --json
```

Train a minimal SONIC RWM-U dynamics checkpoint from the exported data:

```bash
python gear_sonic/scripts/train_sonic_rwmu_dynamics.py \
  --dataset /tmp/sonic_rwmu_policy_smoke.pt \
  --output /tmp/sonic_rwmu_dynamics_smoke.pt \
  --report /tmp/sonic_rwmu_train_report.json \
  --device cuda \
  --history-horizon 8 \
  --forecast-horizon 2 \
  --ensemble-size 2 \
  --hidden-size 64 \
  --batch-size 8 \
  --epochs 1
```

For real SONIC RWM training, run the same exporter with
`+exp=manager/universal_token/all_modes/sonic_release` inside an Isaac Lab env.
The smoke dataset reports `missing_zero_fallback:*` state terms because the RWM
placeholder has no physical robot state; do not use smoke data for model quality
claims.


## 7. Current End-To-End Smoke Result

The following smoke chain has been verified locally with 12 steps and 2 parallel
environments:

```bash
python gear_sonic/scripts/collect_sonic_rwmu_dataset.py \
  --output /tmp/sonic_rwmu_12.pt \
  --steps 12 \
  --device cuda \
  --action-source policy \
  --checkpoint sonic_release/last.pt \
  -- +exp=rwm/sonic_release num_envs=2 headless=True use_wandb=false

python gear_sonic/scripts/collect_sonic_rwmu_dataset.py \
  --output /tmp/sonic_rwmu_12_random.pt \
  --steps 12 \
  --device cuda \
  --action-source random \
  -- +exp=rwm/sonic_release num_envs=2 headless=True use_wandb=false

python gear_sonic/scripts/validate_sonic_rwmu_dataset.py /tmp/sonic_rwmu_12.pt --json

python gear_sonic/scripts/train_sonic_rwmu_dynamics.py \
  --dataset /tmp/sonic_rwmu_12.pt \
  --dataset /tmp/sonic_rwmu_12_random.pt \
  --output /tmp/sonic_rwmu_dynamics_12.pt \
  --report /tmp/sonic_rwmu_dynamics_12_report.json \
  --device cuda \
  --history-horizon 6 \
  --forecast-horizon 2 \
  --ensemble-size 2 \
  --hidden-size 64 \
  --batch-size 4 \
  --epochs 1

python gear_sonic/train_agent_trl.py \
  +exp=rwm/sonic_release \
  +checkpoint=sonic_release/last.pt \
  num_envs=2 headless=True \
  rwm_env.backend=rwmu \
  +rwm_env.checkpoint=/tmp/sonic_rwmu_dynamics_12.pt \
  algo.config.num_steps_per_env=2 \
  algo.config.num_learning_iterations=1 \
  algo.config.num_learning_epochs=1 \
  algo.config.num_mini_batches=1 \
  use_wandb=false
```

The final training smoke logged `Env/rwm/reward`, `Env/rwm/aleatoric`, and
`Env/rwm/epistemic`, confirming that policy training used the loaded RWM-U
checkpoint path rather than the placeholder smoke reward.

The same smoke dataset intentionally fails strict physical-state validation:

```bash
python gear_sonic/scripts/validate_sonic_rwmu_dataset.py \
  /tmp/sonic_rwmu_12.pt --require-physical-state
```

That failure is expected because `+exp=rwm/sonic_release` has no Isaac robot
state and therefore uses `missing_zero_fallback:*` state terms. Real Isaac data
must pass:

```bash
python gear_sonic/scripts/validate_sonic_rwmu_dataset.py \
  /path/to/sonic_isaac_dataset.pt \
  --require-physical-state
```

Add `--require-contact` once contact labels are mandatory for the selected RWM-U
training target.

A real Isaac collection attempt reached Isaac AppLauncher, loaded the G1 assets,
and started loading `data/motion_lib_bones_seed/robot_filtered` with 129785
motions, but was manually stopped before rollout write-out because first startup
was taking several minutes. For the next real run, use a smaller motion subset or
prebuild the motion cache before collecting RWM-U data.

## 8. Sampling Strategy For Real RWM-U Data

Do not copy the original SONIC PPO sampling distribution exactly. PPO mostly
samples on-policy states for improving the current policy, while RWM-U needs to
cover states that future policy optimization may visit after the simulator is
replaced.

Recommended initial mixture:

- Released policy, deterministic action mean: stable nominal tracking manifold.
- Released policy, stochastic actions: local action perturbations around the
  nominal manifold.
- Random/noisy actions with short horizons: dynamics sensitivity and recovery
  labels, but cap episode length to avoid wasting data far outside useful states.
- Multiple fine-tuned or intermediate checkpoints: policy-distribution diversity.
- Failure-heavy clips and difficult motion bins: termination boundary coverage.
- Held-out motion IDs and start phases: n-step validation without train leakage.

Keep separate train/validation splits by motion ID and policy checkpoint. If the
same motion and same policy appears in both splits, n-step error will be overly
optimistic.


## 9. Managed First-Pass Isaac Sampling

Use a manifest rather than sampling the entire motion tree directly. The manifest
records selected motions, unselected candidates, train/val split labels, and the
materialized subset directories used by the collector.

Create a small first-pass subset:

```bash
python gear_sonic/scripts/prepare_sonic_rwmu_sampling_manifest.py \
  --robot-motion-root data/motion_lib_bones_seed/robot_filtered \
  --smpl-motion-root data/smpl_filtered \
  --output-root data/rwmu_sampling/v1_small \
  --output data/rwmu_sampling/v1_small/manifest.json \
  --name v1_small \
  --max-per-category 2 \
  --materialize symlink
```

The default categories are intentionally broad and small:

```text
nominal_walk, turning, jog, jump, balance_idle, hard_lower_body
```

The local smoke run used `--max-per-category 1` and selected 6 motions. This is
not enough for training quality, but it is enough to verify directory layout,
manifest bookkeeping, and command generation.

Dry-run the real Isaac collection plan before starting expensive jobs:

```bash
python gear_sonic/scripts/run_sonic_rwmu_sampling_plan.py \
  --manifest data/rwmu_sampling/v1_small/manifest.json \
  --output-dir /mnt/datasets/sonic_rwmu/v1_small/raw \
  --ledger /mnt/datasets/sonic_rwmu/v1_small/ledger.json \
  --policy sonic_release/last.pt \
  --action-source policy_mean \
  --action-source policy_stochastic \
  --action-source random \
  --sim-preset isaac \
  --steps 512 \
  --num-envs 128 \
  --device cuda \
  --require-physical-state \
  --dry-run
```

Run the same command without `--dry-run` on the compute machine. The runner writes
a ledger after every task, so interrupted jobs still show which datasets finished
and which failed. Each output dataset is validated immediately.

For a local no-Isaac plumbing check, use the smoke preset:

```bash
python gear_sonic/scripts/run_sonic_rwmu_sampling_plan.py \
  --manifest /tmp/sonic_rwmu_sampling_v1_small/manifest.json \
  --output-dir /tmp/sonic_rwmu_sampling_v1_small/out_smoke \
  --ledger /tmp/sonic_rwmu_sampling_v1_small/ledger_smoke.json \
  --policy sonic_release/last.pt \
  --action-source policy_mean \
  --action-source random \
  --sim-preset rwm-smoke \
  --steps 8 \
  --num-envs 2 \
  --device cuda
```

This smoke path has been verified locally and produced two valid datasets plus a
ledger with `ok=2`. It still uses `missing_zero_fallback:*` state terms and must
not be used for RWM-U quality training.

### First-Pass Collection Recommendation

For the first useful Isaac dataset, keep the run modest but diverse:

- 12 to 24 selected motions from the manifest categories.
- 3 action sources: `policy_mean`, `policy_stochastic`, and short-horizon
  `random`.
- 1 to 3 policy checkpoints initially; add intermediate fine-tuned checkpoints
  later.
- `num_envs=128` and `steps=512` per task as a first scalable check, then expand.
- Require `--require-physical-state`; add `--require-contact` only after contact
  labels are confirmed non-empty for the selected task.

Use separate manifests or split labels for train and validation. Do not evaluate
n-step error on the same motion-policy pairs used for training.
