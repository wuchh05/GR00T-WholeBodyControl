# RWM-U Training And SONIC Field Contract

This document records the runnable RWM-U path that is currently wired into this
repo, plus the SONIC-to-RWM-U field contract required before replacing Isaac.

## What RWM-U Provides

The upstream RWM-U repository under `external_dependencies/robotic_world_model`
ships a small offline smoke dataset and a pretrained dynamics checkpoint:

- `assets/data/state_action_data_0.csv`: 10000 rows, no header.
- `assets/models/pretrain_rnn_ens.pt`: pretrained RWM-U dynamics model.

This dataset is for the upstream ANYmal-D flat locomotion example. It proves the
offline RWM-U pipeline runs, but it is not SONIC humanoid data. SONIC data must
be exported from Isaac or another physical backend by rolling a policy and
recording transitions.

## Environment Installation

Use a separate environment for RWM-U if possible. The model-based RSL-RL package
uses the same import name, `rsl_rl`, as regular RSL-RL, so installing it replaces
the normal `rsl_rl` import in that environment. SONIC itself uses TRL for the
release training path, but keeping RWM-U separate avoids accidental dependency
confusion.

```bash
# In the SONIC/Isaac training environment, or a cloned environment:
pip install -e gear_sonic[training,rwm]

# RWM-U extension and model-based RSL-RL.
# If these directories are absent after cloning this repo, fetch them first:
git clone https://github.com/leggedrobotics/robotic_world_model.git external_dependencies/robotic_world_model
git clone https://github.com/leggedrobotics/rsl_rl_rwm.git external_dependencies/rsl_rl_rwm

pip install -e external_dependencies/robotic_world_model/source/mbrl
pip install -e external_dependencies/rsl_rl_rwm
```

The RWM-U smoke path does not require launching Isaac. Isaac is only needed when
you export SONIC transitions from the real SONIC training environment.

## Upstream RWM-U Data Format

The upstream offline loader reads sequential CSV files named
`state_action_data_0.csv`, `state_action_data_1.csv`, etc. Each file has no
header and each row is one transition label at time `t`:

```text
system_state_t | system_action_t | system_extension_t | system_contact_t | system_termination_t
```

The loader slices the columns in this order using task-specific dimensions. For
the upstream ANYmal-D smoke task the dimensions are:

- `system_state`: 45
- `system_action`: 12
- `system_extension`: 0
- `system_contact`: 8
- `system_termination`: 1

For SONIC, use the schema in `gear_sonic/config/rwm/sonic_schema.yaml` instead
of the ANYmal dimensions. Keep episode rows contiguous and mark learned
termination on the final failed row. Do not learn `time_out`; keep it separate
for PPO bootstrap correction.

## SONIC To RWM-U Field Contract

The SONIC release policy/trainer fields are validated by:

```bash
python gear_sonic/scripts/validate_rwm_config_parity.py \
  --num-envs 4096 --checkpoint sonic_release/last.pt
python gear_sonic/scripts/validate_sonic_rwmu_fields.py
```

Required RWM-U groups for SONIC:

### system_action

Use the final applied 29D joint-position action that reaches Isaac, after action
transform and clipping. In `ManagerEnvWrapper.step`, this is `env_actions` just
before `self.env.step(env_actions)`.

### system_state

Use clean simulator/robot state, not noisy `actor_obs`. Recommended first layout:

1. `root_pos_w`: 3
2. `root_quat_w`: 4, wxyz
3. `root_lin_vel_b`: 3
4. `root_ang_vel_b`: 3
5. `joint_pos`: 29
6. `joint_vel`: 29
7. `body_pos_w`: 14 x 3
8. `body_quat_w`: 14 x 4, wxyz
9. `body_lin_vel_w`: 14 x 3
10. `body_ang_vel_w`: 14 x 3
11. `last_action`: 29
12. `motion_id`: 1
13. `motion_time_step`: 1
14. `motion_start_time_step`: 1

The tracked body order is the order in `sonic_schema.yaml`.

### system_extension

Store every SONIC release reward term before weighting, every weighted
contribution, and total reward. The release reward labels are:

- `tracking_anchor_pos`
- `tracking_anchor_ori`
- `tracking_relative_body_pos`
- `tracking_relative_body_ori`
- `tracking_body_linvel`
- `tracking_body_angvel`
- `action_rate_l2`
- `joint_limit`
- `undesired_contacts`
- `anti_shake_ang_vel`
- `tracking_vr_5point_local`
- `feet_acc`
- `reward_total`

### system_contact

At minimum, store per-body contact flags and the `undesired_contacts` aggregate.
Prefer storing foot/wrist/ankle contact flags explicitly because several reward
and termination terms depend on contact or foot state.

### system_termination

Learn only physical/task failure terms:

- `anchor_pos`
- `anchor_ori_full`
- `ee_body_pos`
- `foot_pos_xyz`

Keep `time_out` separate from learned termination.

## Smoke Training

Run the bundled upstream RWM-U offline smoke test:

```bash
python gear_sonic/scripts/run_rwmu_offline_smoke.py \
  --device cuda \
  --num-envs 2 \
  --num-steps-per-env 2 \
  --max-iterations 1 \
  --max-episode-length 8 \
  --output-dir /tmp/rwmu_offline_smoke
```

Expected result:

```text
RWM-U offline smoke OK: /tmp/rwmu_offline_smoke/policy_0.pt
```

For real training on a larger platform, start from the upstream command and
increase the config values in `scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py`
or create a SONIC-specific config with SONIC dimensions:

```bash
cd external_dependencies/robotic_world_model
python scripts/reinforcement_learning/model_based/train.py --task anymal_d_flat
```

The upstream default trains an imagined policy using the bundled ANYmal-D data
and pretrained dynamics. For SONIC, first export SONIC transitions from Isaac and
replace the dataset/config dimensions with the SONIC schema.

## Verification Commands

```bash
python -m py_compile \
  gear_sonic/scripts/run_rwmu_offline_smoke.py \
  gear_sonic/scripts/validate_sonic_rwmu_fields.py \
  gear_sonic/scripts/export_rwmu_transitions.py \
  gear_sonic/scripts/validate_rwm_config_parity.py

python gear_sonic/scripts/validate_rwm_config_parity.py \
  --num-envs 4096 --checkpoint sonic_release/last.pt

python gear_sonic/scripts/validate_sonic_rwmu_fields.py

python gear_sonic/scripts/export_rwmu_transitions.py \
  --output /tmp/sonic_rwmu_smoke.pt \
  --steps 3 --device cpu --action-mode zeros -- \
  +exp=rwm/sonic_release +checkpoint=sonic_release/last.pt num_envs=2 headless=True

python gear_sonic/scripts/run_rwmu_offline_smoke.py \
  --device cuda --num-envs 2 --num-steps-per-env 2 \
  --max-iterations 1 --max-episode-length 8 \
  --output-dir /tmp/rwmu_offline_smoke
```
