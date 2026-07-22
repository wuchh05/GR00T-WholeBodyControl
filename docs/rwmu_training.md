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

Use two environments. The reason is simple: upstream RWM-U depends on
`rsl_rl_rwm`, which installs as the Python package name `rsl_rl`. That can replace
the regular `rsl_rl` used by Isaac/RSL-RL projects. SONIC release training uses
TRL rather than RSL-RL, but keeping RWM-U separate still avoids accidental import
or dependency conflicts.

### Environment A: SONIC / Isaac

Use this environment for original SONIC training, Isaac rollout export, policy
evaluation, and final SONIC policy fine-tuning.

```bash
conda create -n sonic-isaac python=3.11 -y
conda activate sonic-isaac

# Install Isaac Lab / Isaac Sim in this environment first, following the Isaac
# Lab installation guide used by SONIC.

pip install -e "gear_sonic[training,rwm]"
```

Do not install `external_dependencies/rsl_rl_rwm` here unless you deliberately
want one mixed smoke environment. This environment is the one that runs commands
such as:

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=sonic_release/last.pt \
  num_envs=4096 headless=True

python gear_sonic/scripts/export_rwmu_transitions.py \
  --output /tmp/sonic_rwmu_export.pt --steps 128 --device cuda -- \
  +exp=rwm/sonic_release +checkpoint=sonic_release/last.pt num_envs=64 headless=True
```

The current exporter smoke uses the RWM smoke backend. Full Isaac export still
needs Isaac AppLauncher wiring before it can export real Isaac transitions.

### Environment B: RWM-U

Use this environment for RWM-U dynamics/offline imagination training. It does not
need to launch Isaac for the bundled upstream smoke data.

```bash
conda create -n rwmu python=3.11 -y
conda activate rwmu

# Choose the PyTorch/CUDA build that matches the training machine. Example only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -e "gear_sonic[rwm]"

# Fetch upstream code if it is not already present after clone.
git clone https://github.com/leggedrobotics/robotic_world_model.git external_dependencies/robotic_world_model
git clone https://github.com/leggedrobotics/rsl_rl_rwm.git external_dependencies/rsl_rl_rwm

pip install -e external_dependencies/robotic_world_model/source/mbrl
pip install -e external_dependencies/rsl_rl_rwm
```

This environment is the one that runs:

```bash
python gear_sonic/scripts/run_rwmu_offline_smoke.py \
  --device cuda --num-envs 2 --num-steps-per-env 2 \
  --max-iterations 1 --max-episode-length 8 \
  --output-dir /tmp/rwmu_offline_smoke
```

When SONIC-specific RWM-U data has been exported, train RWM-U in this environment
on that exported dataset. The trained RWM-U checkpoint can then be loaded by a
SONIC-side adapter. If the adapter imports only PyTorch checkpoint code, keep it
inside the SONIC environment. If it imports `rsl_rl_rwm` classes directly, run the
adapter in the RWM-U environment or vendor a small inference-only loader.

## Upstream RWM-U Data Format

Think of the RWM-U dataset as a simulator diary. Each row is one time step from a
real simulator rollout:

```text
what the robot is now | what action was applied | extra labels | contact labels | failure labels
```

The upstream file is named `state_action_data_0.csv`. It has no column names
because the code already knows how many numbers belong to each block. In the
bundled ANYmal-D smoke example, the first 45 numbers are robot state, the next 12
numbers are action, then 0 extension numbers, 8 contact numbers, and 1
termination number.

A tiny toy row would look conceptually like this:

```text
[base velocity, gravity, joint pos, joint vel, torque] | [12 motor commands] | [] | [feet/thigh contacts] | [base hit ground]
```

SONIC uses a humanoid, so its row is wider. The idea is unchanged: keep a
continuous episode as continuous rows, and let RWM-U learn how action at time `t`
changes the robot state at time `t+1`. The target motion file alone is not enough
because it has no applied policy action and no simulated robot response.

## SONIC To RWM-U Field Contract

The field contract means: for every thing Isaac gives SONIC during training, we
must decide where that thing lives in the RWM-U row. This is just bookkeeping so
we can later replace Isaac without changing the SONIC policy/trainer interface.

The validation commands are:

```bash
python gear_sonic/scripts/validate_rwm_config_parity.py \
  --num-envs 4096 --checkpoint sonic_release/last.pt
python gear_sonic/scripts/validate_sonic_rwmu_fields.py
```

The five groups mean:

- `system_state`: clean robot state. Example: `joint_pos` is 29 numbers because
  release SONIC controls 29 body joints. `body_pos_w` is `14 x 3` numbers because
  SONIC tracks 14 body links and each link has xyz world position.
- `system_action`: the action that actually reaches the simulator. For SONIC this
  is the final 29D joint-position command after action transform/clipping, not an
  intermediate latent or noisy observation.
- `system_extension`: extra continuous labels, mainly reward information. Example:
  `tracking_anchor_pos` says how far the robot pelvis/anchor is from the target
  motion anchor; `reward_total` is the final scalar reward.
- `system_contact`: contact labels. Example: whether feet or other bodies touch
  the ground, plus aggregates such as `undesired_contacts`.
- `system_termination`: failure labels. Example: `anchor_pos` becomes true when
  the robot drifts too far from the target anchor. `time_out` is not learned; it
  is just episode length bookkeeping.

So a SONIC row is conceptually:

```text
[clean humanoid state and motion cursor] |
[29D applied joint action] |
[reward term labels and total reward] |
[contact flags] |
[failure flags]
```

The noisy `actor_obs` that enters the policy is not the best training target for
RWM-U. It is derived from clean state by adding history, target motion, and noise.
RWM-U should predict clean state/contact/failure; the adapter can rebuild
SONIC-style `actor_obs`, `critic_obs`, reward, and done from those predictions.

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
