# RWM-U Training Setup

This repo uses the stable engineering path: keep SONIC/Isaac and RWM-U training
in separate conda environments. Do not install `rsl_rl_rwm` in the SONIC/Isaac
environment. RWM-U weights should later be exported as an inference-only artifact
and loaded by `gear_sonic/envs/rwm_env.py` without importing `rsl_rl_rwm`.

Current status:

- Upstream RWM-U bundled-data smoke training runs.
- SONIC `+exp=rwm/sonic_release` smoke runs without Isaac using placeholder dynamics.
- A real SONIC-trained `rwm_env.backend=rwmu` checkpoint loader is not implemented yet.
- Full Isaac transition export still needs Isaac AppLauncher wiring; current exporter smoke uses the RWM smoke backend.

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

Step 4 is the missing implementation piece. Until it is added, use the commands
above for installation, smoke testing, and field validation.
