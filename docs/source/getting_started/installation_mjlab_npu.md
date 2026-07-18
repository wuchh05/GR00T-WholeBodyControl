# Installation (mjlab / Ascend NPU Training)

This guide is for the SONIC mjlab path. It does not require Isaac Sim or Isaac
Lab at runtime. The intended Ascend setup is:

- CPU rollout with mjlab / MuJoCo.
- Neural-network forward, backward, and PPO updates on Ascend NPU via
  `torch-npu`.

The original Isaac Lab training path is still documented in
[Installation (Training)](installation_training.md).

## Prerequisites

- Ubuntu 22.04+.
- Conda.
- Python 3.11.
- Git LFS.
- For Ascend: driver, firmware, and CANN are installed before running this repo
  installer. `torch-npu`, PyTorch, and CANN versions must match the Ascend
  software matrix for your machine.

## Clone Your Fork

```bash
git clone https://github.com/<your-org>/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git lfs install
git lfs pull
```

## One-Command Install

CPU-only rollout/development environment:

```bash
bash install_scripts/install_mjlab_training.sh
```

CUDA reference environment:

```bash
DEVICE_BACKEND=cuda bash install_scripts/install_mjlab_training.sh
```

Ascend NPU environment:

```bash
DEVICE_BACKEND=npu \
TORCH_NPU_PACKAGE='torch-npu' \
bash install_scripts/install_mjlab_training.sh
```

If your Ascend platform requires a local wheel or exact package version, pass it
through `TORCH_NPU_PACKAGE`, for example:

```bash
DEVICE_BACKEND=npu \
TORCH_NPU_PACKAGE=/path/to/torch_npu-*.whl \
bash install_scripts/install_mjlab_training.sh
```

Useful installer variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENV_NAME` | `sonic` | Conda environment name |
| `PYTHON_VERSION` | `3.11` | Python version |
| `DEVICE_BACKEND` | `cpu` | `cpu`, `cuda`, `npu`, or `none` |
| `TORCH_VERSION` | `2.7.0` | PyTorch version installed before SONIC |
| `TORCHVISION_VERSION` | `0.22.0` | torchvision version |
| `MJLAB_SPEC` | `mjlab==1.5.0` | mjlab package spec or wheel path |
| `INSTALL_DATA` | `0` | Set to `1` to run `download_from_hf.py --training` |
| `RUN_CHECKS` | `1` | Run `check_environment.py` after installation |

## Verify

```bash
conda activate sonic
python check_environment.py --mjlab
python check_environment.py --npu      # Ascend machines only
```

`--mjlab` verifies `gear_sonic`, training dependencies, `mjlab`, and MuJoCo.
`--npu` additionally imports `torch_npu` and checks that `torch.npu` reports at
least one available device.

## Download Checkpoint and SMPL Data

```bash
python download_from_hf.py --training
```

This downloads:

- `sonic_release/last.pt`
- `data/smpl_filtered/`

## Prepare mjlab Motion Data

First prepare the official SONIC robot PKLs with the original pipeline:

```bash
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
  --input /path/to/bones_seed/g1/csv/ \
  --output data/motion_lib_bones_seed/robot \
  --fps 30 --fps_source 120 --individual --num_workers 16

python gear_sonic/data_process/filter_and_copy_bones_data.py \
  --source data/motion_lib_bones_seed/robot \
  --dest data/motion_lib_bones_seed/robot_filtered --workers 16
```

Then export the official MotionLib reference tensors into mjlab NPZ files:

```bash
python gear_sonic/data_process/export_motionlib_robot_to_mjlab_npz.py \
  --input data/motion_lib_bones_seed/robot_filtered \
  --output data/motion_lib_bones_seed/mjlab_npz_motionlib_50f \
  --device cpu --workers 16 --skip-existing
```

Validate the NPZ dataset:

```bash
python gear_sonic/scripts/check_mjlab_npz_dataset.py \
  --npz-root data/motion_lib_bones_seed/mjlab_npz_motionlib_50f \
  --check-finite --workers 16 \
  --output-json logs/data_processing/parity_checks/mjlab_npz_dataset.json
```

## CUDA Reference Smoke

Use this before changing NPU code:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n sonic python gear_sonic/train_agent_trl.py \
  +exp=mjlab/sonic_mjlab_universal_smoke \
  checkpoint=sonic_release/last.pt \
  mjlab_env.motion_dir=data/motion_lib_bones_seed/mjlab_npz_motionlib_50f \
  mjlab_env.smpl_motion_file=data/smpl_filtered \
  mjlab_env.max_motions=4 \
  num_envs=32 \
  algo.config.num_steps_per_env=16 \
  algo.config.num_learning_iterations=200 \
  algo.config.num_learning_epochs=1 \
  algo.config.num_mini_batches=1 \
  experiment_name=mjlab_cuda_overfit_4m_32e_200it \
  use_wandb=false
```

Summarize the run:

```bash
python gear_sonic/scripts/summarize_mjlab_training.py \
  --run-dir logs_rl/TRL_G1_MjLab/<run-dir> \
  --output-json logs/data_processing/mjlab_cuda_overfit_4m_32e_200it_summary.json
```

The first acceptance check is:

- `nonfinite_values` is empty.
- `objective/rewards` improves.
- `loss/value_avg` decreases.
- `loss/total_aux_loss_avg` decreases.

## Notes

- `mjlab` provides the G1 MuJoCo XML used by the SONIC mjlab environment. No
  extra XML copy step is required when `mjlab` is installed as a package.
- For small tests with `mjlab_env.max_motions=N`, add
  `mjlab_env.sort_motion_files=true` if you need deterministic "first N after
  sorting" sampling.
- Full Isaac Lab training and evaluation still require the original Isaac Lab
  environment.
