# RWM-U Local Validation

Date: 2026-07-22

## Environment Status

A new isolated conda environment was created:

```bash
conda create -n rwmu-local python=3.11 -y
```

Installing the CUDA PyTorch stack into `rwmu-local` was not completed on this machine. The root `/tmp` filesystem became full during the CUDA wheel download; after cleanup it had about 11GB free, still not enough margin for the full CUDA dependency set. The validation below therefore used the existing `sonic` environment, which already has PyTorch 2.7.0+cu128 and `rsl_rl_rwm` available.

Runtime used for validation:

```text
Python: /mnt/conda/wuchenghui/miniconda3/envs/sonic/bin/python
Torch: 2.7.0+cu128
CUDA: available
GPU: NVIDIA L40, 44.39GB
```

## Commands Run

RWM-U offline policy-training smoke with the upstream pretrained dynamics checkpoint:

```bash
/mnt/conda/wuchenghui/miniconda3/envs/sonic/bin/python \
  gear_sonic/scripts/run_rwmu_offline_smoke.py \
  --device cuda \
  --num-envs 2 \
  --num-steps-per-env 2 \
  --max-iterations 1 \
  --max-episode-length 8 \
  --output-dir /tmp/rwmu_offline_smoke_real
```

Result:

```text
RWM-U offline smoke OK: /tmp/rwmu_offline_smoke_real/policy_0.pt
```

32-step open-loop error with the upstream bundled ANYmal-D data and real pretrained RWM-U dynamics checkpoint:

```bash
/mnt/conda/wuchenghui/miniconda3/envs/sonic/bin/python \
  gear_sonic/scripts/evaluate_rwmu_nstep.py \
  --device cuda \
  --batch-size 512 \
  --max-horizon 32 \
  --output /tmp/rwmu_nstep_error.json
```

Full JSON report:

```text
/tmp/rwmu_nstep_error.json
```

## Scope

This is an upstream RWM-U validation, not a SONIC humanoid validation. It uses:

- data: `external_dependencies/robotic_world_model/assets/data/state_action_data_0.csv`
- checkpoint: `external_dependencies/robotic_world_model/assets/models/pretrain_rnn_ens.pt`
- checkpoint iteration: 5000
- state/action/contact/termination dims: `45 / 12 / 8 / 1`
- history horizon: 32
- evaluated horizons: 1 to 32

The upstream RWM-U offline smoke trains a policy in imagined rollouts using a fixed pretrained dynamics model. RWM-U dynamics pretraining itself is described upstream as an Isaac-based online collection/training path, so it is not the Isaac-free path we want for SONIC unless we first provide an exported SONIC transition dataset.

## Key n-step Metrics

| horizon | state RMSE | base lin vel RMSE | joint pos RMSE | joint vel RMSE | torque RMSE | contact acc | termination acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.5267 | 0.0404 | 0.0085 | 0.2908 | 4.8842 | 0.9976 | 1.0000 |
| 8 | 2.4314 | 0.0720 | 0.0160 | 0.3176 | 4.6973 | 0.9963 | 1.0000 |
| 16 | 2.7498 | 0.0774 | 0.0221 | 0.4513 | 5.3052 | 0.9912 | 1.0000 |
| 32 | 3.1365 | 0.0864 | 0.0330 | 0.6214 | 6.0408 | 0.9800 | 1.0000 |

The aggregate state RMSE is dominated by the torque slice because torque values have a much larger physical scale. For SONIC gating, compare state groups separately instead of only looking at the aggregate number.
