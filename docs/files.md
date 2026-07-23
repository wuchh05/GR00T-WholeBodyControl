# Repository File Map

This file records project-specific helper scripts and integration documents that
are easy to miss in the larger SONIC tree.

## RWM-U Integration

- `docs/rwm_sonic_integration.md` - code-reading map, active SONIC release MDP
  fields, and the RWM-U field contract.
- `docs/rwmu_training.md` - RWM-U install, bundled data, smoke training,
  SONIC field contract, and deployment commands.
- `docs/rwmu_local_validation.md` - local RWM-U environment status, real-weight
  smoke command, and 32-step open-loop error results.
- `gear_sonic/config/exp/rwm/sonic_release.yaml` - RWM backend variant of the
  original `manager/universal_token/all_modes/sonic_release` experiment. It
  preserves the original SONIC policy/trainer architecture and only swaps the
  environment backend.
- `gear_sonic/config/exp/rwm/sonic_rwm_smoke.yaml` - small RWM smoke experiment
  for fast interface checks; not architecture-equivalent to the release config.
- `gear_sonic/config/rwm/sonic_schema.yaml` - structured schema for exporting
  SONIC transitions into RWM-U groups.
- `gear_sonic/envs/rwm_env.py` - SONIC trainer-compatible RWM/RWM-U environment
  wrapper. It supports `backend=smoke` for plumbing and `backend=rwmu` for
  loading SONIC RWM-U dynamics checkpoints produced by the local training script.
- `gear_sonic/scripts/validate_rwm_config_parity.py` - Hydra config check that
  compares original SONIC release config against `exp=rwm/sonic_release` and
  verifies actor/critic/trainer/MDP config parity.
- `gear_sonic/scripts/validate_sonic_rwmu_fields.py` - validates that the
  SONIC release schema covers required RWM-U groups and tracked body metadata.
- `gear_sonic/scripts/run_rwmu_offline_smoke.py` - runs the upstream RWM-U
  offline training smoke using bundled ANYmal-D data and pretrained dynamics.
- `gear_sonic/scripts/evaluate_rwmu_nstep.py` - evaluates n-step open-loop error
  for the upstream bundled RWM-U dynamics checkpoint and CSV data.
- `gear_sonic/scripts/export_rwmu_transitions.py` - backend-agnostic transition
  exporter for RWM-U datasets. It records obs/action/reward/done/timeouts and
  best-effort robot/motion state fields.
- `gear_sonic/scripts/train_sonic_rwmu_dynamics.py` - trains a SONIC-specific
  RWM-U `SystemDynamicsEnsemble` checkpoint from exported rollout datasets.
- `gear_sonic/scripts/validate_sonic_rwmu_dataset.py` - validates exported SONIC
  RWM-U dataset tensor ranks, dimensions, finite values, and schema metadata.
- `gear_sonic/scripts/collect_sonic_rwmu_dataset.py` - collects SONIC rollouts
  from one or more policy checkpoints and exports raw transitions plus RWM-U
  `state/action/extension/contact/termination` tensors.
- `external_dependencies/robotic_world_model/` - upstream ETH RWM/RWM-U codebase
  cloned for reference and later checkpoint/model integration.
