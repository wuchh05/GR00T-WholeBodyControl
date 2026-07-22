# Repository File Map

This file records project-specific helper scripts and integration documents that
are easy to miss in the larger SONIC tree.

## RWM-U Integration

- `docs/rwm_sonic_integration.md` - code-reading map, active SONIC release MDP
  fields, and the RWM-U field contract.
- `docs/rwmu_training.md` - RWM-U install, bundled data, smoke training,
  SONIC field contract, and deployment commands.
- `gear_sonic/config/exp/rwm/sonic_release.yaml` - RWM backend variant of the
  original `manager/universal_token/all_modes/sonic_release` experiment. It
  preserves the original SONIC policy/trainer architecture and only swaps the
  environment backend.
- `gear_sonic/config/exp/rwm/sonic_rwm_smoke.yaml` - small RWM smoke experiment
  for fast interface checks; not architecture-equivalent to the release config.
- `gear_sonic/config/rwm/sonic_schema.yaml` - structured schema for exporting
  SONIC transitions into RWM-U groups.
- `gear_sonic/envs/rwm_env.py` - SONIC trainer-compatible RWM/RWM-U environment
  wrapper. Current implemented backend is `smoke`; real RWM-U checkpoint loading
  should be implemented behind this same wrapper.
- `gear_sonic/scripts/validate_rwm_config_parity.py` - Hydra config check that
  compares original SONIC release config against `exp=rwm/sonic_release` and
  verifies actor/critic/trainer/MDP config parity.
- `gear_sonic/scripts/validate_sonic_rwmu_fields.py` - validates that the
  SONIC release schema covers required RWM-U groups and tracked body metadata.
- `gear_sonic/scripts/run_rwmu_offline_smoke.py` - runs the upstream RWM-U
  offline training smoke using bundled ANYmal-D data and pretrained dynamics.
- `gear_sonic/scripts/export_rwmu_transitions.py` - backend-agnostic transition
  exporter for RWM-U datasets. It records obs/action/reward/done/timeouts and
  best-effort robot/motion state fields.
- `external_dependencies/robotic_world_model/` - upstream ETH RWM/RWM-U codebase
  cloned for reference and later checkpoint/model integration.
