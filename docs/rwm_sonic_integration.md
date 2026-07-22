# SONIC RWM-U Integration Notes

This document records the current SONIC training data flow and the field
contract that a RWM-U backend must reproduce. Paths are relative to the repo
root.

## Where To Read The Code

- Training entrypoint: `gear_sonic/train_agent_trl.py`
  - Chooses backend by `sim_type`.
  - Isaac path calls `create_manager_env()`.
  - MuJoCo path calls `create_mjlab_env()`.
  - RWM smoke path now calls `create_rwm_env()`.
- PPO rollout loop: `gear_sonic/trl/trainer/ppo_trainer.py`
  - `reset_all()` is called around line 1700.
  - `env.step(policy_state_dict)` is called in `_rollout_step()`.
  - Required step return: `obs_dict, rewards, dones, infos`.
  - Required info keys: `episode`, `to_log`, `time_outs`.
- Policy network config: `gear_sonic/config/actor_critic/mlp.yaml`
  - Actor input key: `actor_obs`.
  - Critic input key: `critic_obs`.
  - Actor output dim: `robot_action_dim`, resolved from `env.config.robot.actions_dim`.
- Actor/Critic implementation: `gear_sonic/trl/modules/actor_critic_modules.py`
  - `Actor.rollout()` samples Gaussian actions from `actor_obs`.
  - `Critic.evaluate()` consumes `critic_obs`.
- Isaac task config: `gear_sonic/config/manager_env/base_env.yaml`
  - Composes action, observation, reward, termination, command, event configs.
- Isaac task instantiation: `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py`
  - Builds `MySceneCfg`, managers, robot, terrain, sensors.
- Isaac wrapper: `gear_sonic/envs/wrapper/manager_env_wrapper.py`
  - Converts manager env outputs to SONIC trainer contract.
- Existing non-Isaac backend: `gear_sonic/envs/mjlab_env.py`
  - Creates mjlab env and wraps it with `MjlabSonicEnvWrapper`.
- Existing non-Isaac wrapper: `gear_sonic/envs/wrapper/mjlab_sonic_env_wrapper.py`
  - Best template for RWM adapter behavior.
- RWM-U upstream code: `external_dependencies/robotic_world_model`
  - Offline imagined env base: `scripts/reinforcement_learning/model_based/envs/base.py`.
  - ANYmal task example: `scripts/reinforcement_learning/model_based/envs/anymal_d_flat.py`.
  - Isaac/RWM observation heads: `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py`.

## Active Default Isaac Tracking MDP

The release Isaac config is `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`.
The RWM backend equivalent is `gear_sonic/config/exp/rwm/sonic_release.yaml`. Both compose `gear_sonic/config/manager_env/base_env.yaml`; the parity script verifies that actor, critic, trainer, observations, rewards, and terminations match.

### Action

- Config: `gear_sonic/config/manager_env/actions/tracking/base.yaml`
- Term: `joint_pos`
- Term config: `gear_sonic/config/manager_env/actions/terms/joint_pos.yaml`
- Isaac class: `isaaclab.envs.mdp.actions.JointPositionActionCfg`
- Fields:
  - `asset_name: robot`
  - `joint_names: [".*"]`
  - `use_default_offset: true`
  - scale is injected in `ModularTrackingEnvCfg.override_settings()` from
    `gear_sonic/envs/manager_env/robots/g1.py`.
- Default G1 body action order is declared in
  `gear_sonic/envs/manager_env/mdp/actions.py` as `G1_MUJOCO_ORDER`.

### Policy Observation Group

Release config: `gear_sonic/config/manager_env/observations/policy/local_dir_hist.yaml`.
`concatenate_terms: true`; term order is the order below.

| Term | Function | Shape for 29-DOF / 14 bodies | Source state |
|---|---|---:|---|
| `gravity_dir` | `gravity_dir` | 30 | 3D gravity direction with 10-frame history |
| `base_ang_vel` | IsaacLab `base_ang_vel` | 30 | 3D root angular velocity with 10-frame history |
| `joint_pos` | IsaacLab `joint_pos_rel` | 290 | 29D relative joint position with 10-frame history |
| `joint_vel` | IsaacLab `joint_vel_rel` | 290 | 29D relative joint velocity with 10-frame history |
| `actions` | IsaacLab `last_action` | 290 | 29D last action with 10-frame history |

Policy observation corruption/noise is enabled in the Isaac policy group. For
RWM open-loop evaluation, log both clean state labels and post-noise policy obs.

### Critic Observation Group

Release config: `gear_sonic/config/manager_env/observations/critic/privileged_mf_hist.yaml`.
Term order:

| Term | Function | Shape for 29-DOF / 14 bodies | Source state |
|---|---|---:|---|
| `command` | `generated_commands` | command-dependent | `TrackingCommand` |
| `motion_anchor_pos_b` | `motion_anchor_pos_b` | 3 | ref-vs-robot anchor translation |
| `motion_anchor_ori_b` | `motion_anchor_ori_b` | 6 | ref-vs-robot anchor orientation, 6D |
| `body_pos` | `robot_body_pos_b` | 42 | tracked body positions in robot anchor frame |
| `body_ori` | `robot_body_ori_b` | 84 | tracked body orientations in robot anchor frame, 6D |
| `base_lin_vel` | IsaacLab `base_lin_vel` | 3 | root linear velocity in body frame |
| `base_ang_vel` | IsaacLab `base_ang_vel` | 3 | root angular velocity in body frame |
| `joint_pos` | IsaacLab `joint_pos_rel` | 29 | relative joint position |
| `joint_vel` | IsaacLab `joint_vel_rel` | 29 | relative joint velocity |
| `actions` | IsaacLab `last_action` | 29 | previous applied action |

### Motion Command

Config: `gear_sonic/config/manager_env/commands/terms/motion.yaml`.
Implementation: `gear_sonic/envs/manager_env/mdp/commands.py`.

Important fields:

- `motion_lib_cfg.motion_file`: reference motion file.
- `target_fps: 50`.
- `dt_future_ref_frames: 0.1`.
- `num_future_frames: 5`.
- `anchor_body: pelvis`.
- `body_names`: 14 tracked bodies: `pelvis`, left/right hip-roll, knee,
  ankle-roll, `torso_link`, left/right shoulder-roll, elbow, wrist-yaw.
- `vr_3point_body`: left wrist yaw, right wrist yaw, torso.
- `reward_point_body`: pelvis, two wrists, two ankle-roll links.

The command object owns current reference state, robot state projected into the
same body set, per-env motion ids, start time steps, and future-frame tensors.

### Reward Terms

Release config: `gear_sonic/config/manager_env/rewards/tracking/base_5point_local_feet_acc.yaml`.

| Term | Function | Weight | Label needed for RWM-U |
|---|---|---:|---|
| `tracking_anchor_pos` | `tracking_anchor_pos_error` | 0.5 | yes |
| `tracking_anchor_ori` | `tracking_anchor_ori_error` | 0.5 | yes |
| `tracking_relative_body_pos` | `tracking_relative_body_pos_error` | 1.0 | yes |
| `tracking_relative_body_ori` | `tracking_relative_body_ori_error` | 1.0 | yes |
| `tracking_body_linvel` | `tracking_body_linvel_error` | 1.0 | yes |
| `tracking_body_angvel` | `tracking_body_angvel_error` | 1.0 | yes |
| `action_rate_l2` | IsaacLab `action_rate_l2` | -0.1 | yes |
| `joint_limit` | IsaacLab `joint_pos_limits` | -10.0 | yes |
| `undesired_contacts` | IsaacLab `undesired_contacts` | -0.1 | yes |
| `anti_shake_ang_vel` | `anti_shake_ang_vel_l2` | -0.005 | yes |
| `tracking_vr_5point_local` | `tracking_local_vr_5point_error` | 2.0 | yes |
| `feet_acc` | IsaacLab `joint_acc_l2` | -2.5e-6 in release override | yes |

For RWM-U, store each unweighted term, each weighted contribution, and total
reward. Store contact-force labels because `undesired_contacts` depends on
`contact_forces.data.net_forces_w_history`.

### Termination Terms

Release config: `gear_sonic/config/manager_env/terminations/tracking/base_adaptive_strict_ori_foot_xyz.yaml`.

| Term | Function | Meaning |
|---|---|---|
| `anchor_pos` | `exceeded_anchor_height` | adaptive ref-vs-robot anchor z error, release threshold 0.15 m |
| `anchor_ori_full` | `exceeded_anchor_ori` | squared quaternion error exceeds 0.2 |
| `ee_body_pos` | `exceeded_body_height` | adaptive ankle/wrist z error, release threshold 0.15 m |
| `foot_pos_xyz` | `exceeded_body_pos` | foot xyz body error exceeds 0.2 m |
| `time_out` | `tracking_time_out` | reference clip consumed |

Keep `terminated` and `time_out` separate. PPO uses `infos["time_outs"]` for
bootstrap correction.

## RWM-U Field Contract For SONIC

Use RWM-U groups with these SONIC-specific meanings:

- `system_state`: clean physical/control state used by dynamics prediction.
- `system_action`: applied joint-position action after SONIC policy output is
  selected, before the next state.
- `system_extension`: continuous privileged heads, including reward terms.
- `system_contact`: binary or logit contact heads.
- `system_termination`: binary or logit termination heads excluding timeout.

Recommended first `system_state` layout:

1. `root_pos_w`: 3
2. `root_quat_w`: 4, wxyz
3. `root_lin_vel_b`: 3
4. `root_ang_vel_b`: 3
5. `joint_pos`: 29, absolute
6. `joint_vel`: 29, absolute
7. `body_pos_w`: 14 x 3
8. `body_quat_w`: 14 x 4, wxyz
9. `body_lin_vel_w`: 14 x 3
10. `body_ang_vel_w`: 14 x 3
11. `last_action`: 29
12. `motion_id`: 1
13. `motion_time_step`: 1
14. `motion_start_time_step`: 1

Recommended `system_extension` layout:

1. all 9 reward terms above, unweighted
2. all 9 weighted reward contributions
3. `reward_total`
4. `motion_anchor_pos_b`
5. `motion_anchor_ori_b`

Recommended `system_contact` layout:

1. per-body contact flags for all robot bodies
2. undesired-contact aggregate count
3. optional foot/wrist/ankle contact flags

Recommended `system_termination` layout:

1. `anchor_pos`
2. `anchor_ori_full`
3. `ee_body_pos`
4. `terminated_any`

Timeout should be computed by the adapter, not learned.

## Current RWM Smoke Adapter

- Entry: `gear_sonic/envs/rwm_env.py`
- Release-equivalent config: `gear_sonic/config/exp/rwm/sonic_release.yaml`
- Small interface config: `gear_sonic/config/exp/rwm/sonic_rwm_smoke.yaml`
- Parity checker: `gear_sonic/scripts/validate_rwm_config_parity.py`
- Transition exporter: `gear_sonic/scripts/export_rwmu_transitions.py`

The implemented adapter backend is deliberately `smoke`. It proves the original SONIC release policy/trainer stack can run without Isaac. The next implementation step is real RWM-U checkpoint loading behind the same wrapper, using exporter data from Isaac or mjlab rollouts.
