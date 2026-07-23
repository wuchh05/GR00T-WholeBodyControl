#!/usr/bin/env python3
"""Collect SONIC policy rollouts and export RWM-U dynamics-training tensors.

The exported ``.pt`` contains the raw trainer transition contract and a normalized
``rwm_u`` group with tensors shaped for SONIC-specific RWM-U dynamics training:

- state: ``(steps, num_envs, state_dim)``; PhysX input state before ``env.step``
- next_state: ``(steps, num_envs, state_dim)``; PhysX output state after ``env.step``
- action: ``(steps, num_envs, action_dim)``; exact joint action handed to Isaac Lab
- contact: ``(steps, num_envs, contact_dim)``; post-step contact sensor labels
- extension: zero-width by default; reward is stored only as a diagnostic label
- termination: zero-width by default; done is stored only as a diagnostic label
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from gear_sonic.scripts.export_rwmu_transitions import (  # noqa: E402
    _extract_motion_state,
    _extract_robot_state,
    _flatten_obs_value,
    _stack_time,
)
from gear_sonic.trl.utils.common import custom_instantiate, materialize_lazy_params
from gear_sonic.utils.common import seeding
from gear_sonic.utils.config_utils import register_rl_resolvers
from gear_sonic.utils.obs_utils import get_group_term_obs_shape

register_rl_resolvers()


def _debug(message: str) -> None:
    if os.environ.get("SONIC_RWMU_DEBUG"):
        print(f"[sonic-rwmu-export] {message}", flush=True)


_PHYSX_STATE_SPECS = (
    ("root_pos_w", ("root_pos_w",), 3),
    ("root_quat_w", ("root_quat_w",), 4),
    ("root_lin_vel_w", ("root_lin_vel_w",), 3),
    ("root_ang_vel_w", ("root_ang_vel_w",), 3),
    ("root_lin_vel_b", ("root_lin_vel_b",), 3),
    ("root_ang_vel_b", ("root_ang_vel_b",), 3),
    ("projected_gravity_b", ("projected_gravity_b",), 3),
    ("joint_pos", ("joint_pos",), None),
    ("joint_vel", ("joint_vel",), None),
    ("body_pos_w", ("body_pos_w",), None),
    ("body_quat_w", ("body_quat_w",), None),
    ("body_lin_vel_w", ("body_lin_vel_w",), None),
    ("body_ang_vel_w", ("body_ang_vel_w",), None),
)


def _safe_close_sim(simulation_app: Any) -> None:
    if simulation_app is not None:
        try:
            simulation_app.close()
        except Exception:
            pass


def _launch_isaac_app(config: DictConfig, device: str):
    try:
        with open("./rl/simulator/isaacsim/.isaacsim_version", encoding="utf-8") as f:
            version = f.read().strip()
    except FileNotFoundError:
        version = "4.5"

    if version == "4.2":
        from omni.isaac.lab.app import AppLauncher
    else:
        from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args([])
    env_config = config.manager_env
    args_cli.num_envs = int(config.num_envs)
    args_cli.seed = int(config.seed)
    args_cli.env_spacing = env_config.config.env_spacing
    args_cli.output_dir = config.output_dir
    args_cli.enable_cameras = False
    args_cli.headless = bool(config.headless)
    args_cli.multi_gpu = False
    args_cli.distributed = False
    args_cli.device = device
    args_cli.kit_args = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
    app_launcher = AppLauncher(args_cli)
    return app_launcher.app, args_cli


def _create_env(config: DictConfig, device: str):
    sim_type = str(config.get("sim_type", "isaacsim")).lower()
    simulation_app = None
    args_cli = SimpleNamespace(headless=bool(config.headless))
    if sim_type in {"rwm", "rwm-u", "rwm_u", "world_model"}:
        from gear_sonic.envs.rwm_env import create_rwm_env

        return create_rwm_env(config, device), simulation_app
    if sim_type in {"mjlab", "mujoco"}:
        from gear_sonic.envs.mjlab_env import create_mjlab_env

        return create_mjlab_env(config, device), simulation_app

    simulation_app, args_cli = _launch_isaac_app(config, device)
    from gear_sonic.train_agent_trl import create_manager_env

    return create_manager_env(config, device, args_cli), simulation_app


def _update_env_config_for_policy(env, config: DictConfig) -> dict[str, Any]:
    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key not in ["policy", "critic"]:
            group_obs_dims, group_obs_names, group_obs_total_dim = get_group_term_obs_shape(example_obs, key)
            env.config["obs"]["group_obs_dims"][key] = group_obs_dims
            env.config["obs"]["group_obs_names"][key] = group_obs_names
            env.config["obs"]["obs_dims"][key] = group_obs_total_dim
            env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim
    manager_env_config = config.get("manager_env", {}).get("config", {})
    if manager_env_config.get("meta_action_dim", None) is not None:
        env.config["robot"]["actions_dim"] = manager_env_config.meta_action_dim
    else:
        env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]
    return example_obs


def _load_policy(config: DictConfig, env, checkpoint_path: Path, device: str):
    # Registers trl.trainer.utils.OnlineTrainerState for older SONIC checkpoints.
    import gear_sonic.trl.trainer.ppo_trainer  # noqa: F401

    module_dim_dict = getattr(config.algo.config, "module_dim", {})
    policy = custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs={},
        _resolve=False,
    ).to(device)
    materialize_lazy_params(policy, env)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "actor_model_state_dict" in checkpoint:
        state_dict = checkpoint["actor_model_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        raise KeyError(f"{checkpoint_path} does not contain actor_model_state_dict or policy_state_dict")

    model_uses_std = "std" in policy.state_dict()
    checkpoint_has_std = "std" in state_dict
    checkpoint_has_log_std = "log_std" in state_dict
    if model_uses_std and checkpoint_has_log_std and not checkpoint_has_std:
        state_dict = dict(state_dict)
        state_dict["std"] = torch.exp(state_dict.pop("log_std"))
    elif not model_uses_std and checkpoint_has_std and not checkpoint_has_log_std:
        state_dict = dict(state_dict)
        state_dict["log_std"] = torch.log(state_dict.pop("std"))
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[policy load] {checkpoint_path}: missing={len(missing)} unexpected={len(unexpected)}")
    policy.eval()
    policy.init_rollout()
    return policy


def _tensor_or_zeros(record: dict[str, Any], names: tuple[str, ...], steps: int, num_envs: int, dim: int | None):
    for name in names:
        value = record.get(name)
        if isinstance(value, torch.Tensor):
            return value.reshape(steps, num_envs, -1).float(), name
    if dim is None:
        return None, None
    return torch.zeros(steps, num_envs, dim), f"missing_zero_fallback:{names[0]}"


def _flatten_physx_state(record: dict[str, Any], steps: int, num_envs: int):
    parts = []
    terms = []
    for out_name, candidates, fallback_dim in _PHYSX_STATE_SPECS:
        tensor, source = _tensor_or_zeros(record, candidates, steps, num_envs, fallback_dim)
        if tensor is not None:
            parts.append(tensor)
            terms.append({"name": out_name, "source": source, "dim": tensor.shape[-1]})
    if not parts:
        raise ValueError("Cannot build PhysX RWM-U state: no robot_state tensors were exported")
    return torch.cat(parts, dim=-1), terms


def _extract_action_to_sim(env: Any, fallback: torch.Tensor) -> torch.Tensor:
    for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None)):
        value = getattr(candidate, "_last_env_actions_to_sim", None)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
    return fallback.detach().cpu()


def _get_action_dim(env: Any) -> int:
    try:
        value = env.config.get("robot", {}).get("actions_dim")
        if value is not None:
            return int(value)
    except Exception:
        pass
    for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None)):
        action_manager = getattr(candidate, "action_manager", None)
        action = getattr(action_manager, "action", None)
        if isinstance(action, torch.Tensor) and action.dim() >= 2:
            return int(action.shape[-1])
    return int(env.action_space.shape[-1])


def _build_rwmu_groups(payload: dict[str, Any]) -> dict[str, Any]:
    steps = int(payload["steps"])
    num_envs = int(payload["num_envs"])
    robot_state = payload.get("robot_state", {})
    next_robot_state = payload.get("next_robot_state", {})

    state, state_terms = _flatten_physx_state(robot_state, steps, num_envs)
    next_state, next_state_terms = _flatten_physx_state(next_robot_state, steps, num_envs)
    if state.shape[-1] != next_state.shape[-1]:
        raise ValueError(f"state_dim={state.shape[-1]} does not match next_state_dim={next_state.shape[-1]}")

    action = payload.get("env_actions_to_sim", payload["actions"]).reshape(steps, num_envs, -1).float()

    contact_terms = []
    contact_force = next_robot_state.get("contact_forces_net_forces_w")
    if isinstance(contact_force, torch.Tensor):
        force = contact_force.reshape(steps, num_envs, -1, 3).float()
        contact = (torch.linalg.norm(force, dim=-1) > 1.0).float()
        contact_terms.append({
            "name": "contact_forces",
            "source": "next_robot_state.contact_forces_net_forces_w",
            "dim": contact.shape[-1],
        })
    else:
        contact = torch.zeros(steps, num_envs, 0, dtype=torch.float32)

    reward = payload["rewards"].float()
    if reward.dim() == 2:
        reward = reward.unsqueeze(-1)
    elif reward.dim() > 3:
        reward = reward.reshape(steps, num_envs, -1)
    done_excluding_timeout = (payload["dones"].bool() & ~payload["time_outs"].bool()).float()
    if done_excluding_timeout.dim() == 2:
        done_excluding_timeout = done_excluding_timeout.unsqueeze(-1)

    extension = torch.zeros(steps, num_envs, 0, dtype=torch.float32)
    termination = torch.zeros(steps, num_envs, 0, dtype=torch.float32)

    return {
        "state": state,
        "next_state": next_state,
        "action": action,
        "extension": extension,
        "contact": contact,
        "termination": termination,
        "diagnostics": {
            "reward_total": reward[..., :1],
            "done_excluding_timeout": done_excluding_timeout,
        },
        "schema": {
            "name": "sonic-rwmu-physx-v1",
            "contract": "RWM-U learns Isaac/PhysX transition only: physics_state_t + env_action_t -> physics_state_t_plus_1 and contact_t_plus_1. Reward/done stay diagnostics for SONIC manager parity checks.",
            "state_source": "robot_state_before_step",
            "next_state_source": "robot_state_after_step",
            "state_terms": state_terms,
            "next_state_terms": next_state_terms,
            "action_terms": [{"name": "isaac_joint_position_action", "source": "ManagerEnvWrapper._last_env_actions_to_sim", "dim": action.shape[-1]}],
            "extension_terms": [],
            "contact_terms": contact_terms,
            "termination_terms": [],
            "diagnostic_terms": [
                {"name": "reward_total", "source": "Isaac reward_manager output", "dim": reward[..., :1].shape[-1]},
                {"name": "done_excluding_timeout", "source": "terminated_or_truncated minus timeout", "dim": done_excluding_timeout.shape[-1]},
            ],
        },
    }


def _select_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item) for item in args.checkpoint]
    if args.checkpoint_list is not None:
        for line in args.checkpoint_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line))
    return paths


def collect(config: DictConfig, args: argparse.Namespace) -> dict[str, Any]:
    seeding(int(config.get("seed", 0)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    _debug("creating env")
    env, simulation_app = _create_env(config, args.device)
    _debug("env created")
    try:
        _debug("read num_envs begin")
        num_envs = int(env.num_envs)
        _debug(f"read num_envs done: {num_envs}")
        _debug("read action_dim begin")
        action_dim = _get_action_dim(env)
        _debug(f"read action_dim done: {action_dim}")
        checkpoint_paths = _select_checkpoint_paths(args)
        policies = []
        if args.action_source == "policy":
            if not checkpoint_paths:
                raise ValueError("--action-source policy requires at least one --checkpoint")
            _debug("updating env config for policy")
            _update_env_config_for_policy(env, config)
            _debug(f"loading {len(checkpoint_paths)} policy checkpoint(s)")
            policies = [_load_policy(config, env, path, args.device) for path in checkpoint_paths]
            _debug("policies loaded")

        _debug("reset_all begin")
        obs = env.reset_all()
        _debug("reset_all done")
        dones_for_policy = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        records = {
            "obs": [],
            "next_obs": [],
            "actions": [],
            "env_actions_to_sim": [],
            "action_mean": [],
            "action_std": [],
            "rewards": [],
            "dones": [],
            "time_outs": [],
            "to_log": [],
            "robot_state": [],
            "next_robot_state": [],
            "motion_state": [],
            "policy_index": [],
        }

        for step in range(args.steps):
            _debug(f"step {step} action_select begin")
            if args.action_source == "zeros":
                policy_state_dict = {"actions": torch.zeros(num_envs, action_dim, device=env.device)}
                selected_policy = -1
            elif args.action_source == "random":
                policy_state_dict = {"actions": torch.randn(num_envs, action_dim, device=env.device)}
                selected_policy = -1
            else:
                policy_outputs = []
                with torch.no_grad():
                    for policy in policies:
                        out = policy.rollout(obs_dict=obs.copy(), cur_dones=dones_for_policy)
                        if not args.stochastic:
                            out["actions"] = out["action_mean"]
                        policy_outputs.append(out)
                if args.policy_selection == "random_step":
                    selected_policy = random.randrange(len(policy_outputs))
                else:
                    selected_policy = step % len(policy_outputs)
                policy_state_dict = policy_outputs[selected_policy]

            _debug(f"step {step} flatten obs begin")
            records["obs"].append(_flatten_obs_value(obs))
            _debug(f"step {step} flatten obs done")
            _debug(f"step {step} extract robot_state begin")
            records["robot_state"].append(_extract_robot_state(env))
            _debug(f"step {step} extract robot_state done")
            if args.record_motion_state:
                _debug(f"step {step} extract motion_state begin")
                records["motion_state"].append(_extract_motion_state(env))
                _debug(f"step {step} extract motion_state done")
            else:
                records["motion_state"].append({})
            _debug(f"step {step} env.step begin")
            next_obs, rewards, dones, infos = env.step(policy_state_dict)
            _debug(f"step {step} env.step done")
            records["next_obs"].append(_flatten_obs_value(next_obs))
            records["next_robot_state"].append(_extract_robot_state(env))
            records["actions"].append(policy_state_dict["actions"].detach().cpu())
            records["env_actions_to_sim"].append(_extract_action_to_sim(env, policy_state_dict["actions"]))
            records["action_mean"].append(_flatten_obs_value(policy_state_dict.get("action_mean", policy_state_dict["actions"])))
            records["action_std"].append(_flatten_obs_value(policy_state_dict.get("action_sigma", torch.zeros_like(policy_state_dict["actions"]))))
            records["rewards"].append(rewards.detach().cpu())
            records["dones"].append(dones.detach().cpu())
            records["time_outs"].append(infos["time_outs"].detach().cpu())
            records["to_log"].append(
                {key: _flatten_obs_value(value) for key, value in infos.get("to_log", {}).items() if isinstance(value, torch.Tensor)}
            )
            records["policy_index"].append(torch.full((num_envs,), selected_policy, dtype=torch.long))
            obs = next_obs
            dones_for_policy = dones.bool().to(env.device)

        payload = {
            "format": "sonic-rwmu-policy-rollout-v1",
            "sim_type": str(config.get("sim_type", "unknown")),
            "action_source": args.action_source,
            "checkpoints": [str(path) for path in checkpoint_paths],
            "num_envs": num_envs,
            "steps": args.steps,
            "step_dt": float(getattr(env, "step_dt", config.get("rwm_env", {}).get("step_dt", 0.02))),
            "obs": _stack_time(records["obs"]),
            "next_obs": _stack_time(records["next_obs"]),
            "actions": torch.stack(records["actions"], dim=0),
            "env_actions_to_sim": torch.stack(records["env_actions_to_sim"], dim=0),
            "action_mean": torch.stack(records["action_mean"], dim=0),
            "action_std": torch.stack(records["action_std"], dim=0),
            "rewards": torch.stack(records["rewards"], dim=0),
            "dones": torch.stack(records["dones"], dim=0),
            "time_outs": torch.stack(records["time_outs"], dim=0),
            "to_log": _stack_time(records["to_log"]),
            "robot_state": _stack_time(records["robot_state"]),
            "next_robot_state": _stack_time(records["next_robot_state"]),
            "motion_state": _stack_time(records["motion_state"]),
            "policy_index": torch.stack(records["policy_index"], dim=0),
            "config": OmegaConf.to_container(config, resolve=False),
        }
        _debug("building rwm_u groups")
        payload["rwm_u"] = _build_rwmu_groups(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _debug(f"saving {args.output}")
        torch.save(payload, args.output)
        _debug("save done")
        return payload
    finally:
        try:
            env.close()
        finally:
            _safe_close_sim(simulation_app)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-source", choices=["policy", "random", "zeros"], default="policy")
    parser.add_argument("--checkpoint", action="append", default=[], help="Policy checkpoint path. Repeat for diverse policies.")
    parser.add_argument("--checkpoint-list", type=Path, default=None, help="Text file with one checkpoint path per line.")
    parser.add_argument("--policy-selection", choices=["step_cycle", "random_step"], default="step_cycle")
    parser.add_argument("--stochastic", action="store_true", help="Sample stochastic policy actions instead of action_mean.")
    parser.add_argument("--record-motion-state", action="store_true", help="Also export motion command internals; disabled by default because PhysX-only RWM-U does not train on them.")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    if not any(item.startswith("+exp=") or item.startswith("exp=") for item in overrides):
        overrides.insert(0, "+exp=manager/universal_token/all_modes/sonic_release")
    if not any(item.startswith("experiment_dir=") or item.startswith("+experiment_dir=") for item in overrides):
        overrides.append("experiment_dir=/tmp/sonic_rwmu_collect")
    config_dir = str(_REPO_ROOT / "gear_sonic" / "config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        config = compose(config_name="base", overrides=overrides)

    payload = collect(config, args)
    rwm = payload["rwm_u"]
    print(
        f"saved {payload['steps']} steps x {payload['num_envs']} envs to {args.output}\n"
        f"rwm_u: state={tuple(rwm['state'].shape)} next_state={tuple(rwm['next_state'].shape)} "
        f"action={tuple(rwm['action'].shape)} extension={tuple(rwm['extension'].shape)} contact={tuple(rwm['contact'].shape)} "
        f"termination={tuple(rwm['termination'].shape)}"
    )


if __name__ == "__main__":
    main()
