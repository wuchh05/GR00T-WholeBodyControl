#!/usr/bin/env python3
"""Collect SONIC policy rollouts and export RWM-U dynamics-training tensors.

The exported ``.pt`` contains the raw trainer transition contract and a normalized
``rwm_u`` group with tensors shaped for SONIC-specific RWM-U dynamics training:

- state: ``(steps, num_envs, state_dim)``
- action: ``(steps, num_envs, action_dim)``
- extension: ``(steps, num_envs, extension_dim)``; currently total reward
- contact: ``(steps, num_envs, contact_dim)``; may be zero-width until sensors are wired
- termination: ``(steps, num_envs, 1)``; excludes timeouts
"""

from __future__ import annotations

import argparse
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


_STATE_SPECS = (
    ("root_lin_vel", ("root_lin_vel_b", "root_lin_vel_w"), 3),
    ("root_ang_vel", ("root_ang_vel_b", "root_ang_vel_w"), 3),
    ("projected_gravity", ("projected_gravity_b",), 3),
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


def _build_rwmu_groups(payload: dict[str, Any]) -> dict[str, Any]:
    steps = int(payload["steps"])
    num_envs = int(payload["num_envs"])
    robot_state = payload.get("robot_state", {})
    obs = payload.get("obs", {})
    state_parts = []
    state_terms = []
    for out_name, candidates, fallback_dim in _STATE_SPECS:
        tensor, source = _tensor_or_zeros(robot_state, candidates, steps, num_envs, fallback_dim)
        if tensor is not None:
            state_parts.append(tensor)
            state_terms.append({"name": out_name, "source": source, "dim": tensor.shape[-1]})

    state_source = "robot_state"
    if not state_parts:
        actor_obs = obs.get("actor_obs")
        if not isinstance(actor_obs, torch.Tensor):
            raise ValueError("Cannot build RWM-U state: no robot_state tensors and no actor_obs fallback")
        state_parts = [actor_obs.reshape(steps, num_envs, -1).float()]
        state_terms = [{"name": "actor_obs_fallback", "source": "obs.actor_obs", "dim": state_parts[0].shape[-1]}]
        state_source = "actor_obs_fallback"

    state = torch.cat(state_parts, dim=-1)
    action = payload["actions"].reshape(steps, num_envs, -1).float()
    rewards = payload["rewards"].float()
    if rewards.dim() == 2:
        rewards = rewards.unsqueeze(-1)
    elif rewards.dim() > 3:
        rewards = rewards.reshape(steps, num_envs, -1)
    extension = rewards[..., :1]
    contact = torch.zeros(steps, num_envs, 0, dtype=torch.float32)
    termination = (payload["dones"].bool() & ~payload["time_outs"].bool()).float()
    if termination.dim() == 2:
        termination = termination.unsqueeze(-1)

    return {
        "state": state,
        "action": action,
        "extension": extension,
        "contact": contact,
        "termination": termination,
        "schema": {
            "name": "sonic-rwmu-clean-v1",
            "state_source": state_source,
            "state_terms": state_terms,
            "action_terms": [{"name": "policy_action", "dim": action.shape[-1]}],
            "extension_terms": [{"name": "reward_total", "dim": extension.shape[-1]}],
            "contact_terms": [],
            "termination_terms": [{"name": "done_excluding_timeout", "dim": termination.shape[-1]}],
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
    env, simulation_app = _create_env(config, args.device)
    try:
        num_envs = int(env.num_envs)
        action_dim = int(env.action_space.shape[-1])
        checkpoint_paths = _select_checkpoint_paths(args)
        policies = []
        if args.action_source == "policy":
            if not checkpoint_paths:
                raise ValueError("--action-source policy requires at least one --checkpoint")
            _update_env_config_for_policy(env, config)
            policies = [_load_policy(config, env, path, args.device) for path in checkpoint_paths]

        obs = env.reset_all()
        dones_for_policy = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        records = {
            "obs": [],
            "next_obs": [],
            "actions": [],
            "action_mean": [],
            "action_std": [],
            "rewards": [],
            "dones": [],
            "time_outs": [],
            "to_log": [],
            "robot_state": [],
            "motion_state": [],
            "policy_index": [],
        }

        for step in range(args.steps):
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

            records["obs"].append(_flatten_obs_value(obs))
            records["robot_state"].append(_extract_robot_state(env))
            records["motion_state"].append(_extract_motion_state(env))
            next_obs, rewards, dones, infos = env.step(policy_state_dict)
            records["next_obs"].append(_flatten_obs_value(next_obs))
            records["actions"].append(policy_state_dict["actions"].detach().cpu())
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
            "action_mean": torch.stack(records["action_mean"], dim=0),
            "action_std": torch.stack(records["action_std"], dim=0),
            "rewards": torch.stack(records["rewards"], dim=0),
            "dones": torch.stack(records["dones"], dim=0),
            "time_outs": torch.stack(records["time_outs"], dim=0),
            "to_log": _stack_time(records["to_log"]),
            "robot_state": _stack_time(records["robot_state"]),
            "motion_state": _stack_time(records["motion_state"]),
            "policy_index": torch.stack(records["policy_index"], dim=0),
            "config": OmegaConf.to_container(config, resolve=False),
        }
        payload["rwm_u"] = _build_rwmu_groups(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.output)
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
        f"rwm_u: state={tuple(rwm['state'].shape)} action={tuple(rwm['action'].shape)} "
        f"extension={tuple(rwm['extension'].shape)} contact={tuple(rwm['contact'].shape)} "
        f"termination={tuple(rwm['termination'].shape)}"
    )


if __name__ == "__main__":
    main()
