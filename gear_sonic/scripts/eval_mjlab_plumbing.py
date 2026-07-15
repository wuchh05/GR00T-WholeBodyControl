#!/usr/bin/env python3
"""Lightweight mjlab eval plumbing for SONIC checkpoints.

This is intentionally smaller than the Isaac eval pipeline. It verifies that a
mjlab-trained SONIC checkpoint can be loaded, rolled out deterministically, and
scored with tracking metrics exposed by mjlab's MotionCommand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf, open_dict

# Match train_agent_trl.py behavior when this file is executed directly.
SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.envs.mjlab_env import create_mjlab_env
from gear_sonic.trl.trainer.ppo_trainer import PolicyAndValueWrapper
from gear_sonic.trl.utils.common import custom_instantiate, materialize_lazy_params
from gear_sonic.utils.obs_utils import get_group_term_obs_shape


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--motion-npz", type=Path, default=None)
    parser.add_argument("--mjlab-source-path", default=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _load_config(args):
    config_path = args.config or (args.checkpoint.parent / "config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    config = OmegaConf.load(config_path)
    with open_dict(config):
        config.sim_type = "mjlab"
        config.num_envs = args.num_envs
        config.headless = True
        config.use_wandb = False
        config.checkpoint = str(args.checkpoint)
        if args.motion_npz is not None:
            config.mjlab_env.motion_file = str(args.motion_npz.resolve())
        if args.mjlab_source_path is not None:
            config.mjlab_env.source_path = args.mjlab_source_path
        config.experiment_dir = str(args.checkpoint.parent)
    return config


def _prepare_env_config_dims(env, config):
    module_dim_dict = getattr(config.algo.config, "module_dim", {})
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
    env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]
    return module_dim_dict


def _load_policy_and_value(config, env, device: str, checkpoint_path: Path):
    module_dim_dict = _prepare_env_config_dims(env, config)
    policy = custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs={},
        _resolve=False,
    ).to(device)
    value_model = None
    if not getattr(config.algo.config, "distill_only", False):
        value_model = custom_instantiate(
            config.algo.config.critic,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            backbone_kwargs={},
            _resolve=False,
        ).to(device)
    materialize_lazy_params(policy, env)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "actor_model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["actor_model_state_dict"])
    elif "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
    else:
        raise KeyError("checkpoint has no actor_model_state_dict or policy_state_dict")
    if value_model is not None and "value_state_dict" in checkpoint:
        value_model.load_state_dict(checkpoint["value_state_dict"])
    return PolicyAndValueWrapper(policy, value_model), checkpoint


def _mean_metric(command, name: str):
    value = getattr(command, "metrics", {}).get(name)
    if value is None:
        return None
    return float(value.detach().mean().cpu())


def main() -> None:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    config = _load_config(args)
    env = create_mjlab_env(config, args.device)
    model, checkpoint = _load_policy_and_value(config, env, args.device, args.checkpoint)
    model.eval()
    env.set_is_evaluating(True)

    obs = env.reset_all()
    for key in obs:
        obs[key] = obs[key].to(args.device)

    rewards = []
    dones_total = 0
    finite = True
    metric_history: dict[str, list[float]] = {}
    command = env.motion_command

    with torch.no_grad():
        for _step in range(args.steps):
            model.policy.init_rollout()
            actions_out = model.policy.rollout(obs_dict=obs)
            actor_state = {
                "actions": model.policy.action_mean.detach(),
                "obs_dict": actions_out.get("obs_dict", obs),
            }
            obs, reward, done, infos = env.step(actor_state)
            for key in obs:
                obs[key] = obs[key].to(args.device)
            finite = finite and torch.isfinite(reward).all().item()
            rewards.append(float(reward.mean().detach().cpu()))
            dones_total += int(done.sum().detach().cpu())
            if command is not None:
                for name in (
                    "error_anchor_pos",
                    "error_anchor_rot",
                    "error_body_pos",
                    "error_body_rot",
                    "error_joint_pos",
                    "error_joint_vel",
                ):
                    value = _mean_metric(command, name)
                    if value is not None:
                        metric_history.setdefault(name, []).append(value)

    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_step": int(getattr(checkpoint.get("state", object()), "global_step", -1)),
        "motion_npz": str(config.mjlab_env.motion_file),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "reward_mean": float(sum(rewards) / max(len(rewards), 1)),
        "reward_min": float(min(rewards)) if rewards else None,
        "reward_max": float(max(rewards)) if rewards else None,
        "done_count": dones_total,
        "reward_finite": bool(finite),
        "command_metrics_mean": {
            key: float(sum(values) / max(len(values), 1)) for key, values in metric_history.items()
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
