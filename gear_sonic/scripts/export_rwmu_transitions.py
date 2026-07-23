#!/usr/bin/env python3
"""Export SONIC transitions for RWM-U training.

The exporter records the SONIC trainer contract plus structured optional fields
that RWM-U can use as ``system_state``, ``system_action``, ``system_extension``,
``system_contact`` and ``system_termination`` labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

# Keep package imports stable when run as ``python gear_sonic/scripts/...``.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from gear_sonic.envs.rwm_env import create_rwm_env
from gear_sonic.utils.common import seeding


def _flatten_obs_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _flatten_obs_value(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _safe_tensor(value: Any) -> torch.Tensor | None:
    return value.detach().cpu() if isinstance(value, torch.Tensor) else None


def _extract_robot_state(env) -> dict[str, torch.Tensor]:
    """Best-effort robot state extraction from IsaacLab/mjlab-style envs."""

    candidates = [env, getattr(env, "env", None), getattr(env, "unwrapped", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            robot = candidate.scene["robot"]
            data = robot.data
            state = {}
            for name in (
                "root_pos_w",
                "root_quat_w",
                "root_lin_vel_b",
                "root_ang_vel_b",
                "root_lin_vel_w",
                "root_ang_vel_w",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "projected_gravity_b",
                "default_joint_pos",
                "default_joint_vel",
            ):
                tensor = _safe_tensor(getattr(data, name, None))
                if tensor is not None:
                    state[name] = tensor
            try:
                contact_sensor = candidate.scene["contact_forces"]
                contact_data = contact_sensor.data
                for name in ("net_forces_w", "force_matrix_w"):
                    tensor = _safe_tensor(getattr(contact_data, name, None))
                    if tensor is not None:
                        state[f"contact_forces_{name}"] = tensor
                if hasattr(contact_sensor, "body_names"):
                    state["contact_body_names"] = list(contact_sensor.body_names)
            except Exception:
                pass
            if state:
                if hasattr(robot, "joint_names"):
                    state["joint_names"] = list(robot.joint_names)
                if hasattr(robot, "body_names"):
                    state["body_names"] = list(robot.body_names)
                return state
        except Exception:
            continue
    return {}


def _extract_motion_state(env) -> dict[str, torch.Tensor]:
    command = getattr(env, "motion_command", None)
    if command is None:
        try:
            command = env.env.command_manager.get_term("motion")
        except Exception:
            command = None
    if command is None:
        return {}
    state = {}
    for name in (
        "motion_ids",
        "time_steps",
        "motion_start_time_steps",
        "motion_end_steps",
        "anchor_pos_w",
        "anchor_quat_w",
        "robot_anchor_pos_w",
        "robot_anchor_quat_w",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "robot_body_pos_w",
        "robot_body_quat_w",
        "robot_body_lin_vel_w",
        "robot_body_ang_vel_w",
        "body_pos_relative_w",
        "body_quat_relative_w",
    ):
        tensor = _safe_tensor(getattr(command, name, None))
        if tensor is not None:
            state[name] = tensor
    cfg = getattr(command, "cfg", None)
    if cfg is not None:
        for name in ("anchor_body", "body_names", "reward_point_body", "vr_3point_body"):
            if hasattr(cfg, name):
                state[name] = OmegaConf.to_container(getattr(cfg, name), resolve=True)
    return state


def _stack_time(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record})
    out = {}
    for key in keys:
        values = [record.get(key) for record in records]
        if all(isinstance(value, torch.Tensor) for value in values):
            try:
                out[key] = torch.stack(values, dim=0)
            except RuntimeError:
                out[key] = values
        else:
            out[key] = values
    return out


def _create_env(config: DictConfig, device: str):
    sim_type = str(config.get("sim_type", "isaacsim")).lower()
    if sim_type in {"rwm", "rwm-u", "rwm_u", "world_model"}:
        return create_rwm_env(config, device)
    if sim_type in {"mjlab", "mujoco"}:
        from gear_sonic.envs.mjlab_env import create_mjlab_env

        return create_mjlab_env(config, device)
    raise NotImplementedError(
        "Isaac exporter startup needs Isaac AppLauncher. Use the training script path or "
        "export from mjlab/rwm first; Isaac support should be wired with AppLauncher next."
    )


def export_transitions(config: DictConfig, output: Path, steps: int, device: str, action_mode: str):
    seeding(int(config.get("seed", 0)))
    env = _create_env(config, device)
    obs = env.reset_all()
    num_envs = int(env.num_envs)
    action_dim = int(env.action_space.shape[-1])

    obs_records = []
    next_obs_records = []
    action_records = []
    reward_records = []
    done_records = []
    timeout_records = []
    info_records = []
    robot_records = []
    motion_records = []

    for _ in range(steps):
        if action_mode == "zeros":
            actions = torch.zeros(num_envs, action_dim, dtype=torch.float32, device=env.device)
        elif action_mode == "random":
            actions = torch.randn(num_envs, action_dim, dtype=torch.float32, device=env.device)
        else:
            raise ValueError(f"unknown action_mode={action_mode}")

        obs_records.append(_flatten_obs_value(obs))
        robot_records.append(_extract_robot_state(env))
        motion_records.append(_extract_motion_state(env))
        next_obs, rewards, dones, infos = env.step({"actions": actions})
        next_obs_records.append(_flatten_obs_value(next_obs))
        action_records.append(actions.detach().cpu())
        reward_records.append(rewards.detach().cpu())
        done_records.append(dones.detach().cpu())
        timeout_records.append(infos["time_outs"].detach().cpu())
        info_records.append(
            {
                key: _flatten_obs_value(value)
                for key, value in infos.get("to_log", {}).items()
                if isinstance(value, torch.Tensor)
            }
        )
        obs = next_obs

    payload = {
        "format": "sonic-rwmu-transitions-v1",
        "sim_type": str(config.get("sim_type", "unknown")),
        "num_envs": num_envs,
        "steps": steps,
        "step_dt": float(getattr(env, "step_dt", config.get("rwm_env", {}).get("step_dt", 0.02))),
        "obs": _stack_time(obs_records),
        "next_obs": _stack_time(next_obs_records),
        "actions": torch.stack(action_records, dim=0),
        "rewards": torch.stack(reward_records, dim=0),
        "dones": torch.stack(done_records, dim=0),
        "time_outs": torch.stack(timeout_records, dim=0),
        "to_log": _stack_time(info_records),
        "robot_state": _stack_time(robot_records),
        "motion_state": _stack_time(motion_records),
        "rwm_u_groups": {
            "system_action": "actions",
            "system_state": "robot_state plus motion_state; empty for non-physical smoke backends",
            "system_extension": "to_log plus rewards; fill reward terms in physical backend",
            "system_contact": "contact fields in robot_state/motion_state when available",
            "system_termination": "dones minus time_outs",
        },
        "config": OmegaConf.to_container(config, resolve=False),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    env.close()
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-mode", choices=["zeros", "random"], default="zeros")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides, e.g. +exp=rwm/sonic_release num_envs=4 headless=true",
    )
    args = parser.parse_args()

    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    if not any(item.startswith("+exp=") or item.startswith("exp=") for item in overrides):
        overrides.insert(0, "+exp=rwm/sonic_release")
    if not any(item.startswith("experiment_dir=") or item.startswith("+experiment_dir=") for item in overrides):
        overrides.append("experiment_dir=/tmp/sonic_rwmu_export")
    config_dir = str(_REPO_ROOT / "gear_sonic" / "config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        config = compose(config_name="base", overrides=overrides)
    payload = export_transitions(config, args.output, args.steps, args.device, args.action_mode)
    print(
        f"saved {payload['steps']} steps x {payload['num_envs']} envs "
        f"to {args.output} ({payload['format']})"
    )


if __name__ == "__main__":
    main()
