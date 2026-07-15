"""mjlab environment construction for SONIC training."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from omegaconf import DictConfig, OmegaConf

from gear_sonic.envs.wrapper.mjlab_sonic_env_wrapper import MjlabSonicEnvWrapper


def _as_plain_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def _maybe_add_mjlab_source_path(config) -> None:
    """Allow using a sibling mjlab checkout without requiring installation."""
    mjlab_cfg = config.get("mjlab_env", {})
    source_path = mjlab_cfg.get("source_path", None)
    if source_path is None:
        candidate = Path(__file__).resolve().parents[3] / "mjlab" / "src"
        if candidate.exists():
            source_path = str(candidate)
    if source_path and source_path not in sys.path:
        sys.path.insert(0, source_path)


def _resolve_motion_files(mjlab_cfg: Any) -> list[str]:
    motion_files: list[str] = []

    configured_files = mjlab_cfg.get("motion_files", None)
    if configured_files is not None:
        motion_files.extend(str(path) for path in configured_files)

    motion_dir = mjlab_cfg.get("motion_dir", None)
    if motion_dir is not None:
        motion_glob = mjlab_cfg.get("motion_glob", "**/*.npz")
        motion_files.extend(str(path) for path in sorted(Path(motion_dir).glob(motion_glob)))

    motion_file = mjlab_cfg.get("motion_file", None)
    if motion_file is not None and not motion_files:
        motion_files.append(str(motion_file))

    max_motions = mjlab_cfg.get("max_motions", None)
    if max_motions is not None:
        motion_files = motion_files[: int(max_motions)]

    missing = [path for path in motion_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"mjlab motion file(s) do not exist: {missing[:5]}")

    return motion_files


def create_mjlab_env(config, device: str) -> MjlabSonicEnvWrapper:
    """Create the first-stage G1 flat tracking mjlab environment.

    This intentionally starts from mjlab's built-in G1 tracking task so the
    MuJoCo asset, body names, action scale, observations, rewards, and
    terminations are owned by mjlab.  SONIC-specific additions should be layered
    on top after the minimal PPO loop is verified.
    """

    _maybe_add_mjlab_source_path(config)

    try:
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
    except ImportError as exc:
        raise ImportError(
            "mjlab is required for sim_type=mjlab. Install it or set "
            "mjlab_env.source_path=/path/to/mjlab/src."
        ) from exc

    mjlab_cfg = config.get("mjlab_env", {})
    env_cfg = unitree_g1_flat_tracking_env_cfg(
        has_state_estimation=mjlab_cfg.get("has_state_estimation", True),
        play=mjlab_cfg.get("play", False),
    )

    env_cfg.scene.num_envs = int(config.num_envs)
    # mjlab's seeded reset path is sensitive to invalid smoke-test motion
    # padding. Keep the default unless the mjlab config explicitly opts in.
    if mjlab_cfg.get("seed", None) is not None:
        env_cfg.seed = int(mjlab_cfg.seed)
    env_cfg.episode_length_s = float(mjlab_cfg.get("episode_length_s", env_cfg.episode_length_s))
    env_cfg.decimation = int(mjlab_cfg.get("decimation", env_cfg.decimation))
    env_cfg.auto_reset = bool(mjlab_cfg.get("auto_reset", True))
    env_cfg.scale_rewards_by_dt = bool(mjlab_cfg.get("scale_rewards_by_dt", True))

    physics_dt = mjlab_cfg.get("physics_dt", None)
    if physics_dt is not None:
        env_cfg.sim.mujoco.timestep = float(physics_dt)

    motion_files = _resolve_motion_files(mjlab_cfg)
    if not motion_files:
        raise ValueError(
            "sim_type=mjlab requires mjlab_env.motion_file, motion_files, or motion_dir pointing to mjlab tracking .npz files."
        )

    motion_cmd = env_cfg.commands["motion"]
    motion_cmd.motion_file = motion_files[0]
    if len(motion_files) > 1:
        from gear_sonic.envs.mjlab_multi_motion import MultiMotionCommandCfg

        env_cfg.commands["motion"] = MultiMotionCommandCfg(
            entity_name=motion_cmd.entity_name,
            resampling_time_range=motion_cmd.resampling_time_range,
            debug_vis=motion_cmd.debug_vis,
            pose_range=dict(motion_cmd.pose_range),
            velocity_range=dict(motion_cmd.velocity_range),
            joint_position_range=motion_cmd.joint_position_range,
            adaptive_kernel_size=motion_cmd.adaptive_kernel_size,
            adaptive_lambda=motion_cmd.adaptive_lambda,
            adaptive_uniform_ratio=motion_cmd.adaptive_uniform_ratio,
            adaptive_alpha=motion_cmd.adaptive_alpha,
            sampling_mode="uniform" if motion_cmd.sampling_mode == "adaptive" else motion_cmd.sampling_mode,
            motion_file=motion_files[0],
            motion_files=tuple(motion_files),
            anchor_body_name=motion_cmd.anchor_body_name,
            body_names=motion_cmd.body_names,
            viz=motion_cmd.viz,
        )

    render_mode = "rgb_array" if not bool(config.get("headless", True)) else None
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    wrapper_config = _as_plain_dict(mjlab_cfg.get("config", {}))
    wrapper_config["num_envs"] = int(config.num_envs)
    wrapper_config.setdefault("experiment_dir", str(config.experiment_dir))
    wrapper_config.setdefault("headless", bool(config.get("headless", True)))
    wrapper_config.setdefault("sim_type", "mjlab")
    return MjlabSonicEnvWrapper(raw_env, wrapper_config)

