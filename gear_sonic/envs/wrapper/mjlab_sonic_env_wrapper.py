"""Compatibility wrapper for running SONIC trainers on mjlab environments.

The TRL trainer in :mod:`gear_sonic` expects the IsaacLab wrapper interface:
``reset_all()``, SONIC-style observation keys (``actor_obs``/``critic_obs``),
and info dictionaries with ``episode`` and ``time_outs`` entries.  mjlab already
provides the same manager-based environment structure, but uses ``actor`` for
the policy observation group and returns terminated/truncated separately.

This wrapper keeps that translation in one place so the PPO/model code can stay
simulator-agnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - used only in minimal smoke-test envs.
    OmegaConf = None


class _AttrDict(dict):
    """Small recursive dict with attribute access for environments without OmegaConf."""

    def __init__(self, value=None):
        super().__init__()
        for key, item in dict(value or {}).items():
            self[key] = self._wrap(item)

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict):
            return cls(value)
        return value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)


class _ShapeOnlySpace:
    """Tiny observation-space stand-in used by SONIC-side aliases."""

    def __init__(self, shape):
        self.shape = tuple(shape)


def _make_config(config: dict[str, Any]):
    if OmegaConf is not None:
        return OmegaConf.create(config)
    return _AttrDict(config)


def _prod(shape) -> int:
    return int(math.prod(tuple(shape)))


class MjlabSonicEnvWrapper:
    """Adapt a ``mjlab.envs.ManagerBasedRlEnv`` to the SONIC trainer contract."""

    def __init__(self, env, config: dict[str, Any] | None = None):
        self.unwrapped = env
        # Several existing trainer paths access ``env.env.observation_space`` on
        # the Isaac wrapper.  Point that compatibility handle back to this wrapper.
        self.env = self
        self.device = torch.device(env.device)
        self.num_envs = env.num_envs
        self.is_manager_env = True
        self.is_evaluating = False
        self.use_symmetry = False
        self.motion_command = None
        self._motion_lib = None
        self.obs_buf_dict: dict[str, torch.Tensor] = {}
        self.extras: dict[str, Any] = {}
        self._converter = None
        self._last_actions = torch.zeros(
            (self.num_envs, int(env.action_space.shape[-1])),
            dtype=torch.float32,
            device=self.device,
        )

        config = dict(config or {})
        self.use_universal_token = bool(config.get("universal_token", False))
        self.actor_history_length = int(config.get("actor_history_length", 10))
        self.critic_history_length = int(config.get("critic_history_length", 10))
        self.num_future_frames = int(config.get("num_future_frames", 10))
        self.smpl_num_future_frames = int(config.get("smpl_num_future_frames", 10))
        config.setdefault("num_envs", self.num_envs)
        config.setdefault("obs", {})
        config["obs"].setdefault("obs_dict", {})
        config["obs"].setdefault("obs_dims", {})
        config["obs"].setdefault("group_obs_dims", {})
        config["obs"].setdefault("group_obs_names", {})
        config.setdefault("robot", {})
        config["robot"].setdefault("type", "g1_mjlab")
        config["robot"].setdefault("algo_obs_dim_dict", {})
        config["robot"].setdefault("actions_dim", env.action_space.shape[-1])
        config.setdefault("rewards", {})
        config["rewards"].setdefault("num_critics", 1)
        self.config = _make_config(config)

        self.observation_space = self._build_observation_space_aliases()
        self.action_space = env.action_space

        try:
            self.motion_command = env.command_manager.get_term("motion")
            self._motion_lib = getattr(self.motion_command, "motion", None)
        except Exception:
            self.motion_command = None
            self._motion_lib = None

        self._broadcast_singleton_robot_limit_buffers()
        self._validate_minimal_tracking_contract()
        if self.use_universal_token:
            self._configure_universal_token_spaces()
        self._populate_config_dims_from_spaces()

    def _build_observation_space_aliases(self) -> dict[str, Any]:
        spaces = {}
        raw_spaces = getattr(self.unwrapped.observation_space, "spaces", None)
        if raw_spaces is None:
            raw_spaces = dict(self.unwrapped.observation_space)

        for key, value in raw_spaces.items():
            if key == "actor":
                spaces["policy"] = value
            elif key == "policy":
                spaces["policy"] = value
            else:
                spaces[key] = value
        if "critic" in raw_spaces:
            spaces["critic"] = raw_spaces["critic"]
        if "policy" not in spaces and "actor" in raw_spaces:
            spaces["policy"] = raw_spaces["actor"]
        return spaces

    def _populate_config_dims_from_spaces(self) -> None:
        actor_dim = int(self.observation_space["policy"].shape[-1])
        critic_dim = int(self.observation_space["critic"].shape[-1])
        action_dim = int(self.action_space.shape[-1])

        self.config.obs.obs_dims["actor_obs"] = actor_dim
        self.config.obs.obs_dims["critic_obs"] = critic_dim
        self.config.robot.algo_obs_dim_dict["actor_obs"] = actor_dim
        self.config.robot.algo_obs_dim_dict["critic_obs"] = critic_dim
        self.config.robot.actions_dim = action_dim
        if self.use_universal_token:
            self.config.obs.group_obs_dims["tokenizer"] = dict(self._tokenizer_schema)
            self.config.obs.group_obs_names["tokenizer"] = list(self._tokenizer_schema.keys())
            self.config.obs.obs_dims["tokenizer"] = self._tokenizer_total_dim
            self.config.robot.algo_obs_dim_dict["tokenizer"] = self._tokenizer_total_dim

    def _validate_minimal_tracking_contract(self) -> None:
        if "policy" not in self.observation_space:
            raise ValueError("mjlab env must expose an 'actor' or 'policy' observation group.")
        if "critic" not in self.observation_space:
            raise ValueError("mjlab env must expose a 'critic' observation group.")
        action_dim = int(self.action_space.shape[-1])
        if action_dim != 29:
            raise ValueError(
                f"First-stage SONIC/mjlab support expects a 29-DOF G1 body policy; got action_dim={action_dim}."
            )

    def _configure_universal_token_spaces(self) -> None:
        """Expose checkpoint-compatible SONIC universal-token observation sizes."""
        self._actor_obs_dim = 930
        self._critic_obs_dim = 1645
        self._tokenizer_schema = {
            "encoder_index": (3,),
            "command_multi_future_nonflat": (self.num_future_frames, 58),
            "command_z_multi_future_nonflat": (self.num_future_frames, 1),
            "motion_anchor_ori_b_mf_nonflat": (self.num_future_frames, 6),
            "command_multi_future_lower_body": (self.num_future_frames * 12 * 2,),
            "vr_3point_local_target": (9,),
            "vr_3point_local_orn_target": (12,),
            "motion_anchor_ori_b": (6,),
            "command_z": (1,),
            "smpl_joints_multi_future_local_nonflat": (self.smpl_num_future_frames, 72),
            "smpl_root_ori_b_multi_future": (self.smpl_num_future_frames, 6),
            "joint_pos_multi_future_wrist_for_smpl": (self.smpl_num_future_frames, 6),
        }
        self._tokenizer_total_dim = sum(_prod(shape) for shape in self._tokenizer_schema.values())
        self.observation_space["policy"] = _ShapeOnlySpace((self._actor_obs_dim,))
        self.observation_space["critic"] = _ShapeOnlySpace((self._critic_obs_dim,))
        self.observation_space["tokenizer"] = _ShapeOnlySpace((self._tokenizer_total_dim,))

    def _broadcast_singleton_robot_limit_buffers(self) -> None:
        """Work around mjlab installs that leave joint limit buffers at one env."""
        try:
            robot_data = self.unwrapped.scene["robot"].data
        except Exception:
            return

        for name in (
            "default_joint_pos_limits",
            "joint_pos_limits",
            "soft_joint_pos_limits",
        ):
            value = getattr(robot_data, name, None)
            if not isinstance(value, torch.Tensor):
                continue
            if value.shape[0] == 1 and self.num_envs > 1:
                repeat_shape = (self.num_envs,) + (1,) * (value.ndim - 1)
                setattr(robot_data, name, value.repeat(repeat_shape))

    def _get_robot_data(self):
        try:
            return self.unwrapped.scene["robot"].data
        except Exception:
            return None

    def _get_converter(self):
        if self._converter is not None:
            return self._converter or None
        try:
            from gear_sonic.trl.utils.order_converter import G1Converter

            self._converter = G1Converter()
        except Exception:
            self._converter = False
        return self._converter or None

    def _pad_or_trim(self, value: torch.Tensor, dim: int) -> torch.Tensor:
        value = value.reshape(value.shape[0], -1)
        if value.shape[-1] == dim:
            return value
        if value.shape[-1] > dim:
            return value[..., :dim]
        pad = torch.zeros((value.shape[0], dim - value.shape[-1]), dtype=value.dtype, device=value.device)
        return torch.cat([value, pad], dim=-1)

    def _repeat_history(self, value: torch.Tensor, history_length: int) -> torch.Tensor:
        value = value.reshape(value.shape[0], -1)
        return value.unsqueeze(1).repeat(1, history_length, 1).reshape(value.shape[0], -1)

    def _build_universal_actor_obs(self, fallback_actor_obs: torch.Tensor) -> torch.Tensor:
        robot_data = self._get_robot_data()
        if robot_data is None:
            return self._pad_or_trim(fallback_actor_obs, self._actor_obs_dim)

        gravity = getattr(robot_data, "projected_gravity_b", None)
        if gravity is None:
            gravity = torch.zeros((self.num_envs, 3), device=self.device)
            gravity[:, 2] = -1.0
        base_ang_vel = getattr(robot_data, "root_ang_vel_b", None)
        if base_ang_vel is None:
            base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        joint_pos = getattr(robot_data, "joint_pos", None)
        joint_vel = getattr(robot_data, "joint_vel", None)
        default_joint_pos = getattr(robot_data, "default_joint_pos", None)
        if joint_pos is None or joint_vel is None:
            return self._pad_or_trim(fallback_actor_obs, self._actor_obs_dim)
        if default_joint_pos is not None and default_joint_pos.shape == joint_pos.shape:
            joint_pos = joint_pos - default_joint_pos

        parts = [
            self._repeat_history(gravity, self.actor_history_length),
            self._repeat_history(base_ang_vel, self.actor_history_length),
            self._repeat_history(joint_pos, self.actor_history_length),
            self._repeat_history(joint_vel, self.actor_history_length),
            self._repeat_history(self._last_actions, self.actor_history_length),
        ]
        return self._pad_or_trim(torch.cat(parts, dim=-1), self._actor_obs_dim)

    def _build_universal_critic_obs(self, fallback_critic_obs: torch.Tensor) -> torch.Tensor:
        # Keep the critic checkpoint-compatible for smoke finetuning. Full
        # Isaac-equivalent privileged terms remain a follow-up validation item.
        return self._pad_or_trim(fallback_critic_obs, self._critic_obs_dim)

    def _future_motion_indices(self) -> torch.Tensor:
        base = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self.motion_command is not None and hasattr(self.motion_command, "time_steps"):
            base = self.motion_command.time_steps.long()
        offsets = torch.arange(self.num_future_frames, device=self.device, dtype=torch.long)
        future = base[:, None] + offsets[None, :]
        motion = self._motion_lib
        if motion is not None and hasattr(motion, "time_step_total"):
            future = future.clamp(max=int(motion.time_step_total) - 1)
        if self.motion_command is not None and hasattr(self.motion_command, "motion_end_steps"):
            ends = self.motion_command.motion_end_steps.long().clamp(min=1)
            future = torch.minimum(future, (ends - 1)[:, None])
        return future

    def _to_isaaclab_dof_order(self, value: torch.Tensor) -> torch.Tensor:
        converter = self._get_converter()
        if converter is None:
            return value
        return converter.to_isaaclab(value)

    def _identity_rot6(self, rows: int, frames: int | None = None) -> torch.Tensor:
        base = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
        if frames is None:
            return base.repeat(rows, 1)
        return base.view(1, 1, 6).repeat(rows, frames, 1)

    def _build_tokenizer_dict(self) -> dict[str, torch.Tensor]:
        tokenizer = {
            key: torch.zeros((self.num_envs, *shape), dtype=torch.float32, device=self.device)
            for key, shape in self._tokenizer_schema.items()
        }
        tokenizer["encoder_index"][:, 0] = 1.0
        tokenizer["motion_anchor_ori_b"] = self._identity_rot6(self.num_envs)
        tokenizer["motion_anchor_ori_b_mf_nonflat"] = self._identity_rot6(
            self.num_envs, self.num_future_frames
        )
        motion = self._motion_lib
        if motion is not None and hasattr(motion, "joint_pos") and hasattr(motion, "joint_vel"):
            future = self._future_motion_indices()
            joint_pos = self._to_isaaclab_dof_order(motion.joint_pos[future])
            joint_vel = self._to_isaaclab_dof_order(motion.joint_vel[future])
            tokenizer["command_multi_future_nonflat"] = torch.cat([joint_pos, joint_vel], dim=-1)
            lower = torch.cat([joint_pos[..., :12], joint_vel[..., :12]], dim=-1)
            tokenizer["command_multi_future_lower_body"] = lower.reshape(self.num_envs, -1)
            tokenizer["joint_pos_multi_future_wrist_for_smpl"] = joint_pos[..., 23:29][
                :, : self.smpl_num_future_frames
            ]
            if hasattr(motion, "body_pos_w"):
                root_z = motion.body_pos_w[future, 0, 2:3]
                tokenizer["command_z_multi_future_nonflat"] = root_z
                tokenizer["command_z"] = root_z[:, :1, :].reshape(self.num_envs, 1)
        return tokenizer

    def _flatten_tokenizer(self, tokenizer: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [tokenizer[name].reshape(tokenizer[name].shape[0], -1) for name in self._tokenizer_schema],
            dim=-1,
        )

    def _process_raw_obs(
        self, obs: dict[str, Any], flatten_dict_obs: bool = True
    ) -> dict[str, torch.Tensor]:
        actor_obs = obs.get("policy", obs.get("actor"))
        critic_obs = obs.get("critic")
        if actor_obs is None or critic_obs is None:
            raise KeyError(
                f"Expected mjlab observation groups 'actor'/'policy' and 'critic', got: {list(obs.keys())}"
            )

        if self.use_universal_token:
            actor_obs = self._build_universal_actor_obs(actor_obs)
            critic_obs = self._build_universal_critic_obs(critic_obs)

        processed = {
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
        }
        if self.use_universal_token:
            tokenizer = self._build_tokenizer_dict()
            processed["tokenizer"] = self._flatten_tokenizer(tokenizer) if flatten_dict_obs else tokenizer
        for key, value in obs.items():
            if key not in {"actor", "policy", "critic"}:
                processed[key] = value
        self.obs_buf_dict = processed
        return processed

    def reset_all(self, global_rank: int = 0):  # noqa: ARG002
        return self.reset()

    def reset(self, flatten_dict_obs: bool = True):
        obs, extras = self.unwrapped.reset()
        self.extras = extras
        self._last_actions.zero_()
        return self._process_raw_obs(obs, flatten_dict_obs=flatten_dict_obs)

    def step(self, actions):
        if isinstance(actions, Mapping) or (
            hasattr(actions, "keys") and "actions" in actions.keys()
        ):
            env_actions = actions["actions"]
        else:
            env_actions = actions
        if env_actions.dim() == 1:
            env_actions = env_actions.unsqueeze(0)
        expected_shape = (self.num_envs, int(self.action_space.shape[-1]))
        if tuple(env_actions.shape) != expected_shape:
            raise ValueError(
                f"mjlab actions must have shape {expected_shape}, got {tuple(env_actions.shape)}"
            )
        self._last_actions = env_actions.detach()
        obs, rewards, terminated, truncated, extras = self.unwrapped.step(env_actions)
        dones = (terminated | truncated).to(dtype=torch.long)

        infos = dict(extras or {})
        infos.setdefault("episode", infos.get("log", {}))
        infos.setdefault("to_log", {})
        for key, value in infos.get("log", {}).items():
            if isinstance(value, torch.Tensor):
                infos["to_log"][key] = value
            else:
                infos["to_log"][key] = torch.tensor(value, dtype=torch.float, device=self.device)
        infos["time_outs"] = truncated.to(dtype=torch.bool)
        self.extras = infos
        return self._process_raw_obs(obs, flatten_dict_obs=True), rewards, dones, infos

    def set_is_evaluating(self, is_evaluating: bool = True, *args, **kwargs):  # noqa: ARG002
        self.is_evaluating = is_evaluating

    def set_is_training(self):
        self.is_evaluating = False

    def reinit_dr(self):
        return None

    def sync_and_compute_adaptive_sampling(self, *args, **kwargs):  # noqa: ARG002
        return None

    def resample_motion(self):
        if self.motion_command is not None and hasattr(self.motion_command, "reset"):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self.motion_command.reset(env_ids)

    def get_env_state_dict(self) -> dict[str, Any]:
        return {
            "sim_type": "mjlab",
            "episode_length_buf": self.unwrapped.episode_length_buf.detach().cpu(),
        }

    def load_env_state_dict(self, state_dict: dict[str, Any] | None):
        if not state_dict:
            return
        episode_length_buf = state_dict.get("episode_length_buf")
        if episode_length_buf is not None:
            self.unwrapped.episode_length_buf[:] = episode_length_buf.to(self.device)

    def close(self):
        return self.unwrapped.close()
