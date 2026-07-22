"""RWM/RWM-U environment adapter entrypoint for SONIC training.

This module intentionally starts with a smoke-test backend. The real RWM-U
model can be plugged in behind the same SONIC trainer contract once the exported
SONIC transition schema and model checkpoint are available.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch


class _ShapeOnlySpace:
    """Minimal space object exposing the shape attribute used by the trainer."""

    def __init__(self, shape):
        self.shape = tuple(shape)


def _as_plain_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def _prod(shape) -> int:
    return int(math.prod(tuple(shape)))


def _default_tokenizer_schema(
    encoder_count: int = 3,
    num_future_frames: int = 10,
    smpl_num_future_frames: int = 10,
) -> dict[str, tuple[int, ...]]:
    """SONIC release tokenizer schema used by ``unitoken_all_noz``."""

    return {
        "encoder_index": (encoder_count,),
        "command_multi_future_nonflat": (num_future_frames, 58),
        "command_z_multi_future_nonflat": (num_future_frames, 1),
        "motion_anchor_ori_b_mf_nonflat": (num_future_frames, 6),
        "command_multi_future_lower_body": (num_future_frames * 12 * 2,),
        "vr_3point_local_target": (9,),
        "vr_3point_local_orn_target": (12,),
        "motion_anchor_ori_b": (6,),
        "command_z": (1,),
        "smpl_joints_multi_future_local_nonflat": (smpl_num_future_frames, 72),
        "smpl_root_ori_b_multi_future": (smpl_num_future_frames, 6),
        "joint_pos_multi_future_wrist_for_smpl": (smpl_num_future_frames, 6),
    }


class RwmSonicEnvWrapper:
    """SONIC-compatible VecEnv wrapper for imagined RWM/RWM-U rollouts.

    Contract consumed by :mod:`gear_sonic.trl.trainer.ppo_trainer`:
    - ``reset_all()`` returns ``{"actor_obs": ..., "critic_obs": ...}``
    - ``step(policy_state_dict)`` returns ``obs_dict, rewards, dones, infos``
    - ``infos`` contains ``episode``, ``to_log`` and ``time_outs``.

    The default backend is ``smoke``: it keeps a deterministic tensor state and
    emits valid tensors without depending on Isaac, mjlab, or an RWM checkpoint.
    """

    def __init__(self, config: dict[str, Any], device: str):
        self.env = self
        self.unwrapped = self
        self.device = torch.device(device)
        self.is_manager_env = True
        self.is_evaluating = False
        self.use_symmetry = False
        self.motion_command = None
        self._motion_lib = None

        self.num_envs = int(config.get("num_envs", 1))
        self.backend = str(config.get("backend", "smoke"))
        self.use_universal_token = bool(config.get("universal_token", False))
        if self.use_universal_token:
            config.setdefault("actor_obs_dim", 930)
            config.setdefault("critic_obs_dim", 1645)
            config.setdefault("action_dim", 29)
        self.actor_obs_dim = int(config.get("actor_obs_dim", 137))
        self.critic_obs_dim = int(config.get("critic_obs_dim", 221))
        self.action_dim = int(config.get("action_dim", 29))
        self.num_critics = int(config.get("num_critics", 1))
        self.step_dt = float(config.get("step_dt", 0.02))
        episode_length_s = float(config.get("episode_length_s", 10.0))
        self.max_episode_length = max(1, int(round(episode_length_s / self.step_dt)))
        self.reward_scale = float(config.get("smoke_reward_scale", 1.0))

        if self.backend != "smoke":
            raise NotImplementedError(
                "Only rwm_env.backend=smoke is wired today. Train/export a SONIC RWM-U "
                "checkpoint first, then implement the backend loader behind this wrapper."
            )

        encoder_probs = dict(config.get("encoder_sample_probs", {"g1": 1.0, "teleop": 1.0, "smpl": 1.0}))
        self.encoder_names = tuple(encoder_probs.keys()) or ("g1", "teleop", "smpl")
        self.tokenizer_schema = _default_tokenizer_schema(
            encoder_count=len(self.encoder_names),
            num_future_frames=int(config.get("num_future_frames", 10)),
            smpl_num_future_frames=int(config.get("smpl_num_future_frames", 10)),
        )
        self.tokenizer_total_dim = sum(_prod(shape) for shape in self.tokenizer_schema.values())

        config.setdefault("obs", {})
        config["obs"].setdefault("obs_dict", {})
        config["obs"].setdefault("obs_dims", {})
        config["obs"].setdefault("group_obs_dims", {})
        config["obs"].setdefault("group_obs_names", {})
        config.setdefault("robot", {})
        config["robot"].setdefault("type", "g1_rwm")
        config["robot"].setdefault("algo_obs_dim_dict", {})
        config["robot"]["actions_dim"] = self.action_dim
        config["robot"]["algo_obs_dim_dict"]["actor_obs"] = self.actor_obs_dim
        config["robot"]["algo_obs_dim_dict"]["critic_obs"] = self.critic_obs_dim
        config.setdefault("rewards", {})
        config["rewards"]["num_critics"] = self.num_critics
        config["obs"]["obs_dims"]["actor_obs"] = self.actor_obs_dim
        config["obs"]["obs_dims"]["critic_obs"] = self.critic_obs_dim
        if self.use_universal_token:
            config["obs"]["group_obs_dims"]["tokenizer"] = dict(self.tokenizer_schema)
            config["obs"]["group_obs_names"]["tokenizer"] = list(self.tokenizer_schema.keys())
            config["obs"]["obs_dims"]["tokenizer"] = self.tokenizer_total_dim
            config["robot"]["algo_obs_dim_dict"]["tokenizer"] = self.tokenizer_total_dim
        self.config = OmegaConf.create(config)

        self.observation_space = {
            "policy": _ShapeOnlySpace((self.actor_obs_dim,)),
            "critic": _ShapeOnlySpace((self.critic_obs_dim,)),
        }
        if self.use_universal_token:
            self.observation_space["tokenizer"] = _ShapeOnlySpace((self.tokenizer_total_dim,))
        self.action_space = _ShapeOnlySpace((self.action_dim,))

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._actor_state = torch.zeros(
            self.num_envs, self.actor_obs_dim, dtype=torch.float32, device=self.device
        )
        self._critic_state = torch.zeros(
            self.num_envs, self.critic_obs_dim, dtype=torch.float32, device=self.device
        )
        self._last_actions = torch.zeros(
            self.num_envs, self.action_dim, dtype=torch.float32, device=self.device
        )
        self._encoder_index = torch.zeros(
            self.num_envs, len(self.encoder_names), dtype=torch.float32, device=self.device
        )
        if "g1" in self.encoder_names:
            self._encoder_index[:, self.encoder_names.index("g1")] = 1.0
        elif len(self.encoder_names) > 0:
            self._encoder_index[:, 0] = 1.0
        self.obs_buf_dict: dict[str, torch.Tensor] = {}
        self.extras: dict[str, Any] = {}

    def _build_tokenizer_dict(self) -> dict[str, torch.Tensor]:
        tokenizer = {
            key: torch.zeros((self.num_envs, *shape), dtype=torch.float32, device=self.device)
            for key, shape in self.tokenizer_schema.items()
        }
        tokenizer["encoder_index"] = self._encoder_index
        identity_6d = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=self.device
        )
        tokenizer["motion_anchor_ori_b"] = identity_6d.repeat(self.num_envs, 1)
        tokenizer["motion_anchor_ori_b_mf_nonflat"] = identity_6d.view(1, 1, 6).repeat(
            self.num_envs, self.tokenizer_schema["motion_anchor_ori_b_mf_nonflat"][0], 1
        )
        tokenizer["smpl_root_ori_b_multi_future"] = identity_6d.view(1, 1, 6).repeat(
            self.num_envs, self.tokenizer_schema["smpl_root_ori_b_multi_future"][0], 1
        )
        return tokenizer

    def _flatten_tokenizer(self, tokenizer: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [tokenizer[name].reshape(self.num_envs, -1) for name in self.tokenizer_schema], dim=-1
        )

    def _obs(self, flatten_dict_obs: bool = True) -> dict[str, torch.Tensor]:
        self.obs_buf_dict = {
            "actor_obs": self._actor_state.clone(),
            "critic_obs": self._critic_state.clone(),
        }
        if self.use_universal_token:
            tokenizer = self._build_tokenizer_dict()
            self.obs_buf_dict["tokenizer"] = (
                self._flatten_tokenizer(tokenizer) if flatten_dict_obs else tokenizer
            )
        return self.obs_buf_dict

    def reset_all(self, global_rank: int = 0):  # noqa: ARG002
        return self.reset()

    def reset(self, flatten_dict_obs: bool = True):
        self.episode_length_buf.zero_()
        self._actor_state.zero_()
        self._critic_state.zero_()
        self._last_actions.zero_()
        self.extras = {
            "episode": {},
            "to_log": {},
            "time_outs": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }
        return self._obs(flatten_dict_obs=flatten_dict_obs)

    def step(self, actions):
        if isinstance(actions, Mapping) or (
            hasattr(actions, "keys") and "actions" in actions.keys()
        ):
            env_actions = actions["actions"]
        else:
            env_actions = actions
        if env_actions.dim() == 1:
            env_actions = env_actions.unsqueeze(0)
        expected_shape = (self.num_envs, self.action_dim)
        if tuple(env_actions.shape) != expected_shape:
            raise ValueError(
                f"rwm actions must have shape {expected_shape}, got {tuple(env_actions.shape)}"
            )

        env_actions = env_actions.to(self.device)
        self.episode_length_buf += 1

        action_delta = env_actions - self._last_actions
        self._actor_state = torch.roll(self._actor_state, shifts=-self.action_dim, dims=1)
        copy_dim = min(self.action_dim, self.actor_obs_dim)
        self._actor_state[:, -copy_dim:] = env_actions[:, :copy_dim]
        self._critic_state[:, : self.actor_obs_dim] = self._actor_state
        self._last_actions = env_actions.detach()

        rewards = -self.reward_scale * torch.mean(env_actions.square(), dim=1, keepdim=True)
        if self.num_critics == 1:
            rewards_out = rewards.squeeze(-1)
        else:
            rewards_out = rewards.repeat(1, self.num_critics)

        time_outs = self.episode_length_buf >= self.max_episode_length
        dones = time_outs.to(dtype=torch.long)
        reset_ids = time_outs.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self.episode_length_buf[reset_ids] = 0
            self._actor_state[reset_ids] = 0.0
            self._critic_state[reset_ids] = 0.0
            self._last_actions[reset_ids] = 0.0

        infos = {
            "episode": {
                "rwm_smoke_reward": rewards.mean().detach(),
                "rwm_smoke_action_rate": torch.mean(action_delta.square()).detach(),
            },
            "to_log": {
                "rwm/smoke_reward": rewards.mean().detach(),
                "rwm/smoke_action_rate": torch.mean(action_delta.square()).detach(),
            },
            "time_outs": time_outs,
        }
        self.extras = infos
        return self._obs(flatten_dict_obs=True), rewards_out, dones, infos

    def set_is_evaluating(self, is_evaluating: bool = True, *args, **kwargs):  # noqa: ARG002
        self.is_evaluating = is_evaluating

    def set_is_training(self):
        self.is_evaluating = False

    def reinit_dr(self):
        return None

    def sync_and_compute_adaptive_sampling(self, *args, **kwargs):  # noqa: ARG002
        return None

    def resample_motion(self):
        return None

    def get_env_state_dict(self) -> dict[str, Any]:
        return {
            "sim_type": "rwm",
            "backend": self.backend,
            "episode_length_buf": self.episode_length_buf.detach().cpu(),
        }

    def load_env_state_dict(self, state_dict: dict[str, Any] | None):
        if not state_dict:
            return
        episode_length_buf = state_dict.get("episode_length_buf")
        if episode_length_buf is not None:
            self.episode_length_buf[:] = episode_length_buf.to(self.device)

    def close(self):
        return None


def create_rwm_env(config, device: str) -> RwmSonicEnvWrapper:
    """Create a SONIC-compatible RWM/RWM-U imagined environment."""

    rwm_cfg = _as_plain_dict(config.get("rwm_env", {}))
    rwm_cfg["num_envs"] = int(config.num_envs)
    manager_motion_cfg = config.get("manager_env", {}).get("commands", {}).get("motion", {})
    for key in (
        "num_future_frames",
        "smpl_num_future_frames",
        "encoder_sample_probs",
        "teleop_sample_prob_when_smpl",
        "optimize_encoders_ratio_for_CHIP",
    ):
        if key in manager_motion_cfg and key not in rwm_cfg:
            value = manager_motion_cfg[key]
            rwm_cfg[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    rwm_cfg.setdefault("headless", bool(config.get("headless", True)))
    rwm_cfg.setdefault("experiment_dir", str(config.experiment_dir))
    rwm_cfg.setdefault("sim_type", "rwm")
    return RwmSonicEnvWrapper(rwm_cfg, device)
