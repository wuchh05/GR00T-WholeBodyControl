"""RWM/RWM-U environment adapter entrypoint for SONIC training.

The ``smoke`` backend is a deterministic tensor environment for trainer plumbing.
The ``rwmu`` backend loads a SONIC RWM-U dynamics checkpoint produced by
``gear_sonic/scripts/train_sonic_rwmu_dynamics.py`` and uses it as a minimal
learned simulator behind the same SONIC trainer contract.
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

        self.rwmu_checkpoint_path = config.get("checkpoint", config.get("checkpoint_path", None))
        self.rwmu_model = None
        self.rwmu_ckpt: dict[str, Any] | None = None
        self.rwmu_state_dim = 0
        self.rwmu_extension_dim = 0
        self.rwmu_contact_dim = 0
        self.rwmu_termination_dim = 0
        self.rwmu_history_horizon = 1
        if self.backend not in {"smoke", "rwmu", "rwm-u", "rwm_u"}:
            raise ValueError(f"unknown rwm_env.backend={self.backend}")
        if self.backend in {"rwmu", "rwm-u", "rwm_u"}:
            self._load_rwmu_backend()

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
        self._rwmu_state_norm = torch.zeros(
            self.num_envs, self.rwmu_state_dim, dtype=torch.float32, device=self.device
        ) if self.rwmu_state_dim > 0 else None
        self._rwmu_state_history = torch.zeros(
            self.num_envs, self.rwmu_history_horizon, self.rwmu_state_dim, dtype=torch.float32, device=self.device
        ) if self.rwmu_state_dim > 0 else None
        self._rwmu_action_history = torch.zeros(
            self.num_envs, self.rwmu_history_horizon, self.action_dim, dtype=torch.float32, device=self.device
        ) if self.rwmu_state_dim > 0 else None
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


    def _load_rwmu_backend(self) -> None:
        if self.rwmu_checkpoint_path is None:
            raise ValueError("rwm_env.backend=rwmu requires rwm_env.checkpoint=/path/to/sonic_rwmu.pt")
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        rsl_root = repo_root / "external_dependencies" / "rsl_rl_rwm"
        if str(rsl_root) not in sys.path:
            sys.path.insert(0, str(rsl_root))
        from rsl_rl.modules import SystemDynamicsEnsemble

        ckpt = torch.load(self.rwmu_checkpoint_path, map_location=self.device, weights_only=False)
        dims = ckpt["dims"]
        if int(dims["action_dim"]) != self.action_dim:
            raise ValueError(
                f"RWM-U action_dim={dims['action_dim']} does not match SONIC action_dim={self.action_dim}"
            )
        self.rwmu_ckpt = ckpt
        self.rwmu_state_dim = int(dims["state_dim"])
        self.rwmu_extension_dim = int(dims.get("extension_dim", 0))
        self.rwmu_contact_dim = int(dims.get("contact_dim", 0))
        self.rwmu_termination_dim = int(dims.get("termination_dim", 0))
        self.rwmu_history_horizon = int(ckpt.get("history_horizon", 1))
        self.rwmu_model = SystemDynamicsEnsemble(
            state_dim=self.rwmu_state_dim,
            action_dim=self.action_dim,
            extension_dim=self.rwmu_extension_dim,
            contact_dim=self.rwmu_contact_dim,
            termination_dim=self.rwmu_termination_dim,
            device=str(self.device),
            ensemble_size=int(ckpt.get("ensemble_size", 1)),
            history_horizon=self.rwmu_history_horizon,
            architecture_config=ckpt["architecture_config"],
        ).to(self.device)
        self.rwmu_model.load_state_dict(ckpt["system_dynamics_state_dict"])
        self.rwmu_model.eval()
        self.rwmu_state_mean = ckpt["state_data_mean"].to(self.device).float()
        self.rwmu_state_std = ckpt["state_data_std"].to(self.device).float().clamp_min(1.0e-6)
        self.rwmu_action_mean = ckpt["action_data_mean"].to(self.device).float()
        self.rwmu_action_std = ckpt["action_data_std"].to(self.device).float().clamp_min(1.0e-6)

    def _rwmu_denormalized_state(self) -> torch.Tensor | None:
        if self._rwmu_state_norm is None:
            return None
        return self._rwmu_state_norm * self.rwmu_state_std + self.rwmu_state_mean

    def _sync_obs_from_rwmu_state(self) -> None:
        state = self._rwmu_denormalized_state()
        if state is None:
            return
        state_dim = min(state.shape[-1], self.actor_obs_dim)
        self._actor_state[:, :state_dim] = state[:, :state_dim]
        critic_dim = min(state.shape[-1], self.critic_obs_dim)
        self._critic_state[:, :critic_dim] = state[:, :critic_dim]

    @torch.no_grad()
    def _rwmu_step(self, env_actions: torch.Tensor):
        assert self.rwmu_model is not None
        action_norm = (env_actions - self.rwmu_action_mean) / self.rwmu_action_std
        if self._rwmu_state_history is None or self._rwmu_action_history is None:
            raise RuntimeError("RWM-U histories are not initialized")
        self._rwmu_action_history = torch.roll(self._rwmu_action_history, shifts=-1, dims=1)
        self._rwmu_action_history[:, -1] = action_norm
        self.rwmu_model.reset()
        pred_state, aleatoric, epistemic, extension, contact, termination = self.rwmu_model(
            self._rwmu_state_history, self._rwmu_action_history
        )
        self._rwmu_state_norm = pred_state.detach()
        self._rwmu_state_history = torch.roll(self._rwmu_state_history, shifts=-1, dims=1)
        self._rwmu_state_history[:, -1] = self._rwmu_state_norm
        self._sync_obs_from_rwmu_state()
        self._actor_state = torch.roll(self._actor_state, shifts=-self.action_dim, dims=1)
        copy_dim = min(self.action_dim, self.actor_obs_dim)
        self._actor_state[:, -copy_dim:] = env_actions[:, :copy_dim]
        self._critic_state[:, : self.actor_obs_dim] = self._actor_state
        if extension is not None and extension.shape[-1] > 0:
            rewards = extension[:, :1]
        else:
            rewards = -self.reward_scale * torch.mean(env_actions.square(), dim=1, keepdim=True)
        if termination is not None and termination.shape[-1] > 0:
            learned_dones = (torch.sigmoid(termination[:, 0]) > 0.5)
        else:
            learned_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return rewards, learned_dones, aleatoric, epistemic

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
        if self._rwmu_state_norm is not None:
            self._rwmu_state_norm.zero_()
        if self._rwmu_state_history is not None:
            self._rwmu_state_history.zero_()
        if self._rwmu_action_history is not None:
            self._rwmu_action_history.zero_()
        self._actor_state.zero_()
        self._critic_state.zero_()
        self._last_actions.zero_()
        self._sync_obs_from_rwmu_state()
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
        self._last_actions = env_actions.detach()
        if self.backend in {"rwmu", "rwm-u", "rwm_u"}:
            rewards, learned_dones, aleatoric, epistemic = self._rwmu_step(env_actions)
        else:
            self._actor_state = torch.roll(self._actor_state, shifts=-self.action_dim, dims=1)
            copy_dim = min(self.action_dim, self.actor_obs_dim)
            self._actor_state[:, -copy_dim:] = env_actions[:, :copy_dim]
            self._critic_state[:, : self.actor_obs_dim] = self._actor_state
            rewards = -self.reward_scale * torch.mean(env_actions.square(), dim=1, keepdim=True)
            learned_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            aleatoric = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            epistemic = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if self.num_critics == 1:
            rewards_out = rewards.squeeze(-1)
        else:
            rewards_out = rewards.repeat(1, self.num_critics)

        time_outs = self.episode_length_buf >= self.max_episode_length
        dones_bool = learned_dones | time_outs
        dones = dones_bool.to(dtype=torch.long)
        reset_ids = time_outs.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self.episode_length_buf[reset_ids] = 0
            self._actor_state[reset_ids] = 0.0
            self._critic_state[reset_ids] = 0.0
            self._last_actions[reset_ids] = 0.0
            if self._rwmu_state_norm is not None:
                self._rwmu_state_norm[reset_ids] = 0.0
            if self._rwmu_state_history is not None:
                self._rwmu_state_history[reset_ids] = 0.0
            if self._rwmu_action_history is not None:
                self._rwmu_action_history[reset_ids] = 0.0

        infos = {
            "episode": {
                "rwm_reward": rewards.mean().detach(),
                "rwm_action_rate": torch.mean(action_delta.square()).detach(),
                "rwm_aleatoric": aleatoric.mean().detach(),
                "rwm_epistemic": epistemic.mean().detach(),
            },
            "to_log": {
                "rwm/reward": rewards.mean().detach(),
                "rwm/action_rate": torch.mean(action_delta.square()).detach(),
                "rwm/aleatoric": aleatoric.mean().detach(),
                "rwm/epistemic": epistemic.mean().detach(),
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
            "rwmu_state_norm": self._rwmu_state_norm.detach().cpu() if self._rwmu_state_norm is not None else None,
        }

    def load_env_state_dict(self, state_dict: dict[str, Any] | None):
        if not state_dict:
            return
        episode_length_buf = state_dict.get("episode_length_buf")
        if episode_length_buf is not None:
            self.episode_length_buf[:] = episode_length_buf.to(self.device)
        rwmu_state_norm = state_dict.get("rwmu_state_norm")
        if rwmu_state_norm is not None and self._rwmu_state_norm is not None:
            self._rwmu_state_norm[:] = rwmu_state_norm.to(self.device)
            self._rwmu_state_history[:, -1] = self._rwmu_state_norm
            self._sync_obs_from_rwmu_state()

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
