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


def _make_config(config: dict[str, Any]):
    if OmegaConf is not None:
        return OmegaConf.create(config)
    return _AttrDict(config)


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

        config = dict(config or {})
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

        self._validate_minimal_tracking_contract()
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

    def _process_raw_obs(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        actor_obs = obs.get("policy", obs.get("actor"))
        critic_obs = obs.get("critic")
        if actor_obs is None or critic_obs is None:
            raise KeyError(
                f"Expected mjlab observation groups 'actor'/'policy' and 'critic', got: {list(obs.keys())}"
            )

        processed = {
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
        }
        for key, value in obs.items():
            if key not in {"actor", "policy", "critic"}:
                processed[key] = value
        self.obs_buf_dict = processed
        return processed

    def reset_all(self, global_rank: int = 0):  # noqa: ARG002
        return self.reset()

    def reset(self, flatten_dict_obs: bool = True):  # noqa: ARG002
        obs, extras = self.unwrapped.reset()
        self.extras = extras
        return self._process_raw_obs(obs)

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
        return self._process_raw_obs(obs), rewards, dones, infos

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

