#!/usr/bin/env python3
"""Run a short zero/random-action rollout sanity check for SONIC/mjlab."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-npz", type=Path, required=True)
    parser.add_argument("--mjlab-source-path", default="/home/wuchenghui/mjlab/src")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--random-action-scale", type=float, default=0.2)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from gear_sonic.envs.mjlab_env import create_mjlab_env

    cfg = OmegaConf.create(
        {
            "num_envs": args.num_envs,
            "headless": True,
            "experiment_dir": "logs_rl/TRL_G1_MjLab/partial_rollout_sanity",
            "mjlab_env": {
                "source_path": args.mjlab_source_path,
                "motion_file": str(args.motion_npz.resolve()),
                "episode_length_s": 2.0,
                "auto_reset": True,
            },
        }
    )

    env = create_mjlab_env(cfg, device=args.device)
    obs = env.reset_all()
    print(
        "reset",
        "actor_obs",
        tuple(obs["actor_obs"].shape),
        "critic_obs",
        tuple(obs["critic_obs"].shape),
    )
    assert torch.isfinite(obs["actor_obs"]).all(), "actor_obs has non-finite values after reset"
    assert torch.isfinite(obs["critic_obs"]).all(), "critic_obs has non-finite values after reset"

    action_dim = env.action_space.shape[-1]
    for mode in ("zero", "random"):
        total_reward = 0.0
        done_sum = 0
        for step in range(args.steps):
            if mode == "zero":
                actions = torch.zeros(env.num_envs, action_dim, device=args.device)
            else:
                actions = args.random_action_scale * torch.randn(
                    env.num_envs, action_dim, device=args.device
                )
            obs, rewards, dones, _infos = env.step({"actions": actions})
            assert torch.isfinite(obs["actor_obs"]).all(), f"{mode} step {step}: actor_obs"
            assert torch.isfinite(obs["critic_obs"]).all(), f"{mode} step {step}: critic_obs"
            assert torch.isfinite(rewards).all(), f"{mode} step {step}: rewards"
            total_reward += float(rewards.mean().detach().cpu())
            done_sum += int(dones.sum().detach().cpu())
        print(mode, "mean_reward_sum", round(total_reward, 6), "done_sum", done_sum)

    print("short rollout sanity passed")


if __name__ == "__main__":
    main()
