#!/usr/bin/env python3
"""Validate that RWM SONIC config preserves the original release architecture."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _compose(overrides: list[str]):
    config_dir = str(_REPO_ROOT / "gear_sonic" / "config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        return compose(config_name="base", overrides=overrides)


def _select(cfg, path: str):
    cur = cfg
    for part in path.split("."):
        cur = cur[part]
    if OmegaConf.is_config(cur):
        return OmegaConf.to_container(cur, resolve=True)
    return cur


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--checkpoint", default="sonic_release/last.pt")
    args = parser.parse_args()

    common = [f"+checkpoint={args.checkpoint}", f"num_envs={args.num_envs}", "headless=True"]
    original = _compose(["+exp=manager/universal_token/all_modes/sonic_release", *common])
    rwm = _compose(["+exp=rwm/sonic_release", *common])

    equal_paths = [
        "trainer._target_",
        "algo.config.actor",
        "algo.config.critic",
        "algo.config.num_steps_per_env",
        "algo.config.use_clampped_std",
        "algo.config.std_clamp_min",
        "algo.config.std_clamp_max",
        "algo.config.max_grad_norm",
        "actor_prop_history_length",
        "actor_actions_history_length",
        "critic_prop_history_length",
        "critic_actions_history_length",
        "manager_env.observations",
        "manager_env.rewards",
        "manager_env.terminations",
        "manager_env.commands.motion.num_future_frames",
        "manager_env.commands.motion.smpl_num_future_frames",
    ]
    mismatches = []
    for path in equal_paths:
        left = _select(original, path)
        right = _select(rwm, path)
        if left != right:
            mismatches.append(path)

    expected_diffs = {
        "sim_type": (str(original.sim_type), str(rwm.sim_type)),
        "use_manager_env": (bool(original.use_manager_env), bool(rwm.use_manager_env)),
    }
    ok_expected = expected_diffs == {
        "sim_type": ("isaacsim", "rwm"),
        "use_manager_env": (True, False),
    }

    if mismatches or not ok_expected:
        print("RWM config parity failed")
        if mismatches:
            print("mismatched paths:")
            for path in mismatches:
                print(f"  - {path}")
        print(f"expected diffs: {expected_diffs}")
        raise SystemExit(1)

    print("RWM config parity OK")
    print("identical checked paths:")
    for path in equal_paths:
        print(f"  - {path}")
    print("intentional differences:")
    for key, value in expected_diffs.items():
        print(f"  - {key}: {value[0]} -> {value[1]}")


if __name__ == "__main__":
    main()
