#!/usr/bin/env python3
"""Run a minimal RWM-U offline training smoke test with bundled assets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RWM_ROOT = _REPO_ROOT / "external_dependencies" / "robotic_world_model"
_RWM_MODEL_BASED = _RWM_ROOT / "scripts" / "reinforcement_learning" / "model_based"

for path in (_REPO_ROOT, _RWM_MODEL_BASED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _assert_assets() -> None:
    data_path = _RWM_ROOT / "assets" / "data" / "state_action_data_0.csv"
    model_path = _RWM_ROOT / "assets" / "models" / "pretrain_rnn_ens.pt"
    missing = [str(path) for path in (data_path, model_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "RWM-U bundled smoke assets are missing:\n  " + "\n  ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="Torch device for the smoke run.")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps-per-env", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--max-episode-length", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/rwmu_offline_smoke"),
        help="Directory for the smoke policy checkpoints.",
    )
    args = parser.parse_args()

    _assert_assets()
    os.environ.setdefault("WANDB_MODE", "disabled")

    import wandb
    from configs import AnymalDFlatConfig
    import train as rwmu_train

    config = AnymalDFlatConfig()
    config.experiment_config.device = args.device
    config.environment_config.num_envs = args.num_envs
    config.environment_config.max_episode_length = args.max_episode_length
    config.data_config.dataset_root = str(_RWM_ROOT / "assets")
    config.data_config.dataset_folder = "data"
    config.data_config.file_data_size = 10000
    config.data_config.batch_data_size = 10000
    config.model_architecture_config.resume_path = str(
        _RWM_ROOT / "assets" / "models" / "pretrain_rnn_ens.pt"
    )
    config.policy_training_config.num_steps_per_env = args.num_steps_per_env
    config.policy_training_config.max_iterations = args.max_iterations
    config.policy_training_config.save_interval = 1
    config.policy_training_config.export_dir = str(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with wandb.init(project=config.experiment_name, mode="disabled"):
        model_experiment = rwmu_train.ModelBasedExperiment(**config.experiment_config.to_dict())
        model_experiment.prepare_environment(**config.environment_config.to_dict())
        model_experiment.prepare_model(**config.model_architecture_config.to_dict())
        model_experiment.prepare_policy(**config.policy_architecture_config.to_dict())
        model_experiment.prepare_algorithm(**config.policy_algorithm_config.to_dict())
        model_experiment.prepare_data(**config.data_config.to_dict())
        model_experiment.train_policy(str(args.output_dir), **config.policy_training_config.to_dict())

    checkpoints = sorted(args.output_dir.rglob("policy_*.pt"))
    if not checkpoints:
        raise RuntimeError(f"RWM-U smoke finished but no policy checkpoint was found in {args.output_dir}")
    print(f"RWM-U offline smoke OK: {checkpoints[-1]}")


if __name__ == "__main__":
    main()
