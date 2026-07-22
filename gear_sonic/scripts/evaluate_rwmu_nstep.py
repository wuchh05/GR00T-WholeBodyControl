#!/usr/bin/env python3
"""Evaluate n-step open-loop error for the bundled upstream RWM-U model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
RWM_ROOT = REPO_ROOT / "external_dependencies" / "robotic_world_model"
RWM_MODEL_BASED = RWM_ROOT / "scripts" / "reinforcement_learning" / "model_based"
if str(RWM_MODEL_BASED) not in sys.path:
    sys.path.insert(0, str(RWM_MODEL_BASED))


STATE_GROUPS = {
    "base_lin_vel": (0, 3),
    "base_ang_vel": (3, 6),
    "projected_gravity": (6, 9),
    "joint_pos": (9, 21),
    "joint_vel": (21, 33),
    "joint_torque": (33, 45),
}


def _binary_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.bool()
    target = target.bool()
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    denom = 2.0 * tp + fp + fn
    if denom.item() == 0:
        return 1.0
    return (2.0 * tp / denom).item()


def _load_data(path: Path, state_dim: int, action_dim: int, contact_dim: int, termination_dim: int, device: str):
    data = pd.read_csv(path, header=None)
    expected_dim = state_dim + action_dim + contact_dim + termination_dim
    if data.shape[1] != expected_dim:
        raise ValueError(f"{path} has {data.shape[1]} columns, expected {expected_dim}")
    tensor = torch.tensor(data.values, dtype=torch.float32, device=device)
    state = tensor[:, :state_dim]
    action = tensor[:, state_dim : state_dim + action_dim]
    contact = tensor[:, state_dim + action_dim : state_dim + action_dim + contact_dim]
    termination = tensor[:, state_dim + action_dim + contact_dim :]
    return state, action, contact, termination


def _make_windows(
    state: torch.Tensor,
    action: torch.Tensor,
    contact: torch.Tensor,
    termination: torch.Tensor,
    history: int,
    horizon: int,
    batch_size: int,
    seed: int,
):
    valid_starts = state.shape[0] - history - horizon + 1
    if valid_starts <= 0:
        raise ValueError(f"Need at least {history + horizon} rows, got {state.shape[0]}")

    # Avoid windows crossing termination flags, matching upstream dataset filtering.
    term_flat = termination.flatten()
    starts = []
    for i in range(valid_starts):
        if not torch.any(term_flat[i : i + history + horizon - 1] > 0.5):
            starts.append(i)
    if not starts:
        raise ValueError("No valid non-terminated windows were found.")

    generator = torch.Generator(device=state.device).manual_seed(seed)
    start_tensor = torch.tensor(starts, device=state.device)
    if len(starts) > batch_size:
        ids = torch.randperm(len(starts), generator=generator, device=state.device)[:batch_size]
        start_tensor = start_tensor[ids]

    offsets = torch.arange(history + horizon, device=state.device)
    indices = start_tensor[:, None] + offsets[None, :]
    return state[indices], action[indices], contact[indices], termination[indices]


def _to_float_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            out[key] = _to_float_dict(value)
        elif isinstance(value, torch.Tensor):
            out[key] = value.item()
        else:
            out[key] = value
    return out


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from configs import AnymalDFlatConfig
    from rsl_rl.modules import SystemDynamicsEnsemble

    cfg = AnymalDFlatConfig()
    state_dim = 45
    action_dim = 12
    contact_dim = cfg.model_architecture_config.contact_dim
    termination_dim = cfg.model_architecture_config.termination_dim
    history = cfg.model_architecture_config.history_horizon
    max_horizon = args.max_horizon

    state, action, contact, termination = _load_data(
        args.data,
        state_dim,
        action_dim,
        contact_dim,
        termination_dim,
        args.device,
    )
    state_win, action_win, contact_win, termination_win = _make_windows(
        state,
        action,
        contact,
        termination,
        history,
        max_horizon,
        args.batch_size,
        args.seed,
    )

    state_mean = torch.tensor(cfg.data_config.state_data_mean, dtype=torch.float32, device=args.device)
    state_std = torch.tensor(cfg.data_config.state_data_std, dtype=torch.float32, device=args.device)
    action_mean = torch.tensor(cfg.data_config.action_data_mean, dtype=torch.float32, device=args.device)
    action_std = torch.tensor(cfg.data_config.action_data_std, dtype=torch.float32, device=args.device)
    state_norm = (state_win - state_mean) / state_std
    action_norm = (action_win - action_mean) / action_std

    model = SystemDynamicsEnsemble(
        state_dim=state_dim,
        action_dim=action_dim,
        extension_dim=0,
        contact_dim=contact_dim,
        termination_dim=termination_dim,
        device=args.device,
        ensemble_size=cfg.model_architecture_config.ensemble_size,
        history_horizon=history,
        architecture_config=cfg.model_architecture_config.architecture_config,
    ).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["system_dynamics_state_dict"])
    model.eval()
    model.reset()

    x_state = state_norm[:, :history]
    predictions = []
    contacts = []
    terminations = []
    aleatoric = []
    epistemic = []
    horizons = []

    for step in range(max_horizon):
        if step == 0:
            x_action = action_norm[:, 1 : history + 1]
        else:
            x_action = action_norm[:, history + step : history + step + 1]
        pred_state, ale, epi, _, pred_contact, pred_term = model(x_state, x_action)
        predictions.append(pred_state)
        contacts.append(pred_contact)
        terminations.append(pred_term)
        aleatoric.append(ale)
        epistemic.append(epi)
        horizons.append(step + 1)
        x_state = pred_state.unsqueeze(1)

    pred_norm = torch.stack(predictions, dim=1)
    pred_state = pred_norm * state_std + state_mean
    target_state = state_win[:, history : history + max_horizon]
    pred_contact = torch.stack(contacts, dim=1)
    target_contact = contact_win[:, history : history + max_horizon]
    pred_termination = torch.stack(terminations, dim=1)
    target_termination = termination_win[:, history : history + max_horizon]

    per_horizon: dict[str, Any] = {}
    for idx, horizon in enumerate(horizons):
        err = pred_state[:, idx] - target_state[:, idx]
        group_rmse = {
            name: torch.sqrt(torch.mean(err[:, start:end] ** 2)).item()
            for name, (start, end) in STATE_GROUPS.items()
        }
        contact_pred_binary = torch.sigmoid(pred_contact[:, idx]) > 0.5
        contact_target_binary = target_contact[:, idx] > 0.5
        term_pred_binary = torch.sigmoid(pred_termination[:, idx]) > 0.5
        term_target_binary = target_termination[:, idx] > 0.5
        per_horizon[str(horizon)] = {
            "state_rmse": torch.sqrt(torch.mean(err**2)).item(),
            "state_group_rmse": group_rmse,
            "contact_accuracy": (contact_pred_binary == contact_target_binary).float().mean().item(),
            "contact_f1": _binary_f1(contact_pred_binary, contact_target_binary),
            "termination_accuracy": (term_pred_binary == term_target_binary).float().mean().item(),
            "termination_f1": _binary_f1(term_pred_binary, term_target_binary),
            "aleatoric_uncertainty_mean": aleatoric[idx].mean().item(),
            "epistemic_uncertainty_mean": epistemic[idx].mean().item(),
        }

    return _to_float_dict(
        {
            "scope": "upstream RWM-U ANYmal-D bundled dataset/checkpoint",
            "data": str(args.data),
            "checkpoint": str(args.checkpoint),
            "checkpoint_iter": ckpt.get("iter"),
            "device": args.device,
            "num_windows": int(state_win.shape[0]),
            "history_horizon": history,
            "max_horizon": max_horizon,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "contact_dim": contact_dim,
            "termination_dim": termination_dim,
            "metrics": per_horizon,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data",
        type=Path,
        default=RWM_ROOT / "assets" / "data" / "state_action_data_0.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=RWM_ROOT / "assets" / "models" / "pretrain_rnn_ens.pt",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/rwmu_nstep_error.json"))
    args = parser.parse_args()

    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
