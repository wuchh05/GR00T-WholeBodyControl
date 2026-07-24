#!/usr/bin/env python3
"""Train a SONIC-specific RWM-U dynamics model from exported rollout datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_RWM_RSL_ROOT = _REPO_ROOT / "external_dependencies" / "rsl_rl_rwm"
if str(_RWM_RSL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RWM_RSL_ROOT))


def _architecture_config(hidden_size: int) -> dict[str, Any]:
    return {
        "type": "rnn",
        "rnn_type": "gru",
        "rnn_num_layers": 2,
        "rnn_hidden_size": hidden_size,
        "state_mean_shape": [hidden_size // 2],
        "state_logstd_shape": [hidden_size // 2],
        "extension_shape": [hidden_size // 2],
        "contact_shape": [hidden_size // 2],
        "termination_shape": [hidden_size // 2],
    }


def _load_concat(paths: list[Path], device: str):
    states = []
    next_states = []
    actions = []
    extensions = []
    contacts = []
    terminations = []
    schemas = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rwm = payload["rwm_u"]
        states.append(rwm["state"].float())
        next_states.append(rwm.get("next_state", rwm["state"]).float())
        actions.append(rwm["action"].float())
        extensions.append(rwm["extension"].float())
        contacts.append(rwm["contact"].float())
        terminations.append(rwm["termination"].float())
        schemas.append(rwm.get("schema", {}))
    state_dim = states[0].shape[-1]
    next_state_dim = next_states[0].shape[-1]
    action_dim = actions[0].shape[-1]
    extension_dim = extensions[0].shape[-1]
    contact_dim = contacts[0].shape[-1]
    termination_dim = terminations[0].shape[-1]
    for tensor_list, name, dim in (
        (states, "state", state_dim),
        (next_states, "next_state", next_state_dim),
        (actions, "action", action_dim),
        (extensions, "extension", extension_dim),
        (contacts, "contact", contact_dim),
        (terminations, "termination", termination_dim),
    ):
        bad = [tuple(t.shape) for t in tensor_list if t.shape[-1] != dim]
        if bad:
            raise ValueError(f"all datasets must share {name}_dim={dim}, got incompatible shapes {bad}")
    return (
        torch.cat(states, dim=1).to(device),
        torch.cat(next_states, dim=1).to(device),
        torch.cat(actions, dim=1).to(device),
        torch.cat(extensions, dim=1).to(device),
        torch.cat(contacts, dim=1).to(device),
        torch.cat(terminations, dim=1).to(device),
        schemas,
    )


def _make_physx_series(state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor):
    if state.shape != next_state.shape:
        raise ValueError(f"state shape {tuple(state.shape)} must match next_state shape {tuple(next_state.shape)}")
    # Convert explicit transition rows (s_t, a_t, s_t+1) into the sequential
    # state/action layout expected by upstream RWM-U. The final action is a pad
    # value and is never used as an input for a real logged transition.
    state_series = torch.cat([state, next_state[-1:].clone()], dim=0)
    action_pad = torch.zeros_like(action[:1])
    action_series = torch.cat([action_pad, action], dim=0)
    return state_series, action_series


def _make_windows(state, action, extension, contact, termination, history: int, forecast: int):
    # input tensors are (T, N, D); output windows are (B, history + forecast, D)
    total = history + forecast
    steps, num_envs, _ = state.shape
    if steps < total:
        raise ValueError(f"need at least history+forecast={total} steps, got {steps}")
    state = state.transpose(0, 1)
    action = action.transpose(0, 1)
    extension = extension.transpose(0, 1)
    contact = contact.transpose(0, 1)
    termination = termination.transpose(0, 1)
    state_w = []
    action_w = []
    extension_w = []
    contact_w = []
    termination_w = []
    for env_id in range(num_envs):
        for start in range(steps - total + 1):
            # Do not train windows that cross true terminations.
            if torch.any(termination[env_id, start : start + total - 1] > 0.5):
                continue
            state_w.append(state[env_id, start : start + total])
            action_w.append(action[env_id, start : start + total])
            extension_w.append(extension[env_id, start : start + total])
            contact_w.append(contact[env_id, start : start + total])
            termination_w.append(termination[env_id, start : start + total])
    if not state_w:
        raise ValueError("no valid non-terminated windows found")
    return (
        torch.stack(state_w, dim=0),
        torch.stack(action_w, dim=0),
        torch.stack(extension_w, dim=0),
        torch.stack(contact_w, dim=0),
        torch.stack(termination_w, dim=0),
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    from rsl_rl.modules import SystemDynamicsEnsemble

    state, next_state, action, extension, contact, termination, schemas = _load_concat(args.dataset, args.device)
    state, action = _make_physx_series(state, next_state, action)
    extension = torch.cat([extension, torch.zeros_like(extension[-1:])], dim=0)
    contact = torch.cat([contact, torch.zeros_like(contact[-1:])], dim=0)
    termination = torch.cat([termination, torch.zeros_like(termination[-1:])], dim=0)
    state_mean = state.flatten(0, 1).mean(dim=0)
    state_std = state.flatten(0, 1).std(dim=0).clamp_min(1.0e-6)
    action_mean = action.flatten(0, 1).mean(dim=0)
    action_std = action.flatten(0, 1).std(dim=0).clamp_min(1.0e-6)
    state_n = (state - state_mean) / state_std
    action_n = (action - action_mean) / action_std

    state_w, action_w, extension_w, contact_w, termination_w = _make_windows(
        state_n, action_n, extension, contact, termination, args.history_horizon, args.forecast_horizon
    )
    dataset = TensorDataset(state_w, action_w, extension_w, contact_w, termination_w)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SystemDynamicsEnsemble(
        state_dim=state.shape[-1],
        action_dim=action.shape[-1],
        extension_dim=extension.shape[-1],
        contact_dim=contact.shape[-1],
        termination_dim=termination.shape[-1],
        device=args.device,
        ensemble_size=args.ensemble_size,
        history_horizon=args.history_horizon,
        architecture_config=_architecture_config(args.hidden_size),
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics = []
    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for batch in loader:
            s_b, a_b, e_b, c_b, t_b = [item.to(args.device) for item in batch]
            model.reset()
            optimizer.zero_grad()
            state_loss, sequence_loss, bound_loss, kl_loss, extension_loss, contact_loss, termination_loss = model.compute_loss(
                s_b,
                a_b,
                e_b if extension.shape[-1] > 0 else None,
                c_b if contact.shape[-1] > 0 else None,
                t_b if termination.shape[-1] > 0 else None,
            )
            loss = state_loss + sequence_loss + 0.01 * bound_loss + kl_loss + extension_loss + contact_loss + termination_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total += float(loss.detach().cpu())
            count += 1
        metrics.append({"epoch": epoch + 1, "loss": total / max(count, 1)})
        print(f"epoch={epoch + 1} loss={metrics[-1]['loss']:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "sonic-rwmu-physx-dynamics-v1",
        "iter": args.epochs,
        "system_dynamics_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "state_data_mean": state_mean.detach().cpu(),
        "state_data_std": state_std.detach().cpu(),
        "action_data_mean": action_mean.detach().cpu(),
        "action_data_std": action_std.detach().cpu(),
        "dims": {
            "state_dim": state.shape[-1],
            "action_dim": action.shape[-1],
            "extension_dim": extension.shape[-1],
            "contact_dim": contact.shape[-1],
            "termination_dim": termination.shape[-1],
        },
        "history_horizon": args.history_horizon,
        "forecast_horizon": args.forecast_horizon,
        "ensemble_size": args.ensemble_size,
        "architecture_config": _architecture_config(args.hidden_size),
        "schemas": schemas,
        "metrics": metrics,
    }
    torch.save(checkpoint, args.output)
    summary = {k: v for k, v in checkpoint.items() if k not in {"system_dynamics_state_dict", "optimizer_state_dict"}}
    args.report.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"saved checkpoint: {args.output}")
    print(f"saved report: {args.report}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("/tmp/sonic_rwmu_train_report.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history-horizon", type=int, default=8)
    parser.add_argument("--forecast-horizon", type=int, default=2)
    parser.add_argument("--ensemble-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
