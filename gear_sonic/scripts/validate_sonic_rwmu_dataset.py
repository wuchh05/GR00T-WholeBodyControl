#!/usr/bin/env python3
"""Validate a SONIC RWM-U rollout dataset exported by collect_sonic_rwmu_dataset.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


REQUIRED_TOP_LEVEL = ("format", "num_envs", "steps", "actions", "rewards", "dones", "time_outs", "rwm_u")
REQUIRED_RWMU = ("state", "action", "extension", "contact", "termination", "schema")


def _shape(value: Any):
    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def validate(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = [key for key in REQUIRED_TOP_LEVEL if key not in payload]
    if missing:
        raise KeyError(f"missing top-level keys: {missing}")
    rwm = payload["rwm_u"]
    missing = [key for key in REQUIRED_RWMU if key not in rwm]
    if missing:
        raise KeyError(f"missing rwm_u keys: {missing}")

    steps = int(payload["steps"])
    num_envs = int(payload["num_envs"])
    for key in ("state", "action", "extension", "contact", "termination"):
        tensor = rwm[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"rwm_u.{key} must be a tensor")
        if tensor.shape[0] != steps or tensor.shape[1] != num_envs:
            raise ValueError(f"rwm_u.{key} has shape {tuple(tensor.shape)}, expected ({steps}, {num_envs}, dim)")
        if tensor.dim() != 3:
            raise ValueError(f"rwm_u.{key} must be rank-3, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"rwm_u.{key} contains NaN/Inf")

    if rwm["termination"].shape[-1] != 1:
        raise ValueError("rwm_u.termination must have dim 1")
    if rwm["action"].shape[-1] != payload["actions"].reshape(steps, num_envs, -1).shape[-1]:
        raise ValueError("rwm_u.action dim does not match raw actions")

    state_terms = rwm["schema"].get("state_terms", [])
    missing_fallback_terms = [
        term for term in state_terms if str(term.get("source", "")).startswith("missing_zero_fallback:")
    ]
    if getattr(validate, "require_physical_state", False) and missing_fallback_terms:
        raise ValueError(f"dataset has missing physical state fallback terms: {missing_fallback_terms}")
    if getattr(validate, "require_contact", False) and rwm["contact"].shape[-1] == 0:
        raise ValueError("dataset has no contact labels")

    summary = {
        "path": str(path),
        "format": payload["format"],
        "sim_type": payload.get("sim_type"),
        "action_source": payload.get("action_source"),
        "steps": steps,
        "num_envs": num_envs,
        "state_shape": _shape(rwm["state"]),
        "action_shape": _shape(rwm["action"]),
        "extension_shape": _shape(rwm["extension"]),
        "contact_shape": _shape(rwm["contact"]),
        "termination_shape": _shape(rwm["termination"]),
        "state_source": rwm["schema"].get("state_source"),
        "state_terms": state_terms,
        "missing_fallback_terms": missing_fallback_terms,
        "contact_terms": rwm["schema"].get("contact_terms", []),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-physical-state", action="store_true")
    parser.add_argument("--require-contact", action="store_true")
    args = parser.parse_args()
    validate.require_physical_state = args.require_physical_state
    validate.require_contact = args.require_contact
    summary = validate(args.dataset)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"OK {summary['path']} | steps={summary['steps']} num_envs={summary['num_envs']} "
            f"state={summary['state_shape']} action={summary['action_shape']} "
            f"extension={summary['extension_shape']} contact={summary['contact_shape']} "
            f"termination={summary['termination_shape']} source={summary['state_source']}"
        )


if __name__ == "__main__":
    main()
