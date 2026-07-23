#!/usr/bin/env python3
"""Validate a SONIC RWM-U rollout dataset exported by collect_sonic_rwmu_dataset.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


REQUIRED_TOP_LEVEL = ("format", "num_envs", "steps", "actions", "rewards", "dones", "time_outs", "rwm_u")
REQUIRED_RWMU = ("state", "next_state", "action", "extension", "contact", "termination", "schema")


def _shape(value: Any):
    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def _schema_term_dim(schema: dict[str, Any], name: str) -> int | None:
    for term in schema.get("state_terms", []):
        if term.get("name") == name:
            return int(term["dim"])
    return None


def _check_schema_dimensions(rwm: dict[str, Any]) -> None:
    schema = rwm["schema"]
    if "state_dim" in schema and schema["state_dim"] is not None:
        if int(schema["state_dim"]) != int(rwm["state"].shape[-1]):
            raise ValueError("schema.state_dim does not match rwm_u.state")
    if "next_state_dim" in schema and schema["next_state_dim"] is not None:
        if int(schema["next_state_dim"]) != int(rwm["next_state"].shape[-1]):
            raise ValueError("schema.next_state_dim does not match rwm_u.next_state")
    if "action_dim" in schema and schema["action_dim"] is not None:
        if int(schema["action_dim"]) != int(rwm["action"].shape[-1]):
            raise ValueError("schema.action_dim does not match rwm_u.action")
    if "contact_dim" in schema and schema["contact_dim"] is not None:
        if int(schema["contact_dim"]) != int(rwm["contact"].shape[-1]):
            raise ValueError("schema.contact_dim does not match rwm_u.contact")

    num_joints = schema.get("num_joints")
    if num_joints is not None:
        for name in ("joint_pos", "joint_vel"):
            dim = _schema_term_dim(schema, name)
            if dim is not None and dim != int(num_joints):
                raise ValueError(f"{name} dim={dim} does not match schema.num_joints={num_joints}")

    num_bodies = schema.get("num_robot_bodies")
    if num_bodies is not None:
        expected = {
            "body_pos_w": int(num_bodies) * 3,
            "body_quat_w": int(num_bodies) * 4,
            "body_lin_vel_w": int(num_bodies) * 3,
            "body_ang_vel_w": int(num_bodies) * 3,
        }
        for name, expected_dim in expected.items():
            dim = _schema_term_dim(schema, name)
            if dim is not None and dim != expected_dim:
                raise ValueError(f"{name} dim={dim} does not match {num_bodies} robot bodies")

    contact_names = schema.get("contact_body_names") or []
    if contact_names and len(contact_names) != int(rwm["contact"].shape[-1]):
        raise ValueError("schema.contact_body_names length does not match rwm_u.contact dim")


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
    for key in ("state", "next_state", "action", "extension", "contact", "termination"):
        tensor = rwm[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"rwm_u.{key} must be a tensor")
        if tensor.shape[0] != steps or tensor.shape[1] != num_envs:
            raise ValueError(f"rwm_u.{key} has shape {tuple(tensor.shape)}, expected ({steps}, {num_envs}, dim)")
        if tensor.dim() != 3:
            raise ValueError(f"rwm_u.{key} must be rank-3, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"rwm_u.{key} contains NaN/Inf")

    if rwm["state"].shape[-1] != rwm["next_state"].shape[-1]:
        raise ValueError("rwm_u.state and rwm_u.next_state dims must match")
    _check_schema_dimensions(rwm)
    action_source = payload.get("env_actions_to_sim", payload["actions"])
    if rwm["action"].shape[-1] != action_source.reshape(steps, num_envs, -1).shape[-1]:
        raise ValueError("rwm_u.action dim does not match env_actions_to_sim/raw actions")

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
        "next_state_shape": _shape(rwm["next_state"]),
        "action_shape": _shape(rwm["action"]),
        "extension_shape": _shape(rwm["extension"]),
        "contact_shape": _shape(rwm["contact"]),
        "termination_shape": _shape(rwm["termination"]),
        "schema_name": rwm["schema"].get("name"),
        "state_source": rwm["schema"].get("state_source"),
        "next_state_source": rwm["schema"].get("next_state_source"),
        "dimension_policy": rwm["schema"].get("dimension_policy"),
        "schema_state_dim": rwm["schema"].get("state_dim"),
        "schema_action_dim": rwm["schema"].get("action_dim"),
        "schema_contact_dim": rwm["schema"].get("contact_dim"),
        "num_joints": rwm["schema"].get("num_joints"),
        "num_robot_bodies": rwm["schema"].get("num_robot_bodies"),
        "num_contact_bodies": rwm["schema"].get("num_contact_bodies"),
        "joint_names": rwm["schema"].get("joint_names", []),
        "robot_body_names": rwm["schema"].get("robot_body_names", []),
        "contact_body_names": rwm["schema"].get("contact_body_names", []),
        "state_dim_formula": rwm["schema"].get("state_dim_formula"),
        "state_dim_formula_terms": rwm["schema"].get("state_dim_formula_terms", {}),
        "state_terms": state_terms,
        "next_state_terms": rwm["schema"].get("next_state_terms", []),
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
            f"state={summary['state_shape']} next_state={summary['next_state_shape']} action={summary['action_shape']} "
            f"extension={summary['extension_shape']} contact={summary['contact_shape']} "
            f"termination={summary['termination_shape']} source={summary['state_source']}"
        )


if __name__ == "__main__":
    main()
