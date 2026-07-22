#!/usr/bin/env python3
"""Validate SONIC release fields against the RWM-U dataset group contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from omegaconf import OmegaConf


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


REQUIRED_GROUPS = {
    "system_state": [
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_b",
        "root_ang_vel_b",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "last_action",
        "motion_id",
        "motion_time_step",
        "motion_start_time_step",
    ],
    "system_action": ["joint_pos"],
    "system_extension": [
        "tracking_anchor_pos",
        "tracking_anchor_ori",
        "tracking_relative_body_pos",
        "tracking_relative_body_ori",
        "tracking_body_linvel",
        "tracking_body_angvel",
        "action_rate_l2",
        "joint_limit",
        "undesired_contacts",
        "anti_shake_ang_vel",
        "tracking_vr_5point_local",
        "feet_acc",
        "reward_total",
    ],
    "system_contact": ["body_contact", "undesired_contacts"],
    "system_termination": [
        "anchor_pos",
        "anchor_ori_full",
        "ee_body_pos",
        "foot_pos_xyz",
    ],
}


def _load_schema(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return OmegaConf.load(path)


def _schema_names(schema, section: str) -> set[str]:
    if section == "action":
        return {term.name for term in schema.action.terms}
    if section == "reward":
        names = {term.name for term in schema.reward.terms}
        names.add("reward_total")
        return names
    if section == "termination":
        return {term.name for term in schema.termination.terms if term.get("learned", True)}
    raise KeyError(section)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema",
        type=Path,
        default=_REPO_ROOT / "gear_sonic" / "config" / "rwm" / "sonic_schema.yaml",
    )
    args = parser.parse_args()

    schema = _load_schema(args.schema)

    action_names = _schema_names(schema, "action")
    reward_names = _schema_names(schema, "reward")
    termination_names = _schema_names(schema, "termination")
    tracked_bodies = list(schema.tracked_bodies.body_names)

    failures = []
    if not REQUIRED_GROUPS["system_action"][0] in action_names:
        failures.append("system_action.joint_pos is missing from schema.action")

    missing_rewards = sorted(set(REQUIRED_GROUPS["system_extension"]) - reward_names)
    if missing_rewards:
        failures.append(f"system_extension missing reward labels: {missing_rewards}")

    missing_terms = sorted(set(REQUIRED_GROUPS["system_termination"]) - termination_names)
    if missing_terms:
        failures.append(f"system_termination missing labels: {missing_terms}")

    if len(tracked_bodies) != 14:
        failures.append(f"expected 14 tracked bodies, got {len(tracked_bodies)}")

    if failures:
        print("SONIC RWM-U field validation failed")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("SONIC RWM-U field validation OK")
    print("required groups:")
    for group, names in REQUIRED_GROUPS.items():
        print(f"  - {group}: {len(names)} fields")
    print(f"tracked_bodies: {len(tracked_bodies)}")


if __name__ == "__main__":
    main()
