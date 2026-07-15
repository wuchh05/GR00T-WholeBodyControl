#!/usr/bin/env python3
"""Check first-stage SONIC/mjlab G1 data and ordering assumptions.

This is a low-cost correctness check. It intentionally avoids importing Isaac
Lab so it can run in the lightweight ``sonic`` environment. It verifies:

* Bones-SEED CSV columns match the expected 29-DOF G1 MuJoCo order.
* SONIC IsaacLab-order DOFs map to the same MuJoCo order used by the converter.
* mjlab can resolve all 29 G1 joints and the 14 tracking bodies.
* Optional mjlab motion NPZ files have expected keys, shapes, and finite values.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from gear_sonic.data_process.convert_bones_csv_to_mjlab_npz import (
    BONES_CSV_JOINT_NAMES,
    MJLAB_G1_JOINT_NAMES,
    _maybe_add_mjlab_source_path,
)
from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER


REQUIRED_NPZ_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
TRACKING_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
ANCHOR_BODY_NAME = "torso_link"


def _load_literal_from_source(source_file: Path, name: str):
    tree = ast.parse(source_file.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {source_file}")


def _check_converter_order(repo_root: Path) -> dict:
    g1_py = repo_root / "gear_sonic/envs/manager_env/robots/g1.py"
    isaac_to_mujoco = _load_literal_from_source(g1_py, "G1_ISAACLAB_TO_MUJOCO_DOF")
    mapped_names = [G1_ISAACLab_ORDER[i] for i in isaac_to_mujoco]
    if mapped_names != MJLAB_G1_JOINT_NAMES:
        raise AssertionError(
            "IsaacLab->MuJoCo DOF mapping does not match converter MJLAB_G1_JOINT_NAMES\n"
            f"mapped={mapped_names}\nconverter={MJLAB_G1_JOINT_NAMES}"
        )
    if [name.removesuffix("_dof") for name in BONES_CSV_JOINT_NAMES] != MJLAB_G1_JOINT_NAMES:
        raise AssertionError("Bones CSV joint columns do not match mjlab G1 joint names")
    return {
        "num_dof": len(MJLAB_G1_JOINT_NAMES),
        "isaaclab_to_mujoco_dof": isaac_to_mujoco,
        "mujoco_joint_names": MJLAB_G1_JOINT_NAMES,
    }


def _check_bones_csv(path: Path | None) -> dict | None:
    if path is None:
        return None
    data = pd.read_csv(path, nrows=3)
    required = [
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *BONES_CSV_JOINT_NAMES,
    ]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise AssertionError(f"{path} is missing required columns: {missing}")
    return {
        "path": str(path),
        "checked_rows": int(len(data)),
        "num_required_columns": len(required),
    }


def _check_motion_npz(path: Path | None) -> dict | None:
    if path is None:
        return None
    data = np.load(path)
    missing = [key for key in REQUIRED_NPZ_KEYS if key not in data.files]
    if missing:
        raise AssertionError(f"{path} is missing keys: {missing}")
    shapes = {key: tuple(int(v) for v in data[key].shape) for key in REQUIRED_NPZ_KEYS}
    if shapes["joint_pos"][-1] != 29:
        raise AssertionError(f"joint_pos must have 29 DOF, got {shapes['joint_pos']}")
    frame_count = shapes["joint_pos"][0]
    for key in REQUIRED_NPZ_KEYS:
        value = data[key]
        if key != "fps" and value.shape[0] != frame_count:
            raise AssertionError(f"{key} frame count {value.shape[0]} != {frame_count}")
        if not np.isfinite(value).all():
            raise AssertionError(f"{key} contains non-finite values")
    if shapes["body_pos_w"][1] < len(TRACKING_BODY_NAMES):
        raise AssertionError(
            f"body_pos_w must contain at least {len(TRACKING_BODY_NAMES)} bodies, got {shapes['body_pos_w']}"
        )
    return {"path": str(path), "shapes": shapes}


def _check_mjlab_model(mjlab_source_path: str | None, device: str) -> dict:
    _maybe_add_mjlab_source_path(mjlab_source_path)

    from mjlab.entity import Entity
    from mjlab.scene import Scene
    from mjlab.sim.sim import Simulation, SimulationCfg
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    cfg = unitree_g1_flat_tracking_env_cfg()
    motion_cmd = cfg.commands["motion"]
    if tuple(motion_cmd.body_names) != TRACKING_BODY_NAMES:
        raise AssertionError(
            "mjlab tracking body_names changed; update checker and converter assumptions\n"
            f"cfg={motion_cmd.body_names}\nexpected={TRACKING_BODY_NAMES}"
        )
    if motion_cmd.anchor_body_name != ANCHOR_BODY_NAME:
        raise AssertionError(
            f"anchor body changed: {motion_cmd.anchor_body_name} != {ANCHOR_BODY_NAME}"
        )

    sim_cfg = SimulationCfg()
    scene = Scene(cfg.scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    scene.reset()

    robot: Entity = scene["robot"]
    joint_indexes, joint_names = robot.find_joints(MJLAB_G1_JOINT_NAMES, preserve_order=True)
    if joint_names != MJLAB_G1_JOINT_NAMES:
        raise AssertionError(f"mjlab joint lookup mismatch: {joint_names}")
    if len(joint_indexes) != 29 or len(set(joint_indexes)) != 29:
        raise AssertionError(f"expected 29 unique joint indexes, got {joint_indexes}")

    body_indexes, body_names = robot.find_bodies(TRACKING_BODY_NAMES, preserve_order=True)
    if tuple(body_names) != TRACKING_BODY_NAMES:
        raise AssertionError(f"mjlab body lookup mismatch: {body_names}")
    if len(body_indexes) != len(TRACKING_BODY_NAMES) or len(set(body_indexes)) != len(body_indexes):
        raise AssertionError(f"tracking body indexes are invalid: {body_indexes}")

    return {
        "device": device,
        "num_robot_joints": len(robot.joint_names),
        "num_robot_bodies": len(robot.body_names),
        "joint_indexes": joint_indexes,
        "tracking_body_indexes": body_indexes,
        "tracking_body_names": body_names,
        "anchor_body_name": motion_cmd.anchor_body_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bones-csv", type=Path, default=None)
    parser.add_argument("--motion-npz", type=Path, default=None)
    parser.add_argument("--mjlab-source-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    report = {
        "converter_order": _check_converter_order(repo_root),
        "bones_csv": _check_bones_csv(args.bones_csv),
        "motion_npz": _check_motion_npz(args.motion_npz),
        "mjlab_model": _check_mjlab_model(args.mjlab_source_path, args.device),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("GEAR-SONIC mjlab alignment check passed")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
