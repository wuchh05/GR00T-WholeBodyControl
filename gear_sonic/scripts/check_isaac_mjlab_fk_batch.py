#!/usr/bin/env python3
"""Batch Isaac-vs-mjlab FK alignment checks for partial motion sets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--npz-root", type=Path, required=True)
    parser.add_argument("--npz-glob", default="**/*.npz")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--frames", default="0,middle,last")
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--position-warning-threshold", type=float, default=0.05)
    parser.add_argument("--orientation-warning-threshold", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args = _parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context

from gear_sonic.data_process.convert_bones_csv_to_mjlab_npz import _load_bones_csv, _resample_motion
from gear_sonic.envs.manager_env.robots.g1 import (
    G1_CYLINDER_MODEL_12_DEX_CFG,
    G1_MUJOCO_TO_ISAACLAB_DOF,
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

MJLAB_G1_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)


def _quat_abs_angle_error_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(1e-8)
    b = b / torch.linalg.norm(b, dim=-1, keepdim=True).clamp_min(1e-8)
    dot = torch.abs(torch.sum(a * b, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _resolve_csv_path(npz_path: Path) -> Path:
    rel = npz_path.relative_to(args.npz_root).with_suffix(".csv")
    csv_path = args.csv_root / rel
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV pair not found for {npz_path}: {csv_path}")
    return csv_path


def _select_frames(frame_count: int) -> list[int]:
    out = []
    for token in [item.strip() for item in args.frames.split(",") if item.strip()]:
        if token == "middle":
            frame = frame_count // 2
        elif token == "last":
            frame = frame_count - 1
        else:
            frame = int(token)
        if frame < 0:
            frame = frame_count + frame
        if frame < 0 or frame >= frame_count:
            raise ValueError(f"frame {token!r} resolved to {frame}, outside 0..{frame_count - 1}")
        if frame not in out:
            out.append(frame)
    return out


def _load_resampled(csv_path: Path):
    root_pos, root_rot, joint_pos_mj = _load_bones_csv(csv_path)
    return _resample_motion(
        root_pos,
        root_rot,
        joint_pos_mj,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        max_output_frames=None,
    )


def _write_robot_frame(robot, sim, device: str, root_pos, root_quat_wxyz, joint_pos_mj, frame: int):
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0:3] = torch.as_tensor(root_pos[frame], device=device).unsqueeze(0)
    root_state[:, 3:7] = torch.as_tensor(root_quat_wxyz[frame], device=device).unsqueeze(0)
    root_state[:, 7:10] = torch.zeros((1, 3), device=device)
    root_state[:, 10:13] = torch.zeros((1, 3), device=device)
    robot.write_root_state_to_sim(root_state)

    isaac_joint_pos = torch.as_tensor(
        joint_pos_mj[frame][G1_MUJOCO_TO_ISAACLAB_DOF], device=device
    ).unsqueeze(0)
    robot.write_joint_state_to_sim(isaac_joint_pos, torch.zeros_like(isaac_joint_pos))
    robot.reset()
    for _ in range(2):
        sim.step(render=False)
        robot.update(sim.cfg.dt)


def main() -> None:
    npz_paths = sorted(args.npz_root.glob(args.npz_glob))
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        raise FileNotFoundError(f"no NPZ files found in {args.npz_root} with {args.npz_glob}")

    device = getattr(args, "device", "cuda:0")
    rows = []
    with build_simulation_context(
        device=device,
        auto_add_lighting=False,
        gravity_enabled=False,
        add_ground_plane=False,
    ) as sim:
        sim._app_control_on_stop_handle = None
        sim_utils.create_prim("/World/Env_0", "Xform", translation=(0.0, 0.0, 0.0))
        robot = Articulation(G1_CYLINDER_MODEL_12_DEX_CFG.replace(prim_path="/World/Env_0/Robot"))
        sim.reset()
        assert robot.is_initialized

        for npz_path in npz_paths:
            csv_path = _resolve_csv_path(npz_path)
            root_pos, root_quat_wxyz, _root_lin_vel, _root_ang_vel, joint_pos_mj, _joint_vel = _load_resampled(csv_path)
            mjlab = np.load(npz_path)
            frame_count = min(joint_pos_mj.shape[0], mjlab["body_pos_w"].shape[0])
            for frame in _select_frames(frame_count):
                _write_robot_frame(robot, sim, device, root_pos, root_quat_wxyz, joint_pos_mj, frame)
                pos_errors = []
                ori_errors = []
                worst_body = None
                worst_pos = -1.0
                for body_name in TRACKING_BODY_NAMES:
                    isaac_idx = robot.body_names.index(body_name)
                    mjlab_idx = MJLAB_G1_BODY_NAMES.index(body_name)
                    isaac_pos = robot.data.body_link_pos_w[0, isaac_idx].detach()
                    isaac_quat = robot.data.body_link_quat_w[0, isaac_idx].detach()
                    mj_pos = torch.as_tensor(mjlab["body_pos_w"][frame, mjlab_idx], device=device)
                    mj_quat = torch.as_tensor(mjlab["body_quat_w"][frame, mjlab_idx], device=device)
                    pos_err = torch.linalg.norm(isaac_pos - mj_pos)
                    ori_err = _quat_abs_angle_error_wxyz(isaac_quat, mj_quat)
                    pos_f = float(pos_err.detach().cpu())
                    if pos_f > worst_pos:
                        worst_pos = pos_f
                        worst_body = body_name
                    pos_errors.append(pos_err)
                    ori_errors.append(ori_err)
                pos_t = torch.stack(pos_errors)
                ori_t = torch.stack(ori_errors)
                row = {
                    "csv": str(csv_path),
                    "npz": str(npz_path),
                    "frame": frame,
                    "max_position_error_m": float(pos_t.max().detach().cpu()),
                    "mean_position_error_m": float(pos_t.mean().detach().cpu()),
                    "max_orientation_error_rad": float(ori_t.max().detach().cpu()),
                    "mean_orientation_error_rad": float(ori_t.mean().detach().cpu()),
                    "worst_position_body": worst_body,
                }
                row["status"] = "pass"
                if row["max_position_error_m"] > args.position_warning_threshold:
                    row["status"] = "warn_position"
                if row["max_orientation_error_rad"] > args.orientation_warning_threshold:
                    row["status"] = "warn_orientation"
                rows.append(row)
                print(json.dumps(row), flush=True)

    summary = {
        "num_checks": len(rows),
        "num_pass": sum(row["status"] == "pass" for row in rows),
        "num_warn": sum(row["status"] != "pass" for row in rows),
        "max_position_error_m": max(row["max_position_error_m"] for row in rows),
        "mean_position_error_m": float(np.mean([row["mean_position_error_m"] for row in rows])),
        "max_orientation_error_rad": max(row["max_orientation_error_rad"] for row in rows),
        "mean_orientation_error_rad": float(np.mean([row["mean_orientation_error_rad"] for row in rows])),
        "rows": rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


try:
    main()
finally:
    simulation_app.close()
