#!/usr/bin/env python3
"""Compare Isaac Lab FK against a mjlab-generated motion NPZ for one Bones frame.

The mjlab side is represented by the NPZ produced by
``convert_bones_csv_to_mjlab_npz.py``.  For the same source CSV frame, this
script writes root pose + 29 DOF into the Isaac Lab G1 articulation, updates FK,
and compares selected body poses by body name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bones-csv", type=Path, required=True)
    parser.add_argument("--mjlab-npz", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--position-warning-threshold", type=float, default=0.05)
    parser.add_argument("--orientation-warning-threshold", type=float, default=0.25)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
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


def _quat_abs_angle_error_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(1e-8)
    b = b / torch.linalg.norm(b, dim=-1, keepdim=True).clamp_min(1e-8)
    dot = torch.abs(torch.sum(a * b, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


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


def main() -> None:
    root_pos, root_rot, joint_pos_mj = _load_bones_csv(args.bones_csv)
    root_pos, root_quat_wxyz, root_lin_vel, root_ang_vel, joint_pos_mj, joint_vel_mj = _resample_motion(
        root_pos,
        root_rot,
        joint_pos_mj,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        max_output_frames=None,
    )
    if args.frame < 0 or args.frame >= joint_pos_mj.shape[0]:
        raise ValueError(f"frame {args.frame} outside resampled motion length {joint_pos_mj.shape[0]}")

    mjlab = np.load(args.mjlab_npz)
    if args.frame >= mjlab["body_pos_w"].shape[0]:
        raise ValueError(f"frame {args.frame} outside mjlab NPZ length {mjlab['body_pos_w'].shape[0]}")

    device = getattr(args, "device", "cuda:0")
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

        root_state = robot.data.default_root_state.clone()
        root_state[:, 0:3] = torch.as_tensor(root_pos[args.frame], device=device).unsqueeze(0)
        root_state[:, 3:7] = torch.as_tensor(root_quat_wxyz[args.frame], device=device).unsqueeze(0)
        root_state[:, 7:10] = torch.zeros((1, 3), device=device)
        root_state[:, 10:13] = torch.zeros((1, 3), device=device)
        robot.write_root_state_to_sim(root_state)

        isaac_joint_pos = torch.as_tensor(
            joint_pos_mj[args.frame][G1_MUJOCO_TO_ISAACLAB_DOF], device=device
        ).unsqueeze(0)
        isaac_joint_vel = torch.zeros_like(isaac_joint_pos)
        robot.write_joint_state_to_sim(isaac_joint_pos, isaac_joint_vel)
        robot.reset()
        for _ in range(2):
            sim.step(render=False)
            robot.update(sim.cfg.dt)

        results = {}
        pos_errors = []
        ori_errors = []
        for body_name in TRACKING_BODY_NAMES:
            isaac_idx = robot.body_names.index(body_name)
            mujoco_idx = MJLAB_G1_BODY_NAMES.index(body_name)
            isaac_pos = robot.data.body_link_pos_w[0, isaac_idx].detach()
            isaac_quat = robot.data.body_link_quat_w[0, isaac_idx].detach()
            mj_pos = torch.as_tensor(mjlab["body_pos_w"][args.frame, mujoco_idx], device=device)
            mj_quat = torch.as_tensor(mjlab["body_quat_w"][args.frame, mujoco_idx], device=device)
            pos_err = torch.linalg.norm(isaac_pos - mj_pos)
            ori_err = _quat_abs_angle_error_wxyz(isaac_quat, mj_quat)
            pos_errors.append(pos_err)
            ori_errors.append(ori_err)
            results[body_name] = {
                "isaac_body_index": int(isaac_idx),
                "mjlab_body_index": int(mujoco_idx),
                "position_error_m": float(pos_err.detach().cpu()),
                "orientation_error_rad": float(ori_err.detach().cpu()),
            }

        pos_errors_t = torch.stack(pos_errors)
        ori_errors_t = torch.stack(ori_errors)
        summary = {
            "bones_csv": str(args.bones_csv),
            "mjlab_npz": str(args.mjlab_npz),
            "frame": args.frame,
            "num_bodies": len(TRACKING_BODY_NAMES),
            "max_position_error_m": float(pos_errors_t.max().detach().cpu()),
            "mean_position_error_m": float(pos_errors_t.mean().detach().cpu()),
            "max_orientation_error_rad": float(ori_errors_t.max().detach().cpu()),
            "mean_orientation_error_rad": float(ori_errors_t.mean().detach().cpu()),
            "position_warning_threshold": args.position_warning_threshold,
            "orientation_warning_threshold": args.orientation_warning_threshold,
            "status": "pass",
            "bodies": results,
        }
        if summary["max_position_error_m"] > args.position_warning_threshold:
            summary["status"] = "warn_position"
        if summary["max_orientation_error_rad"] > args.orientation_warning_threshold:
            summary["status"] = "warn_orientation"

        payload = json.dumps(summary, indent=2)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(payload + "\n")
        print(payload, flush=True)


try:
    main()
finally:
    simulation_app.close()
