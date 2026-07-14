#!/usr/bin/env python3
"""Convert a Bones-SEED G1 CSV into mjlab's tracking NPZ format.

Bones-SEED G1 CSV files contain root translation/rotation and 29 joint DOFs.
This script replays the motion through mjlab/MuJoCo forward kinematics and
records full robot body positions, orientations, and velocities for mjlab's
tracking task.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
import torch
from tqdm import tqdm


BONES_CSV_JOINT_NAMES = [
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]

MJLAB_G1_JOINT_NAMES = [name.removesuffix("_dof") for name in BONES_CSV_JOINT_NAMES]


def _maybe_add_mjlab_source_path(source_path: str | None) -> None:
    if source_path is None:
        candidate = Path("/home/wuchenghui/mjlab/src")
        if candidate.exists():
            source_path = str(candidate)
    if source_path and source_path not in sys.path:
        sys.path.insert(0, source_path)


def _load_bones_csv(path: Path) -> tuple[np.ndarray, Rotation, np.ndarray]:
    data = pd.read_csv(path)
    missing = [
        col
        for col in (
            "root_translateX",
            "root_translateY",
            "root_translateZ",
            "root_rotateX",
            "root_rotateY",
            "root_rotateZ",
            *BONES_CSV_JOINT_NAMES,
        )
        if col not in data.columns
    ]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    root_pos = (
        data[["root_translateX", "root_translateY", "root_translateZ"]]
        .to_numpy(dtype=np.float32)
        / 100.0
    )
    root_rot = Rotation.from_euler(
        "xyz",
        data[["root_rotateX", "root_rotateY", "root_rotateZ"]].to_numpy(dtype=np.float64),
        degrees=True,
    )
    joint_pos = np.deg2rad(data[BONES_CSV_JOINT_NAMES].to_numpy(dtype=np.float32))
    if joint_pos.shape[1] != 29:
        raise ValueError(f"expected 29 joint DOFs, got {joint_pos.shape[1]}")
    return root_pos, root_rot, joint_pos.astype(np.float32)


def _resample_motion(
    root_pos: np.ndarray,
    root_rot: Rotation,
    joint_pos: np.ndarray,
    input_fps: float,
    output_fps: float,
    max_output_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_frames = root_pos.shape[0]
    if input_frames < 3:
        raise ValueError("at least 3 input frames are required")

    input_times = np.arange(input_frames, dtype=np.float64) / input_fps
    duration = input_times[-1]
    output_times = np.arange(0.0, duration, 1.0 / output_fps, dtype=np.float64)
    if max_output_frames is not None:
        output_times = output_times[:max_output_frames]
    if output_times.shape[0] < 3:
        raise ValueError("at least 3 output frames are required")

    root_pos_out = np.stack(
        [np.interp(output_times, input_times, root_pos[:, i]) for i in range(3)],
        axis=1,
    ).astype(np.float32)
    joint_pos_out = np.stack(
        [np.interp(output_times, input_times, joint_pos[:, i]) for i in range(joint_pos.shape[1])],
        axis=1,
    ).astype(np.float32)

    root_rot_out = Slerp(input_times, root_rot)(output_times)
    root_quat_xyzw = root_rot_out.as_quat().astype(np.float32)
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    dt = 1.0 / output_fps
    root_lin_vel = np.gradient(root_pos_out, dt, axis=0).astype(np.float32)
    joint_vel = np.gradient(joint_pos_out, dt, axis=0).astype(np.float32)
    root_ang_vel = _angular_velocity_wxyz(root_quat_wxyz, dt)
    return (
        root_pos_out,
        root_quat_wxyz,
        root_lin_vel,
        root_ang_vel,
        joint_pos_out,
        joint_vel,
    )


def _angular_velocity_wxyz(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rots = Rotation.from_quat(quat_xyzw)
    out = np.zeros((quat_wxyz.shape[0], 3), dtype=np.float32)
    rel = rots[:-2].inv() * rots[2:]
    out[1:-1] = (rel.as_rotvec() / (2.0 * dt)).astype(np.float32)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def convert_bones_csv_to_mjlab_npz(
    input_file: Path,
    output_file: Path,
    input_fps: float,
    output_fps: float,
    device: str,
    max_output_frames: int | None,
    mjlab_source_path: str | None,
) -> None:
    _maybe_add_mjlab_source_path(mjlab_source_path)

    from mjlab.entity import Entity
    from mjlab.scene import Scene
    from mjlab.sim.sim import Simulation, SimulationCfg
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA is not available; falling back to CPU.")
        device = "cpu"

    root_pos, root_rot, joint_pos = _load_bones_csv(input_file)
    (
        root_pos,
        root_quat_wxyz,
        root_lin_vel,
        root_ang_vel,
        joint_pos,
        joint_vel,
    ) = _resample_motion(
        root_pos,
        root_rot,
        joint_pos,
        input_fps=input_fps,
        output_fps=output_fps,
        max_output_frames=max_output_frames,
    )

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps
    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    scene.reset()

    robot: Entity = scene["robot"]
    joint_indexes = robot.find_joints(MJLAB_G1_JOINT_NAMES, preserve_order=True)[0]
    if len(joint_indexes) != 29:
        raise ValueError(f"expected 29 mjlab G1 joints, found {len(joint_indexes)}")

    log = {
        "fps": np.array(output_fps, dtype=np.float32),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for frame in tqdm(range(joint_pos.shape[0]), desc="Converting Bones CSV", unit="frame"):
        root_state = robot.data.default_root_state.clone()
        root_state[:, 0:3] = torch.as_tensor(root_pos[frame], device=device).unsqueeze(0)
        root_state[:, 3:7] = torch.as_tensor(root_quat_wxyz[frame], device=device).unsqueeze(0)
        root_state[:, 7:10] = torch.as_tensor(root_lin_vel[frame], device=device).unsqueeze(0)
        root_state[:, 10:13] = torch.as_tensor(root_ang_vel[frame], device=device).unsqueeze(0)
        robot.write_root_state_to_sim(root_state)

        full_joint_pos = robot.data.default_joint_pos.clone()
        full_joint_vel = robot.data.default_joint_vel.clone()
        full_joint_pos[:, joint_indexes] = torch.as_tensor(joint_pos[frame], device=device)
        full_joint_vel[:, joint_indexes] = torch.as_tensor(joint_vel[frame], device=device)
        robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        log["joint_pos"].append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].detach().cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_link_pos_w[0].detach().cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_link_quat_w[0].detach().cpu().numpy().copy())
        log["body_lin_vel_w"].append(
            robot.data.body_link_lin_vel_w[0].detach().cpu().numpy().copy()
        )
        log["body_ang_vel_w"].append(
            robot.data.body_link_ang_vel_w[0].detach().cpu().numpy().copy()
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.stack(value, axis=0) for key, value in log.items() if key != "fps"}
    np.savez(output_file, fps=log["fps"], **arrays)
    print(
        f"wrote {output_file} frames={arrays['joint_pos'].shape[0]} "
        f"joints={arrays['joint_pos'].shape[1]} bodies={arrays['body_pos_w'].shape[1]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-output-frames", type=int, default=None)
    parser.add_argument("--mjlab-source-path", default=None)
    args = parser.parse_args()
    convert_bones_csv_to_mjlab_npz(
        input_file=args.input,
        output_file=args.output,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        device=args.device,
        max_output_frames=args.max_output_frames,
        mjlab_source_path=args.mjlab_source_path,
    )


if __name__ == "__main__":
    main()
