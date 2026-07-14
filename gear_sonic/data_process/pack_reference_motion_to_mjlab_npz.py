"""Pack GEAR deploy reference CSVs into the mjlab tracking NPZ format.

This is a smoke-test utility. It expects a directory containing:
  joint_pos.csv, joint_vel.csv, body_pos.csv, body_quat.csv,
  body_lin_vel.csv, body_ang_vel.csv
as found under ``gear_sonic_deploy/reference/example``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"{path} should be 2-D, got shape {data.shape}")
    return data


def _reshape(data: np.ndarray, width: int, name: str) -> np.ndarray:
    if data.shape[1] % width != 0:
        raise ValueError(f"{name} has {data.shape[1]} columns, not divisible by {width}")
    return data.reshape(data.shape[0], data.shape[1] // width, width)


def _pad_bodies(data: np.ndarray, target_body_count: int) -> np.ndarray:
    body_count = data.shape[1]
    if target_body_count < body_count:
        raise ValueError(
            f"target body count {target_body_count} is smaller than source {body_count}"
        )
    if target_body_count == body_count:
        return data
    padded = np.repeat(data[:, 0:1], target_body_count, axis=1)
    padded[:, :body_count] = data
    return padded


def pack_reference_motion(
    input_dir: Path, output_file: Path, fps: int, body_count: int | None
) -> None:
    joint_pos = _load_csv(input_dir / "joint_pos.csv")
    joint_vel = _load_csv(input_dir / "joint_vel.csv")
    body_pos_w = _reshape(_load_csv(input_dir / "body_pos.csv"), 3, "body_pos")
    body_quat_w = _reshape(_load_csv(input_dir / "body_quat.csv"), 4, "body_quat")
    body_lin_vel_w = _reshape(_load_csv(input_dir / "body_lin_vel.csv"), 3, "body_lin_vel")
    body_ang_vel_w = _reshape(_load_csv(input_dir / "body_ang_vel.csv"), 3, "body_ang_vel")

    frame_count = joint_pos.shape[0]
    arrays = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }
    for name, value in arrays.items():
        if value.shape[0] != frame_count:
            raise ValueError(
                f"{name} has {value.shape[0]} frames, expected {frame_count}"
            )

    if joint_pos.shape[1] != 29:
        raise ValueError(f"expected 29 joints, got {joint_pos.shape[1]}")
    source_body_count = body_pos_w.shape[1]
    if source_body_count != 14:
        raise ValueError(f"expected 14 bodies, got {body_pos_w.shape[1]}")

    if body_count is not None:
        body_pos_w = _pad_bodies(body_pos_w, body_count)
        body_quat_w = _pad_bodies(body_quat_w, body_count)
        body_lin_vel_w = _pad_bodies(body_lin_vel_w, body_count)
        body_ang_vel_w = _pad_bodies(body_ang_vel_w, body_count)
        arrays.update(
            {
                "body_pos_w": body_pos_w,
                "body_quat_w": body_quat_w,
                "body_lin_vel_w": body_lin_vel_w,
                "body_ang_vel_w": body_ang_vel_w,
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, fps=np.array(fps, dtype=np.int32), **arrays)
    print(
        f"wrote {output_file} with frames={frame_count}, "
        f"joints={joint_pos.shape[1]}, bodies={body_pos_w.shape[1]} "
        f"(source_bodies={source_body_count})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--body-count",
        type=int,
        default=None,
        help=(
            "Pad body arrays to this count by repeating body_0. This is only for "
            "mjlab smoke tests when starting from 14-body deploy references."
        ),
    )
    args = parser.parse_args()
    pack_reference_motion(args.input_dir, args.output, args.fps, args.body_count)


if __name__ == "__main__":
    main()
