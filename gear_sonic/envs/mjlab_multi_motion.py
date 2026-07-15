"""Multi-motion command support for mjlab tracking environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg


class MultiMotionLoader:
    """Load multiple mjlab tracking NPZ motions into one indexed tensor bank."""

    REQUIRED_KEYS = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )

    def __init__(
        self,
        motion_files: Sequence[str],
        body_indexes: torch.Tensor,
        device: str = "cpu",
    ) -> None:
        if not motion_files:
            raise ValueError("MultiMotionLoader requires at least one motion file.")

        self.motion_files = tuple(str(Path(path)) for path in motion_files)
        self._body_indexes = body_indexes
        arrays: dict[str, list[torch.Tensor]] = {key: [] for key in self.REQUIRED_KEYS}
        starts: list[int] = []
        lengths: list[int] = []
        cursor = 0
        ref_shapes: dict[str, tuple[int, ...]] = {}

        for motion_file in self.motion_files:
            data = np.load(motion_file)
            missing = [key for key in self.REQUIRED_KEYS if key not in data]
            if missing:
                raise KeyError(f"{motion_file} is missing required keys: {missing}")

            num_frames = int(data["joint_pos"].shape[0])
            if num_frames < 2:
                raise ValueError(f"{motion_file} must have at least 2 frames, got {num_frames}")

            for key in self.REQUIRED_KEYS:
                value = data[key]
                shape_tail = tuple(value.shape[1:])
                if key in ref_shapes and ref_shapes[key] != shape_tail:
                    raise ValueError(
                        f"{motion_file} key {key} has shape tail {shape_tail}, expected {ref_shapes[key]}"
                    )
                ref_shapes.setdefault(key, shape_tail)
                arrays[key].append(torch.tensor(value, dtype=torch.float32, device=device))

            starts.append(cursor)
            lengths.append(num_frames)
            cursor += num_frames

        self.motion_starts = torch.tensor(starts, dtype=torch.long, device=device)
        self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.motion_ends = self.motion_starts + self.motion_lengths
        self.num_motions = len(self.motion_files)
        self.time_step_total = int(cursor)

        self.joint_pos = torch.cat(arrays["joint_pos"], dim=0)
        self.joint_vel = torch.cat(arrays["joint_vel"], dim=0)
        self._body_pos_w = torch.cat(arrays["body_pos_w"], dim=0)
        self._body_quat_w = torch.cat(arrays["body_quat_w"], dim=0)
        self._body_lin_vel_w = torch.cat(arrays["body_lin_vel_w"], dim=0)
        self._body_ang_vel_w = torch.cat(arrays["body_ang_vel_w"], dim=0)
        self.body_pos_w = self._body_pos_w[:, self._body_indexes]
        self.body_quat_w = self._body_quat_w[:, self._body_indexes]
        self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
        self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]

    def sample(self, num_samples: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        motion_ids = torch.randint(0, self.num_motions, (num_samples,), device=device)
        starts = self.motion_starts[motion_ids]
        lengths = self.motion_lengths[motion_ids]
        local_steps = (torch.rand(num_samples, device=device) * lengths.float()).long()
        return starts + local_steps, starts + lengths

    def start_frames(self, num_samples: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        motion_ids = torch.randint(0, self.num_motions, (num_samples,), device=device)
        starts = self.motion_starts[motion_ids]
        lengths = self.motion_lengths[motion_ids]
        return starts, starts + lengths


class MultiMotionCommand(MotionCommand):
    cfg: "MultiMotionCommandCfg"

    def __init__(self, cfg: "MultiMotionCommandCfg", env):
        super().__init__(cfg, env)
        self.motion = MultiMotionLoader(cfg.motion_files, self.body_indexes, device=self.device)
        self.motion_end_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)

    def _uniform_sampling(self, env_ids: torch.Tensor):
        sampled_steps, end_steps = self.motion.sample(len(env_ids), self.device)
        self.time_steps[env_ids] = sampled_steps
        self.motion_end_steps[env_ids] = end_steps
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / max(self.motion.num_motions, 1)
        self.metrics["sampling_top1_bin"][:] = 0.5

    def _resample_command(self, env_ids: torch.Tensor):
        if self.cfg.sampling_mode == "start":
            sampled_steps, end_steps = self.motion.start_frames(len(env_ids), self.device)
            self.time_steps[env_ids] = sampled_steps
            self.motion_end_steps[env_ids] = end_steps
            self.metrics["sampling_entropy"][:] = 1.0
            self.metrics["sampling_top1_prob"][:] = 1.0 / max(self.motion.num_motions, 1)
            self.metrics["sampling_top1_bin"][:] = 0.5
        else:
            # Multi-motion adaptive sampling needs per-motion failure histograms. Use
            # uniform sampling for now instead of mixing bins across motion boundaries.
            self._uniform_sampling(env_ids)

        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori = quat_mul(orientations_delta, root_ori)

        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids]
        joint_pos += sample_uniform(
            lower=self.cfg.joint_position_range[0],
            upper=self.cfg.joint_position_range[1],
            size=joint_pos.shape,
            device=joint_pos.device,
        )

        self._write_reference_state_to_sim(
            env_ids,
            root_pos,
            root_ori,
            root_lin_vel,
            root_ang_vel,
            joint_pos,
            joint_vel,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion_end_steps)[0]
        if env_ids.numel() > 0:
            self._resample_command(env_ids)
            self._env.sim.forward()
        self.update_relative_body_poses()
        self._current_bin_failed.zero_()


@dataclass(kw_only=True)
class MultiMotionCommandCfg(MotionCommandCfg):
    motion_files: tuple[str, ...] = field(default_factory=tuple)

    def build(self, env) -> MultiMotionCommand:
        return MultiMotionCommand(self, env)
