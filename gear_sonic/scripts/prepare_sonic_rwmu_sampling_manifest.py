#!/usr/bin/env python3
"""Prepare a managed SONIC motion subset manifest for RWM-U rollout collection.

The script chooses a small deterministic subset of robot motion PKLs by category,
materializes symlinks or copies under an output directory, and writes a JSON
manifest recording selected and unselected candidates. The subset directory can
be passed directly as SONIC's ``motion_lib_cfg.motion_file``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
from typing import Any


DEFAULT_CATEGORIES = {
    "nominal_walk": ["walk_forward_loop", "walk_forward_start", "walk_forward_stop"],
    "turning": ["turn_start_walk", "step_rotate_idle"],
    "jog": ["jog_forward", "jog_sideway", "jog_backward"],
    "jump": ["jump_forward", "jump_left", "jump_right", "turn_jump"],
    "balance_idle": ["idle_one_foot", "idle_loop", "looking_around"],
    "hard_lower_body": ["kneeling", "walk_backward", "walk_sideway"],
}


def _find_pkls(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.pkl") if path.name != "metadata.pkl")


def _match_category(files: list[Path], keywords: list[str]) -> list[Path]:
    out = []
    for path in files:
        name = path.stem.lower()
        if any(keyword.lower() in name for keyword in keywords):
            out.append(path)
    return sorted(out, key=lambda p: p.as_posix())


def _materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _choose_smpl(smpl_by_stem: dict[str, Path], robot_path: Path) -> Path | None:
    return smpl_by_stem.get(robot_path.stem)


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    robot_root = args.robot_motion_root.resolve()
    smpl_root = args.smpl_motion_root.resolve() if args.smpl_motion_root is not None else None
    out_root = args.output_root.resolve()
    robot_subset = out_root / "robot_subset"
    smpl_subset = out_root / "smpl_subset"
    robot_files = _find_pkls(robot_root)
    smpl_by_stem = {path.stem: path for path in _find_pkls(smpl_root)} if smpl_root else {}

    selected: list[dict[str, Any]] = []
    unselected: dict[str, list[str]] = {}
    seen_stems: set[str] = set()
    for category, keywords in DEFAULT_CATEGORIES.items():
        candidates = [path for path in _match_category(robot_files, keywords) if path.stem not in seen_stems]
        chosen = candidates[: args.max_per_category]
        unselected[category] = [str(path.relative_to(robot_root)) for path in candidates[args.max_per_category :]]
        for path in chosen:
            seen_stems.add(path.stem)
            rel = path.relative_to(robot_root)
            dst_robot = robot_subset / rel
            _materialize(path, dst_robot, args.materialize)
            smpl_path = _choose_smpl(smpl_by_stem, path)
            dst_smpl = None
            if smpl_path is not None:
                dst_smpl = smpl_subset / smpl_path.name
                _materialize(smpl_path, dst_smpl, args.materialize)
            selected.append(
                {
                    "category": category,
                    "motion_key": path.stem,
                    "robot_source": str(path),
                    "robot_subset": str(dst_robot),
                    "smpl_source": str(smpl_path) if smpl_path is not None else None,
                    "smpl_subset": str(dst_smpl) if dst_smpl is not None else None,
                    "split": "val" if len(selected) % max(args.val_every, 1) == 0 else "train",
                }
            )

    manifest = {
        "format": "sonic-rwmu-sampling-manifest-v1",
        "name": args.name,
        "robot_motion_root": str(robot_root),
        "smpl_motion_root": str(smpl_root) if smpl_root else None,
        "output_root": str(out_root),
        "robot_subset_dir": str(robot_subset),
        "smpl_subset_dir": str(smpl_subset) if smpl_root else None,
        "materialize": args.materialize,
        "categories": DEFAULT_CATEGORIES,
        "max_per_category": args.max_per_category,
        "selected": selected,
        "unselected": unselected,
        "counts": {
            "available_robot_files": len(robot_files),
            "available_smpl_files": len(smpl_by_stem),
            "selected": len(selected),
            "selected_with_smpl": sum(1 for item in selected if item["smpl_source"] is not None),
            "train": sum(1 for item in selected if item["split"] == "train"),
            "val": sum(1 for item in selected if item["split"] == "val"),
        },
        "notes": [
            "This first manifest is intentionally small; expand categories or max_per_category for later versions.",
            "Selected/unselected candidates are recorded so future manifests can avoid silent sampling drift.",
        ],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-motion-root", type=Path, default=Path("data/motion_lib_bones_seed/robot_filtered"))
    parser.add_argument("--smpl-motion-root", type=Path, default=Path("data/smpl_filtered"))
    parser.add_argument("--output-root", type=Path, default=Path("data/rwmu_sampling/v1_small"))
    parser.add_argument("--output", type=Path, default=Path("data/rwmu_sampling/v1_small/manifest.json"))
    parser.add_argument("--name", default="v1_small")
    parser.add_argument("--max-per-category", type=int, default=2)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--materialize", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()
    manifest = build_manifest(args)
    print(
        f"wrote {args.output} | selected={manifest['counts']['selected']} "
        f"train={manifest['counts']['train']} val={manifest['counts']['val']} "
        f"robot_subset={manifest['robot_subset_dir']}"
    )


if __name__ == "__main__":
    main()
