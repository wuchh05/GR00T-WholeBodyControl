#!/usr/bin/env python3
"""Summarize SONIC training comparison logs.

This parser is intentionally small: it extracts trainer log directories, learning
iterations, and mean reward values from Rich/ANSI console logs produced by
train_agent_trl.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ITER_RE = re.compile(r"Learning iteration\s+(\d+)")
REWARD_RE = re.compile(r"Mean rewards:\s+([-+0-9.eE]+)")
LOG_DIR_RE = re.compile(r"logs_rl/[^\s│╰]+")


def clean(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_log(path: Path) -> dict[str, Any]:
    text = clean(path.read_text(errors="replace"))
    rows: list[dict[str, float | int]] = []
    current_iter: int | None = None
    log_dirs: list[str] = []

    for line in text.splitlines():
        iter_match = ITER_RE.search(line)
        if iter_match:
            current_iter = int(iter_match.group(1))

        reward_match = REWARD_RE.search(line)
        if reward_match and current_iter is not None:
            rows.append({"iteration": current_iter, "mean_reward": float(reward_match.group(1))})

        for log_dir in LOG_DIR_RE.findall(line):
            if log_dir not in log_dirs:
                log_dirs.append(log_dir)

    if not rows:
        raise ValueError(f"No learning iterations with mean rewards found in {path}")

    rewards = [float(row["mean_reward"]) for row in rows]
    return {
        "log": str(path),
        "log_dirs": log_dirs,
        "iterations": rows,
        "first_reward": rewards[0],
        "last_reward": rewards[-1],
        "delta_reward": rewards[-1] - rewards[0],
        "max_reward": max(rewards),
        "num_iterations": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-log", type=Path, required=True)
    parser.add_argument("--mjlab-log", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = {
        "isaac": parse_log(args.isaac_log),
        "mjlab": parse_log(args.mjlab_log),
    }
    result["comparison"] = {
        "last_reward_delta_mjlab_minus_isaac": result["mjlab"]["last_reward"] - result["isaac"]["last_reward"],
        "both_positive_trend": result["isaac"]["delta_reward"] > 0 and result["mjlab"]["delta_reward"] > 0,
    }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
