#!/usr/bin/env python3
"""Run a managed SONIC RWM-U rollout sampling plan from a manifest.

This script is intended for the compute machine. It does not train RWM-U; it
creates rollout datasets with clear bookkeeping so later training knows exactly
which motion subset, policies, action sources, and validation checks were used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import time
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_COLLECT = _REPO_ROOT / "gear_sonic" / "scripts" / "collect_sonic_rwmu_dataset.py"
_VALIDATE = _REPO_ROOT / "gear_sonic" / "scripts" / "validate_sonic_rwmu_dataset.py"


def _load_policy_list(paths: list[str], list_file: Path | None) -> list[Path]:
    out = [Path(item) for item in paths]
    if list_file is not None:
        for line in list_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(Path(line))
    return out


def _split_overrides(value: str | None) -> list[str]:
    return shlex.split(value) if value else []


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=10)


def _run(
    cmd: list[str],
    dry_run: bool,
    *,
    timeout_seconds: float = 0.0,
    success_marker: str | None = None,
    success_grace_seconds: float = 10.0,
) -> dict[str, Any]:
    start = time.time()
    if dry_run:
        return {"returncode": 0, "seconds": 0.0, "stdout_tail": "", "stderr_tail": "", "dry_run": True}

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("SONIC_RWMU_DEBUG", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=_REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
        env=env,
    )
    tail_lines: list[str] = []
    marker_seen_at: float | None = None
    timed_out = False
    marker_terminated = False

    assert proc.stdout is not None
    while True:
        if proc.poll() is not None:
            for line in proc.stdout.readlines():
                print(line, end="", flush=True)
                tail_lines.append(line)
            break

        ready, _, _ = select.select([proc.stdout], [], [], 0.2)
        if ready:
            line = proc.stdout.readline()
            if line:
                print(line, end="", flush=True)
                tail_lines.append(line)
                if len(tail_lines) > 200:
                    tail_lines = tail_lines[-200:]
                if success_marker and success_marker in line:
                    marker_seen_at = time.time()

        elapsed = time.time() - start
        if timeout_seconds > 0 and elapsed > timeout_seconds:
            timed_out = True
            print(f"[timeout] command exceeded {timeout_seconds:.1f}s; terminating", flush=True)
            _terminate_process(proc)
            break
        if marker_seen_at is not None and time.time() - marker_seen_at > success_grace_seconds:
            marker_terminated = True
            print(
                f"[marker] saw {success_marker!r}; terminating child after "
                f"{success_grace_seconds:.1f}s close grace",
                flush=True,
            )
            _terminate_process(proc)
            break

    return {
        "returncode": proc.returncode if proc.returncode is not None else -signal.SIGTERM,
        "seconds": round(time.time() - start, 3),
        "stdout_tail": "".join(tail_lines)[-8000:],
        "stderr_tail": "",
        "dry_run": False,
        "timed_out": timed_out,
        "success_marker_seen": marker_seen_at is not None,
        "terminated_after_success_marker": marker_terminated,
    }


def _base_overrides(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    if args.sim_preset == "rwm-smoke":
        overrides = ["+exp=rwm/sonic_release", f"num_envs={args.num_envs}", "headless=True", "use_wandb=false"]
    else:
        smpl_dir = manifest.get("smpl_subset_dir") or "dummy"
        overrides = [
            "+exp=manager/universal_token/all_modes/sonic_release",
            f"num_envs={args.num_envs}",
            "headless=True",
            "use_wandb=false",
            f"++manager_env.commands.motion.motion_lib_cfg.motion_file={manifest['robot_subset_dir']}",
            f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={smpl_dir}",
            "++manager_env.commands.motion.motion_lib_cfg.load_unique_motions=true",
        ]
    overrides.extend(_split_overrides(args.extra_overrides))
    return overrides


def _task_specs(args: argparse.Namespace, policies: list[Path]) -> list[dict[str, Any]]:
    tasks = []
    for source in args.action_source:
        if source in {"random", "zeros", "sine"}:
            tasks.append({
                "action_source": source,
                "deterministic_policy_action": False,
                "policy": None,
                "tag": source,
            })
        elif source in {"policy", "policy_mean", "policy_stochastic"}:
            if not policies:
                raise ValueError(f"{source} requires --policy or --policy-list")
            deterministic = source == "policy_mean"
            normalized_source = "policy"
            for policy in policies:
                tag = policy.parent.name if policy.name == "last.pt" else policy.stem
                tasks.append(
                    {
                        "action_source": normalized_source,
                        "deterministic_policy_action": deterministic,
                        "policy": str(policy),
                        "tag": f"{tag}_{source}",
                    }
                )
        else:
            raise ValueError(f"unknown action source: {source}")
    return tasks


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policies = _load_policy_list(args.policy, args.policy_list)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_overrides = _base_overrides(args, manifest)
    tasks = _task_specs(args, policies)
    ledger: dict[str, Any] = {
        "format": "sonic-rwmu-sampling-ledger-v1",
        "manifest": str(args.manifest.resolve()),
        "sim_preset": args.sim_preset,
        "output_dir": str(output_dir),
        "steps": args.steps,
        "num_envs": args.num_envs,
        "device": args.device,
        "base_overrides": base_overrides,
        "tasks": [],
    }

    for idx, task in enumerate(tasks):
        out = output_dir / f"{idx:03d}_{task['tag']}.pt"
        cmd = [
            sys.executable,
            str(_COLLECT),
            "--output",
            str(out),
            "--steps",
            str(args.steps),
            "--device",
            args.device,
            "--action-source",
            task["action_source"],
        ]
        if task["policy"] is not None:
            cmd.extend(["--checkpoint", task["policy"]])
        if task.get("deterministic_policy_action"):
            cmd.append("--deterministic-policy-action")
        cmd.append("--")
        cmd.extend(base_overrides)
        entry: dict[str, Any] = {"index": idx, "output": str(out), "task": task, "collect_cmd": cmd, "status": "running"}
        ledger["tasks"].append(entry)
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        args.ledger.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        print(f"[running] {out}", flush=True)
        collect_result = _run(
            cmd,
            args.dry_run,
            timeout_seconds=args.collect_timeout_seconds,
            success_marker="[sonic-rwmu-export] save done" if args.fast_exit_after_save else None,
            success_grace_seconds=args.fast_exit_grace_seconds,
        )
        entry["collect"] = collect_result
        collect_ok = collect_result["returncode"] == 0 or (collect_result.get("success_marker_seen") and out.exists())
        if collect_ok and not args.dry_run:
            val_cmd = [sys.executable, str(_VALIDATE), str(out), "--json"]
            if args.require_physical_state:
                val_cmd.append("--require-physical-state")
            if args.require_contact:
                val_cmd.append("--require-contact")
            entry["validate_cmd"] = val_cmd
            entry["validate"] = _run(val_cmd, False, timeout_seconds=args.validate_timeout_seconds)
            entry["status"] = "ok" if entry["validate"]["returncode"] == 0 else "validate_failed"
        else:
            entry["status"] = "dry_run" if args.dry_run else "collect_failed"
        args.ledger.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        print(f"[{entry['status']}] {out}")

    ledger["summary"] = {
        "total": len(ledger["tasks"]),
        "ok": sum(1 for item in ledger["tasks"] if item["status"] == "ok"),
        "failed": sum(1 for item in ledger["tasks"] if item["status"].endswith("failed")),
        "dry_run": sum(1 for item in ledger["tasks"] if item["status"] == "dry_run"),
    }
    args.ledger.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--policy-list", type=Path, default=None)
    parser.add_argument(
        "--action-source",
        action="append",
        choices=["policy", "policy_mean", "policy_stochastic", "random", "zeros", "sine"],
        default=[],
        help="Repeat to include multiple data sources. Defaults to policy and random. policy_mean uses deterministic action_mean; policy_stochastic is a compatibility alias for policy.",
    )
    parser.add_argument("--sim-preset", choices=["isaac", "rwm-smoke"], default="isaac")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extra-overrides", default=None)
    parser.add_argument("--require-physical-state", action="store_true")
    parser.add_argument("--require-contact", action="store_true")
    parser.add_argument("--collect-timeout-seconds", type=float, default=0.0, help="Per collect task timeout. 0 disables timeout.")
    parser.add_argument("--validate-timeout-seconds", type=float, default=300.0, help="Per validation task timeout.")
    parser.add_argument("--fast-exit-after-save", action=argparse.BooleanOptionalAction, default=True, help="If the collector prints save done but Isaac shutdown hangs, terminate the child after a short grace period and validate the saved dataset.")
    parser.add_argument("--fast-exit-grace-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.action_source:
        args.action_source = ["policy", "random"]
    ledger = run_plan(args)
    print(json.dumps(ledger.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
