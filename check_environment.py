#!/usr/bin/env python3
"""Pre-flight environment check for GR00T-WholeBodyControl.

Run this before training or deployment to verify all prerequisites are met.

Usage:
    python check_environment.py              # Check everything
    python check_environment.py --training   # Training checks only
    python check_environment.py --mjlab      # MuJoCo/mjlab training checks
    python check_environment.py --npu        # Ascend torch-npu checks
    python check_environment.py --deploy     # Deployment checks only
"""

import importlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys


def check(name, passed, msg_pass="", msg_fail=""):
    status = "PASS" if passed else "FAIL"
    symbol = "[+]" if passed else "[X]"
    detail = msg_pass if passed else msg_fail
    print(f"  {symbol} {name}: {detail}" if detail else f"  {symbol} {name}")
    return passed


def check_python(training=False, reason="training"):
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if training:
        ok = v.major == 3 and v.minor == 11
        return check(
            "Python version",
            ok,
            msg_pass=version_str,
            msg_fail=f"{version_str} ({reason} requires 3.11.x)",
        )
    else:
        ok = v.major == 3 and v.minor >= 10
        return check(
            "Python version",
            ok,
            msg_pass=version_str,
            msg_fail=f"{version_str} (need 3.10+)",
        )


def check_git_lfs():
    lfs_installed = shutil.which("git-lfs") is not None
    if not lfs_installed:
        return check("Git LFS", False, msg_fail="not installed (sudo apt install git-lfs)")

    # Check if LFS files are pulled (sample an actual LFS-tracked mesh file)
    mesh_path = "gear_sonic/data/assets/robot_description/urdf/g1/meshes"
    stl_files = [os.path.join(mesh_path, f) for f in os.listdir(mesh_path) if f.endswith(".STL")] if os.path.isdir(mesh_path) else []
    sample_file = stl_files[0] if stl_files else "decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx"
    if os.path.exists(sample_file):
        size = os.path.getsize(sample_file)
        if size < 1000:
            return check(
                "Git LFS",
                False,
                msg_fail=f"{sample_file} is {size} bytes (LFS pointer — run 'git lfs pull')",
            )
        return check("Git LFS", True, msg_pass="installed, files pulled")
    return check("Git LFS", True, msg_pass="installed")


def check_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return check("CUDA", True, msg_pass=f"{device_name} (CUDA {cuda_version})")
        else:
            return check("CUDA", False, msg_fail="torch.cuda.is_available() = False")
    except ImportError:
        return check("CUDA", False, msg_fail="PyTorch not installed")


def check_torch():
    try:
        import torch

        return check("PyTorch", True, msg_pass=torch.__version__)
    except ImportError:
        return check(
            "PyTorch",
            False,
            msg_fail="not installed (pip install torch)",
        )


def check_isaaclab():
    try:
        import isaaclab

        version = getattr(isaaclab, "__version__", "unknown")
        return check("Isaac Lab", True, msg_pass=version)
    except ImportError:
        return check(
            "Isaac Lab",
            False,
            msg_fail="not installed — see https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html",
        )


def check_gear_sonic():
    try:
        ver = importlib.metadata.version("gear_sonic")
        return check("gear_sonic", True, msg_pass=f"installed ({ver})")
    except importlib.metadata.PackageNotFoundError:
        return check(
            "gear_sonic",
            False,
            msg_fail="not installed (pip install -e 'gear_sonic/[training]')",
        )


def check_training_deps():
    results = []
    for pkg, pip_name in [
        ("hydra", "hydra-core"),
        ("trl", "trl"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("wandb", "wandb"),
    ]:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "ok")
            results.append(check(pip_name, True, msg_pass=version))
        except ImportError:
            results.append(
                check(pip_name, False, msg_fail=f"not installed (pip install {pip_name})")
            )
    return all(results)


def check_mjlab():
    try:
        import mjlab

        version = getattr(mjlab, "__version__", "unknown")
        return check("mjlab", True, msg_pass=version)
    except ImportError:
        return check(
            "mjlab",
            False,
            msg_fail="not installed (pip install 'mjlab==1.5.0' or run install_scripts/install_mjlab_training.sh)",
        )


def check_mujoco():
    try:
        import mujoco

        version = getattr(mujoco, "__version__", "unknown")
        return check("MuJoCo", True, msg_pass=version)
    except ImportError:
        return check("MuJoCo", False, msg_fail="not installed (installed by mjlab)")


def check_torch_npu():
    try:
        import torch
        import torch_npu  # noqa: F401

        try:
            count = torch.npu.device_count()
            available = torch.npu.is_available()
            detail = f"torch_npu imported, devices={count}, available={available}"
            return check("Ascend NPU", available and count > 0, msg_pass=detail, msg_fail=detail)
        except Exception as exc:
            return check("Ascend NPU", False, msg_fail=f"torch_npu import ok, runtime check failed: {exc}")
    except ImportError as exc:
        return check("torch-npu", False, msg_fail=f"not installed or not importable: {exc}")


def check_tensorrt():
    trt_root = os.environ.get("TensorRT_ROOT", "")
    if not trt_root:
        return check(
            "TensorRT",
            False,
            msg_fail="TensorRT_ROOT not set (export TensorRT_ROOT=$HOME/TensorRT)",
        )
    if not os.path.isdir(trt_root):
        return check("TensorRT", False, msg_fail=f"TensorRT_ROOT={trt_root} does not exist")

    # Check for the library
    lib_dir = os.path.join(trt_root, "lib")
    if os.path.isdir(lib_dir):
        libs = [f for f in os.listdir(lib_dir) if "nvinfer" in f and f.endswith(".so")]
        if libs:
            # Try to extract version from filename
            for lib in libs:
                if "nvinfer.so." in lib:
                    version = lib.split("nvinfer.so.")[-1]
                    return check("TensorRT", True, msg_pass=f"{version} at {trt_root}")
            return check("TensorRT", True, msg_pass=f"found at {trt_root}")

    return check("TensorRT", False, msg_fail=f"libnvinfer not found in {lib_dir}")


def check_disk_space():
    stat = os.statvfs(".")
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    ok = free_gb > 10
    return check(
        "Disk space",
        ok,
        msg_pass=f"{free_gb:.0f} GB free",
        msg_fail=f"{free_gb:.1f} GB free (recommend 10+ GB)",
    )


def main():
    mode = "all"
    if "--training" in sys.argv:
        mode = "training"
    elif "--mjlab" in sys.argv:
        mode = "mjlab"
    elif "--npu" in sys.argv:
        mode = "npu"
    elif "--deploy" in sys.argv:
        mode = "deploy"

    print(f"GR00T-WholeBodyControl Environment Check")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python:   {sys.executable}")
    print()

    all_pass = True

    # Basic checks (always run)
    print("Basic:")
    python_reason = "training/mjlab" if mode in ("mjlab", "npu") else "training"
    all_pass &= check_python(training=(mode in ("all", "training", "mjlab", "npu")), reason=python_reason)
    all_pass &= check_git_lfs()
    if mode not in ("mjlab", "npu"):
        all_pass &= check_cuda()
    all_pass &= check_torch()
    all_pass &= check_disk_space()
    print()

    if mode in ("all", "training"):
        print("Training:")
        all_pass &= check_isaaclab()
        all_pass &= check_gear_sonic()
        all_pass &= check_training_deps()
        print()

    if mode in ("all", "mjlab", "npu"):
        print("mjlab Training:")
        all_pass &= check_gear_sonic()
        all_pass &= check_training_deps()
        all_pass &= check_mjlab()
        all_pass &= check_mujoco()
        print()

    if mode in ("all", "npu"):
        print("Ascend NPU:")
        all_pass &= check_torch_npu()
        print()

    if mode in ("all", "deploy"):
        print("Deployment:")
        all_pass &= check_tensorrt()
        print()

    if all_pass:
        print("All checks passed.")
    else:
        print("Some checks failed. See above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
