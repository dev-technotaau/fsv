"""training/provenance.py — Gather "who/what/when" metadata for checkpoints.

Captures everything you'd want to know 6 months from now when looking at
a mystery .pt file:
    - what code version produced it          (git SHA + dirty flag)
    - what library versions were active       (torch, transformers, ...)
    - when                                    (UTC timestamp)
    - where                                   (hostname + platform)
    - what hardware                            (GPU name + VRAM)

All gathered safely — failure to read any field never raises.
"""
from __future__ import annotations

import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone


def _git_info() -> dict:
    """Return git SHA + dirty flag + branch + remote, or empty dict if not in a repo."""
    out: dict = {}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if sha.returncode == 0:
            out["sha"] = sha.stdout.strip()
        # Branch
        br = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if br.returncode == 0:
            out["branch"] = br.stdout.strip()
        # Dirty flag (any uncommitted changes)
        st = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if st.returncode == 0:
            out["dirty"] = bool(st.stdout.strip())
        # Remote URL
        rm = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, timeout=5,
        )
        if rm.returncode == 0 and rm.stdout.strip():
            out["remote"] = rm.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return out


def _library_versions() -> dict:
    """Versions of the key libraries the checkpoint depends on."""
    libs: dict = {"python": sys.version.split()[0]}
    for mod_name in ("torch", "torchvision", "transformers", "albumentations",
                     "numpy", "PIL", "cv2"):
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
            if ver is not None:
                key = "Pillow" if mod_name == "PIL" else mod_name
                libs[key] = str(ver)
        except Exception:
            continue
    return libs


def _gpu_info() -> dict:
    """GPU model + VRAM (if CUDA available), else empty."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / 1e9, 2),
                "device_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
            }
    except Exception:
        pass
    return {}


def collect() -> dict:
    """Snapshot everything. Safe to call from anywhere; never raises."""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
        "git": _git_info(),
        "libraries": _library_versions(),
        "gpu": _gpu_info(),
    }
