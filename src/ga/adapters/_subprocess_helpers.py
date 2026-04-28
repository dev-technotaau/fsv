"""Helpers for adapters that shell out to existing project training scripts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def run_training_subprocess(
    cmd: list[str],
    *,
    work_dir: Path,
    env_overrides: Optional[dict[str, str]] = None,
    timeout_s: Optional[int] = None,
) -> tuple[int, str]:
    """Launch a training subprocess, tee stdout to training.log, return (rc, log_text)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    log_path = work_dir / "training.log"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line)
                f.flush()
                if timeout_s and time.time() - t0 > timeout_s:
                    proc.kill()
                    return 124, log_path.read_text(encoding="utf-8", errors="replace")
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            raise
    return rc, log_path.read_text(encoding="utf-8", errors="replace")


def extract_best_iou_from_log(log_text: str) -> Optional[float]:
    """Parse the last 'best IoU: X.XXX' pattern from training log.
    Your project's scripts log things like:
        best IoU: 0.8342
        best val iou: 0.8342
        val_iou: 0.8342
    """
    import re
    patterns = [
        r"best[_ ]*(?:val[_ ]*)?iou[:\s=]+([0-9]*\.?[0-9]+)",
        r"val[_ ]*iou[:\s=]+([0-9]*\.?[0-9]+)",
        r"iou[:\s=]+([0-9]*\.?[0-9]+)",
    ]
    last: Optional[float] = None
    for pat in patterns:
        for m in re.finditer(pat, log_text, re.IGNORECASE):
            try:
                last = float(m.group(1))
            except ValueError:
                continue
    return last


def find_latest_checkpoint(dir_: Path, glob: str = "best_model*.pth") -> Optional[Path]:
    """Find most recently-modified checkpoint matching glob."""
    cands = sorted(dir_.rglob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None
