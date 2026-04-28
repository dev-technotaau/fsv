"""Fitness evaluator — trains a model adapter and computes composite fitness.

Runs each individual in an isolated subprocess-spawned Python process so that
crashes, OOMs, and CUDA context corruption don't take down the GA loop.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .adapters.base import AdapterResult, ModelAdapter
from .config import Config
from .exceptions import AdapterError, FitnessError
from .genome import Genome


# ---------- composite fitness ----------

def compute_composite_fitness(
    metrics: dict[str, float],
    *,
    iou_weight: float,
    boundary_f1_weight: float,
    tv_smoothness_penalty: float,
) -> float:
    """Composite fitness. Higher is better.

    composite = iou_weight*IoU + boundary_f1_weight*BF1 - tv_smoothness_penalty*TV
    Falls back to IoU alone if BF1/TV missing. Returns -inf if IoU missing or NaN.
    """
    iou = metrics.get("iou")
    if iou is None or (isinstance(iou, float) and math.isnan(iou)):
        return float("-inf")
    bf1 = metrics.get("boundary_f1", 0.0) or 0.0
    tv  = metrics.get("tv_penalty", 0.0) or 0.0
    return iou_weight * iou + boundary_f1_weight * bf1 - tv_smoothness_penalty * tv


def score_genome(genome: Genome, metrics: dict[str, float], cfg: Config) -> float:
    fc = cfg.fitness
    if fc.metric == "iou":
        return float(metrics.get("iou", float("-inf")))
    if fc.metric == "boundary_f1":
        return float(metrics.get("boundary_f1", float("-inf")))
    return compute_composite_fitness(
        metrics,
        iou_weight=fc.iou_weight,
        boundary_f1_weight=fc.boundary_f1_weight,
        tv_smoothness_penalty=fc.tv_smoothness_penalty,
    )


# ---------- adapter loading ----------

def load_adapter_class(dotted_path: str) -> type[ModelAdapter]:
    """Dynamically import an adapter class."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise AdapterError(f"Cannot import adapter module {module_path!r}: {e}") from e
    if not hasattr(mod, class_name):
        raise AdapterError(f"Module {module_path!r} has no class {class_name!r}")
    cls = getattr(mod, class_name)
    if not isinstance(cls, type) or not issubclass(cls, ModelAdapter):
        raise AdapterError(f"{dotted_path} is not a ModelAdapter subclass")
    return cls


# ---------- single-individual runner (used both in-process and via subprocess) ----------

@dataclass
class FitnessResult:
    fitness: float
    metrics: dict[str, float]
    duration_s: float
    status: str                         # "ok" | "crashed" | "timeout" | "killed_early" | "cached"
    error: Optional[str] = None
    artifacts_dir: Optional[Path] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitness": self.fitness,
            "metrics": self.metrics,
            "duration_s": self.duration_s,
            "status": self.status,
            "error": self.error,
            "artifacts_dir": str(self.artifacts_dir) if self.artifacts_dir else None,
        }


def evaluate_individual_in_process(
    genome: Genome,
    cfg: Config,
    *,
    work_dir: Path,
    gpu_id: Optional[int] = None,
) -> FitnessResult:
    """In-process fitness eval. Use only for debugging / dry runs.
    For production, use `evaluate_individual_subprocess` below."""
    from .registry import get_combo

    t0 = time.time()
    combo = get_combo(genome.combo_key)
    try:
        adapter_cls = load_adapter_class(combo.adapter)
    except AdapterError as e:
        return FitnessResult(
            fitness=float("-inf"), metrics={}, duration_s=time.time() - t0,
            status="crashed", error=str(e),
        )

    adapter = adapter_cls(
        genome=genome,
        data_cfg=cfg.data,
        fitness_cfg=cfg.fitness,
        work_dir=work_dir,
        gpu_id=gpu_id,
    )
    try:
        result: AdapterResult = adapter.run()
    except Exception as e:
        import traceback
        return FitnessResult(
            fitness=float("-inf"), metrics={}, duration_s=time.time() - t0,
            status="crashed",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            artifacts_dir=work_dir,
        )

    fitness = score_genome(genome, result.metrics, cfg)
    return FitnessResult(
        fitness=fitness,
        metrics=result.metrics,
        duration_s=time.time() - t0,
        status=result.status,
        error=result.error,
        artifacts_dir=work_dir,
    )


def evaluate_individual_subprocess(
    genome: Genome,
    cfg: Config,
    *,
    work_dir: Path,
    gpu_id: Optional[int] = None,
    timeout_minutes: int = 240,
) -> FitnessResult:
    """Subprocess-isolated fitness eval — preferred for production.

    Spawns `python -m src.ga.fitness` with genome+config written to a temp json file.
    Child writes result.json back; we parse it. Crashes in child don't kill us.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    genome_path = work_dir / "genome.json"
    config_path = work_dir / "config.json"
    result_path = work_dir / "result.json"

    with genome_path.open("w", encoding="utf-8") as f:
        json.dump(genome.to_dict(), f)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(mode="json"), f, default=str)

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "-m", "src.ga.fitness",
        "--genome", str(genome_path),
        "--config", str(config_path),
        "--work-dir", str(work_dir),
        "--result", str(result_path),
    ]
    if gpu_id is not None:
        cmd += ["--gpu-id", str(gpu_id)]

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as e:
        return FitnessResult(
            fitness=float("-inf"), metrics={}, duration_s=time.time() - t0,
            status="crashed", error=f"Subprocess spawn failed: {e}",
        )

    # Tee child's stdout to a file
    log_path = work_dir / "subprocess.log"
    timeout_s = timeout_minutes * 60
    try:
        with log_path.open("w", encoding="utf-8") as log_f:
            try:
                for line in proc.stdout or []:
                    log_f.write(line)
                    log_f.flush()
                    if time.time() - t0 > timeout_s:
                        raise TimeoutError("Subprocess exceeded timeout")
            except TimeoutError:
                _kill_proc_tree(proc)
                return FitnessResult(
                    fitness=float("-inf"), metrics={}, duration_s=time.time() - t0,
                    status="timeout", error=f"Timed out after {timeout_minutes} min",
                    artifacts_dir=work_dir,
                )
        returncode = proc.wait()
    except KeyboardInterrupt:
        _kill_proc_tree(proc)
        raise

    if returncode != 0 or not result_path.exists():
        err = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        return FitnessResult(
            fitness=float("-inf"), metrics={}, duration_s=time.time() - t0,
            status="crashed",
            error=f"Subprocess rc={returncode}; last log:\n{err}",
            artifacts_dir=work_dir,
        )

    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return FitnessResult(
        fitness=payload["fitness"],
        metrics=payload["metrics"],
        duration_s=time.time() - t0,
        status=payload.get("status", "ok"),
        error=payload.get("error"),
        artifacts_dir=work_dir,
    )


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ---------- `python -m src.ga.fitness` entry point (child process) ----------

def _child_main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=None)
    args = parser.parse_args()

    from .config import Config

    with args.genome.open("r", encoding="utf-8") as f:
        genome = Genome.from_dict(json.load(f))
    with args.config.open("r", encoding="utf-8") as f:
        cfg = Config(**json.load(f))

    result = evaluate_individual_in_process(
        genome, cfg, work_dir=args.work_dir, gpu_id=args.gpu_id
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


if __name__ == "__main__":
    _child_main()
