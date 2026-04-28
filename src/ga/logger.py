"""Logging: rich console + CSV history + optional TensorBoard + JSONL audit trail."""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    Console = None  # type: ignore

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False
    SummaryWriter = None  # type: ignore


class GALogger:
    """One logger per GA run. Writes to:
        {output_dir}/ga.log          (plaintext file log)
        {output_dir}/history.csv     (per-individual fitness records)
        {output_dir}/history.jsonl   (full JSON audit trail)
        {output_dir}/tb/             (tensorboard, if enabled + available)
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        tensorboard: bool = True,
        csv_log: bool = True,
        level: int = logging.INFO,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console(highlight=False) if _HAS_RICH else None

        # File logger
        self._logger = logging.getLogger(f"ga.{id(self)}")
        self._logger.setLevel(level)
        self._logger.propagate = False
        if not self._logger.handlers:
            fh = logging.FileHandler(self.output_dir / "ga.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self._logger.addHandler(fh)

        # CSV
        self._csv_path = self.output_dir / "history.csv" if csv_log else None
        if self._csv_path and not self._csv_path.exists():
            with self._csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "ts", "generation", "individual", "combo_key", "hash",
                    "fitness", "iou", "boundary_f1", "tv_penalty", "duration_s", "status",
                ])

        # JSONL audit
        self._jsonl_path = self.output_dir / "history.jsonl"

        # TensorBoard
        self._tb: Optional[SummaryWriter] = None
        if tensorboard and _HAS_TB:
            self._tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))

    # ---------- simple pass-through ----------
    def info(self, msg: str) -> None:
        self._logger.info(msg)
        if self.console:
            self.console.print(msg)
        else:
            print(msg)

    def warn(self, msg: str) -> None:
        self._logger.warning(msg)
        if self.console:
            self.console.print(f"[yellow]⚠ {msg}[/yellow]")
        else:
            print(f"WARN: {msg}")

    def error(self, msg: str) -> None:
        self._logger.error(msg)
        if self.console:
            self.console.print(f"[red]✗ {msg}[/red]")
        else:
            print(f"ERROR: {msg}")

    # ---------- structured events ----------
    def log_individual(
        self,
        *,
        generation: int,
        idx: int,
        combo_key: str,
        genome_hash: str,
        fitness: float,
        metrics: dict[str, float],
        duration_s: float,
        status: str,
    ) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        if self._csv_path:
            with self._csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    ts, generation, idx, combo_key, genome_hash,
                    f"{fitness:.6f}",
                    f"{metrics.get('iou', float('nan')):.6f}",
                    f"{metrics.get('boundary_f1', float('nan')):.6f}",
                    f"{metrics.get('tv_penalty', float('nan')):.6f}",
                    f"{duration_s:.1f}",
                    status,
                ])
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": ts, "generation": generation, "individual": idx,
                "combo_key": combo_key, "hash": genome_hash,
                "fitness": fitness, "metrics": metrics,
                "duration_s": duration_s, "status": status,
            }) + "\n")

        if self._tb is not None:
            step = generation * 1000 + idx
            self._tb.add_scalar("fitness", fitness, step)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(f"metrics/{k}", v, step)

    def log_generation_summary(
        self,
        generation: int,
        fitnesses: list[float],
        best_genome: Any,
    ) -> None:
        import statistics as stats
        valid = [f for f in fitnesses if f > float("-inf")]
        summary = {
            "generation": generation,
            "n_valid": len(valid),
            "best": max(valid) if valid else float("-inf"),
            "mean": stats.mean(valid) if valid else float("-inf"),
            "median": stats.median(valid) if valid else float("-inf"),
            "worst": min(valid) if valid else float("-inf"),
            "best_combo": getattr(best_genome, "combo_key", None),
        }
        self._logger.info(f"GEN {generation} summary: {json.dumps(summary)}")
        if self.console:
            tbl = Table(title=f"Generation {generation} summary", show_header=False)
            for k, v in summary.items():
                tbl.add_row(str(k), str(v))
            self.console.print(tbl)
        if self._tb is not None:
            self._tb.add_scalar("gen/best", summary["best"], generation)
            self._tb.add_scalar("gen/mean", summary["mean"], generation)
            self._tb.add_scalar("gen/median", summary["median"], generation)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
        for h in list(self._logger.handlers):
            h.close()
            self._logger.removeHandler(h)
