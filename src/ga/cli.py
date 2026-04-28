"""GA CLI — Typer-based.

Commands:
  list-combos           List all 18 combos with tier + status (full/stub).
  preflight             Run environment / data / disk / GPU checks.
  run                   Run the GA (either stage1 or stage2 per config).
  stage1                Shortcut: force stage1 regardless of config.
  stage2                Shortcut: force stage2 on a given combo key.
  resume                Resume from a checkpoint.
  dry-run               Sanity-check config and adapters without real training.
  inspect               Inspect a checkpoint or hall-of-fame JSON.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    print("typer required. pip install typer[all] rich pyyaml pydantic")
    raise

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None  # type: ignore

from .config import Config, load_config
from .exceptions import AdapterNotImplementedError, ConfigError, PreflightError
from .fitness_cache import FitnessCache
from .logger import GALogger
from .population import GARunner
from .registry import COMBOS, all_keys, get_combo
from .validators import preflight as run_preflight


app = typer.Typer(
    help="Ninja Fence Staining — Genetic Algorithm search over 18 model combos.",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


# ---------- helpers ----------

def _check_adapter_importable(combo_key: str) -> tuple[bool, str]:
    """Return (ok, message) — ok=True if adapter is a real impl, False if stub or missing."""
    combo = get_combo(combo_key)
    module_path, class_name = combo.adapter.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        return False, f"missing module: {e}"
    if not hasattr(mod, class_name):
        return False, f"class not found: {class_name}"
    # Heuristic: stub adapters import AdapterNotImplementedError
    src = Path(mod.__file__ or "").read_text(encoding="utf-8", errors="ignore") if getattr(mod, "__file__", None) else ""
    if "AdapterNotImplementedError" in src and "raise AdapterNotImplementedError" in src:
        return False, "stub (not implemented)"
    return True, "ok"


def _print(s: str) -> None:
    if console is not None:
        console.print(s)
    else:
        print(s)


# ---------- commands ----------

@app.command("list-combos")
def cmd_list_combos(
    tier: Optional[str] = typer.Option(None, "--tier", "-t", help="Filter by tier A/B/C"),
    show_stubs: bool = typer.Option(True, "--stubs/--no-stubs", help="Show stub status column"),
) -> None:
    """List all 18 combos with tier, status, and adapter path."""
    if console is None:
        for c in COMBOS:
            if tier and c.tier != tier:
                continue
            print(f"  {c.tier}  {c.key:<36} {c.name}")
        return
    table = Table(title="GA combo registry (18 total)", show_lines=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Tier")
    table.add_column("Name")
    if show_stubs:
        table.add_column("Status", style="yellow")
    for c in COMBOS:
        if tier and c.tier != tier:
            continue
        row = [c.key, c.tier, c.name]
        if show_stubs:
            ok, msg = _check_adapter_importable(c.key)
            row.append("[green]full[/green]" if ok else f"[yellow]{msg}[/yellow]")
        table.add_row(*row)
    console.print(table)


@app.command("preflight")
def cmd_preflight(
    config: Path = typer.Option(..., "--config", "-c", help="Path to GA config YAML"),
    set_: list[str] = typer.Option([], "--set", help="Override: dotted.key=value"),
) -> None:
    """Run environment / data / GPU / disk pre-flight checks."""
    cfg = load_config(config, overrides=set_)
    warnings = run_preflight(cfg)
    if not warnings:
        _print("[green]✓ preflight passed — no warnings[/green]")
    else:
        _print("[yellow]⚠ preflight warnings:[/yellow]")
        for w in warnings:
            _print(f"  - {w}")


@app.command("run")
def cmd_run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to GA config YAML"),
    set_: list[str] = typer.Option([], "--set", help="Override: dotted.key=value"),
    no_subprocess: bool = typer.Option(False, "--no-subprocess",
                                       help="Run adapters in-process (debugging only)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Random fitness, no real training"),
) -> None:
    """Run the GA (stage specified in config)."""
    cfg = load_config(config, overrides=set_)
    if dry_run:
        cfg.runtime.dry_run = True

    # Pre-flight
    try:
        warnings = run_preflight(cfg)
    except PreflightError as e:
        _print(f"[red]preflight FAILED: {e}[/red]")
        raise typer.Exit(code=2)
    for w in warnings:
        _print(f"[yellow]⚠ {w}[/yellow]")

    # Set up run
    logger = GALogger(
        cfg.runtime.output_dir,
        tensorboard=cfg.runtime.log_to_tensorboard,
        csv_log=cfg.runtime.log_to_csv,
    )
    cache = FitnessCache(cfg.runtime.output_dir / "fitness_cache.jsonl")
    logger.info(f"Fitness cache contains {len(cache)} prior evaluations")
    runner = GARunner(cfg, logger=logger, cache=cache, use_subprocess=not no_subprocess)
    try:
        runner.run()
    finally:
        logger.close()


@app.command("stage1")
def cmd_stage1(
    config: Path = typer.Option(Path("configs/ga_stage1_model_search.yaml"), "--config", "-c"),
    set_: list[str] = typer.Option([], "--set"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Shortcut: run stage1 (model-family search)."""
    cfg = load_config(config, overrides=set_ + ["ga.stage=stage1_model", "ga.fixed_combo=None"])
    if dry_run:
        cfg.runtime.dry_run = True
    _run_cfg(cfg, no_subprocess=False)


@app.command("stage2")
def cmd_stage2(
    combo_key: str = typer.Argument(..., help="Combo key to fine-tune (e.g. 01_dinov2_l_m2f)"),
    config: Path = typer.Option(Path("configs/ga_stage2_hyperparam_search.yaml"), "--config", "-c"),
    set_: list[str] = typer.Option([], "--set"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Shortcut: run stage2 hyperparam GA on the given combo key."""
    if combo_key not in all_keys():
        _print(f"[red]Unknown combo key: {combo_key}. Use `list-combos` to see available.[/red]")
        raise typer.Exit(code=2)
    cfg = load_config(config, overrides=set_ + [
        "ga.stage=stage2_hyperparam",
        f"ga.fixed_combo={combo_key}",
    ])
    if dry_run:
        cfg.runtime.dry_run = True
    _run_cfg(cfg, no_subprocess=False)


@app.command("resume")
def cmd_resume(
    checkpoint: Path = typer.Argument(..., help="Path to ga_checkpoint.pkl"),
    config: Path = typer.Option(..., "--config", "-c"),
    set_: list[str] = typer.Option([], "--set"),
) -> None:
    """Resume a GA run from checkpoint."""
    cfg = load_config(config, overrides=set_ + [f"runtime.resume_from={checkpoint}"])
    _run_cfg(cfg, no_subprocess=False)


@app.command("inspect")
def cmd_inspect(
    path: Path = typer.Argument(..., help="Path to hall_of_fame.json or history.jsonl"),
) -> None:
    """Pretty-print hall_of_fame.json or history.jsonl."""
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if console is not None:
                t = Table(title=f"{path.name} (top {min(20, len(data))})")
                for col in ["rank", "combo_key", "fitness", "iou", "boundary_f1"]:
                    t.add_column(col)
                for rec in data[:20]:
                    t.add_row(
                        str(rec.get("rank", "")),
                        str(rec.get("combo_key", "")),
                        f"{rec.get('fitness', 0):.4f}",
                        f"{rec.get('metrics', {}).get('iou', 0):.4f}",
                        f"{rec.get('metrics', {}).get('boundary_f1', 0):.4f}",
                    )
                console.print(t)
            else:
                for rec in data[:20]:
                    print(rec)
        else:
            _print(json.dumps(data, indent=2))
    else:
        # jsonl
        n = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            _print(json.dumps(rec))
            n += 1
        _print(f"Total records: {n}")


@app.command("dry-run")
def cmd_dry_run(
    config: Path = typer.Option(Path("configs/ga_stage1_model_search.yaml"), "--config", "-c"),
    set_: list[str] = typer.Option([], "--set"),
) -> None:
    """Dry run: validate config + test adapter imports + emit random fitnesses."""
    cfg = load_config(config, overrides=set_)
    cfg.runtime.dry_run = True

    _print("[cyan]== Dry run: combo adapter import check ==[/cyan]")
    ok_count, stub_count, err_count = 0, 0, 0
    for c in COMBOS:
        ok, msg = _check_adapter_importable(c.key)
        tag = "[green]ok[/green]" if ok else ("[yellow]stub[/yellow]" if "stub" in msg else "[red]err[/red]")
        _print(f"  {c.key:<36} {tag}  {msg}")
        if ok: ok_count += 1
        elif "stub" in msg: stub_count += 1
        else: err_count += 1
    _print(f"\nSummary: {ok_count} full, {stub_count} stubs, {err_count} errors")

    _print("\n[cyan]== Dry GA loop (random fitness) ==[/cyan]")
    _run_cfg(cfg, no_subprocess=True)


# ---------- shared runner ----------
def _run_cfg(cfg: Config, no_subprocess: bool) -> None:
    try:
        warnings = run_preflight(cfg)
    except PreflightError as e:
        _print(f"[red]preflight FAILED: {e}[/red]")
        raise typer.Exit(code=2)
    for w in warnings:
        _print(f"[yellow]⚠ {w}[/yellow]")

    logger = GALogger(
        cfg.runtime.output_dir,
        tensorboard=cfg.runtime.log_to_tensorboard,
        csv_log=cfg.runtime.log_to_csv,
    )
    cache = FitnessCache(cfg.runtime.output_dir / "fitness_cache.jsonl")
    logger.info(f"Fitness cache: {len(cache)} prior evaluations loaded")
    runner = GARunner(cfg, logger=logger, cache=cache, use_subprocess=not no_subprocess)
    try:
        runner.run()
    finally:
        logger.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
