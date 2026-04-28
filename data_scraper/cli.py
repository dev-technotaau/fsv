"""Scraper CLI.

Commands:
  run              Run the full scrape.
  run --dry-run    Simulate (no downloads; still runs through pipeline).
  preflight        Check config + credentials + connectivity.
  queries          Preview the list of search queries.
  stats            Print dedup DB stats.
  export           Export metadata to CSV.
  repair           Scan images_dir; drop DB rows whose files are missing.
  retry-failures   Re-attempt URLs previously logged as failures.
"""
from __future__ import annotations

import asyncio
import csv
import json
import sys
from pathlib import Path

try:
    import typer
except ImportError:
    print("typer required: pip install typer[all] rich pyyaml pydantic httpx pillow")
    sys.exit(2)

from .config import Config, load_config
from .coordinator import Coordinator
from .dedup import DedupStore, sha256_of_bytes, dhash_of_bytes
from .downloader import Downloader
from .logger import StructuredLogger, get_logger
from .proxy_rotator import ProxyRotator
from .quality import check_bytes, harden_pil
from .queries import build_queries
from .storage import Storage


app = typer.Typer(
    help="Ninja Fence — multi-source image scraper.",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


def _configure_logging(cfg: Config) -> None:
    StructuredLogger.configure(
        level=cfg.runtime.log_level,
        file_path=cfg.runtime.log_file,
        console_format=cfg.runtime.log_format,
    )


@app.command()
def run(
    config: Path = typer.Option(Path("configs/scraper.yaml"), "--config", "-c"),
    set_: list[str] = typer.Option([], "--set"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without downloading"),
) -> None:
    """Run the full multi-source scrape."""
    cfg = load_config(config, overrides=set_)
    if dry_run:
        cfg.runtime.dry_run = True
    asyncio.run(_run(cfg))


async def _run(cfg: Config) -> None:
    coord = Coordinator(cfg)
    try:
        await coord.run()
    except KeyboardInterrupt:
        get_logger("cli").warn("keyboard_interrupt_stopping")
        coord.stop_event.set()
    finally:
        await coord.close()


@app.command()
def preflight(
    config: Path = typer.Option(Path("configs/scraper.yaml"), "--config", "-c"),
    set_: list[str] = typer.Option([], "--set"),
) -> None:
    """Verify config, API keys, paths, Playwright, Redis, disk, proxies."""
    cfg = load_config(config, overrides=set_)
    _configure_logging(cfg)
    log = get_logger("preflight")
    log.info("preflight_start")

    for name, subcfg, key_attr in [
        ("google_cse", cfg.google_cse, "api_key"),
        ("pexels",     cfg.pexels,     "api_key"),
        ("unsplash",   cfg.unsplash,   "access_key"),
        ("pixabay",    cfg.pixabay,    "api_key"),
        ("flickr",     cfg.flickr,     "api_key"),
    ]:
        if not subcfg.enabled:
            log.info("source_disabled", name=name)
            continue
        key = getattr(subcfg, key_attr, None)
        (log.info if key else log.warn)("source_state", name=name,
                                         has_key=bool(key))
    for name, subcfg in [
        ("wikimedia", cfg.wikimedia), ("reddit", cfg.reddit),
        ("pw_google", cfg.pw_google), ("pw_bing", cfg.pw_bing),
        ("pw_ddg", cfg.pw_ddg),
    ]:
        log.info("source_state", name=name, enabled=subcfg.enabled)

    # paths
    for p, label in [(cfg.storage.images_dir, "images_dir"),
                     (cfg.storage.metadata_jsonl.parent, "metadata_dir"),
                     (cfg.dedup.db_path.parent, "dedup_db_dir")]:
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
            log.info("path_writable", label=label, path=str(p))
        except Exception as e:
            log.error("path_not_writable", label=label, path=str(p), error=str(e))

    # disk
    try:
        import shutil
        free_gb = shutil.disk_usage(str(cfg.storage.images_dir)).free / (1024 ** 3)
        (log.info if free_gb >= cfg.runtime.min_free_disk_gb else log.warn)(
            "disk_space", free_gb=round(free_gb, 2), min_gb=cfg.runtime.min_free_disk_gb)
    except Exception as e:
        log.error("disk_check_failed", error=str(e))

    # playwright
    try:
        from playwright.async_api import async_playwright   # noqa
        log.info("playwright_ok")
    except ImportError:
        log.warn("playwright_missing",
                 hint="pip install playwright && playwright install chromium")

    # redis
    if cfg.distributed.enabled:
        try:
            import redis.asyncio   # type: ignore # noqa
            log.info("redis_module_ok")
        except ImportError:
            log.warn("redis_missing", hint="pip install redis>=5.0")

    # proxies
    if cfg.proxy.enabled:
        pr = ProxyRotator(proxies=cfg.proxy.proxies)
        log.info("proxy_rotator", **pr.snapshot())


@app.command()
def queries(
    config: Path = typer.Option(Path("configs/scraper.yaml"), "--config", "-c"),
    set_: list[str] = typer.Option([], "--set"),
) -> None:
    """Preview the full query list."""
    cfg = load_config(config, overrides=set_)
    qs = build_queries(
        static=cfg.queries.use_static, custom=cfg.queries.custom,
        gemini_extra=cfg.queries.gemini_target_extra if cfg.queries.use_gemini_expansion else 0,
        gemini_api_key=cfg.queries.gemini_api_key,
    )
    print(f"{len(qs)} queries:")
    for i, q in enumerate(qs):
        print(f"  {i:3d}. {q}")


@app.command()
def stats(
    config: Path = typer.Option(Path("configs/scraper.yaml"), "--config", "-c"),
) -> None:
    """Print dedup DB stats."""
    cfg = load_config(config)
    if not cfg.dedup.db_path.exists():
        print(f"No dedup DB at {cfg.dedup.db_path}")
        return
    store = DedupStore(cfg.dedup.db_path, phash_hamming_threshold=cfg.dedup.phash_hamming_threshold)
    print(f"Total images      : {store.count()}")
    print(f"URLs seen         : {store.url_seen_count()}")
    print(f"Failures total    : {store.failures_count()}")
    print(f"Failures unretried: {store.failures_count(unretried_only=True)}")
    print("By source:")
    for src, n in sorted(store.counts_by_source().items(), key=lambda x: -x[1]):
        print(f"  {src:<14} {n}")


@app.command()
def export(
    config: Path = typer.Option(Path("configs/scraper.yaml"), "--config", "-c"),
    output: Path = typer.Option(Path("data_scraped/metadata.csv"), "--output", "-o"),
) -> None:
    """Export metadata.jsonl → CSV."""
    cfg = load_config(config)
    if not cfg.storage.metadata_jsonl.exists():
        print(f"No metadata at {cfg.storage.metadata_jsonl}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "source", "query", "origin_url", "origin_page", "title",
            "sha256", "width", "height", "bytes"]
    with cfg.storage.metadata_jsonl.open("r", encoding="utf-8") as inf, \
         output.open("w", encoding="utf-8", newline="") as outf:
        w = csv.writer(outf)
        w.writerow(cols)
        n = 0
        for line in inf:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            w.writerow([r.get(c, "") for c in cols])
            n += 1
    print(f"wrote {n} rows → {output}")


@app.command()
def repair(
    config: Path = typer.Option(Path("configs/scraper.yaml"), "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without modifying"),
) -> None:
    """Scan images_dir; drop DB rows whose files are missing. Rebuilds BK-tree."""
    cfg = load_config(config)
    _configure_logging(cfg)
    log = get_logger("repair")
    store = DedupStore(cfg.dedup.db_path, phash_hamming_threshold=cfg.dedup.phash_hamming_threshold)
    result = store.repair(cfg.storage.images_dir, dry_run=dry_run)
    log.info("repair_complete", **result)


@app.command("retry-failures")
def retry_failures(
    config: Path = typer.Option(Path("configs/scraper.yaml"), "--config", "-c"),
    set_: list[str] = typer.Option([], "--set"),
    limit: int = typer.Option(1000, "--limit"),
) -> None:
    """Re-attempt URLs previously logged as failures."""
    cfg = load_config(config, overrides=set_)
    _configure_logging(cfg)
    asyncio.run(_retry_failures(cfg, limit))


async def _retry_failures(cfg: Config, limit: int) -> None:
    log = get_logger("retry")
    harden_pil(cfg.runtime.max_image_megapixels)
    store = DedupStore(cfg.dedup.db_path, phash_hamming_threshold=cfg.dedup.phash_hamming_threshold)
    storage = Storage(cfg.storage.images_dir, cfg.storage.metadata_jsonl,
                      cfg.storage.failed_log, cfg.storage.file_extension)
    proxy_rotator = ProxyRotator(proxies=cfg.proxy.proxies) if cfg.proxy.enabled else None
    dl = Downloader(
        timeout_s=cfg.runtime.download_timeout_s,
        max_retries=cfg.runtime.max_retries,
        max_bytes=cfg.quality.max_bytes,
        proxy_rotator=proxy_rotator,
        use_circuit_breakers=cfg.circuit_breaker.enabled,
    )
    failures = store.get_unretried_failures(limit=limit)
    log.info("retry_start", total=len(failures))
    succeeded = 0
    try:
        for fid, url, source, reason in failures:
            try:
                body = await dl.fetch_image(url)
            except Exception:
                store.mark_failure_retried(fid)
                continue
            if not body:
                store.mark_failure_retried(fid)
                continue
            sha = sha256_of_bytes(body)
            if store.exists_sha256(sha):
                store.mark_failure_retried(fid)
                continue
            qc = check_bytes(body, cfg.quality)
            if not qc.ok:
                store.mark_failure_retried(fid)
                continue
            dh = dhash_of_bytes(body)
            if dh is not None and store.near_duplicate(dh):
                store.mark_failure_retried(fid)
                continue
            try:
                path = storage.save_image(image_bytes=body, source=source,
                                           query="__retry__", sha256=sha)
            except Exception:
                store.mark_failure_retried(fid)
                continue
            store.save_image(
                sha256=sha, dhash=dh, path=str(path), source=source,
                query="__retry__", origin_url=url,
                width=qc.width, height=qc.height, bytes_=len(body),
            )
            store.mark_failure_retried(fid)
            succeeded += 1
    finally:
        await dl.close()
    log.info("retry_done", succeeded=succeeded, total=len(failures))


@app.command("vision-qa")
def vision_qa(
    config: Path = typer.Option(Path("configs/scraper.yaml"), "--config", "-c"),
    set_: list[str] = typer.Option([], "--set"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                  help="Don't move files; just report"),
    batch_size: int = typer.Option(16, "--batch-size",
                                    help="Images per Vision batch call (max 16)"),
    parallelism: int = typer.Option(4, "--parallelism", "-p",
                                     help="Concurrent Vision batches in flight "
                                          "(default 4 → ~4× faster; use 1 for serial)"),
    skip_already_rejected: bool = typer.Option(True, "--skip-rejected/--recheck-rejected",
                                                help="Skip files already in rejected/ "
                                                     "(resume after interrupted run)"),
) -> None:
    """Post-hoc Google Vision QA over saved images. Moves failures to
    data_scraped/rejected/ and removes them from the dedup DB.

    Runs `parallelism` batches concurrently — byte upload + Vision API
    overlap across threads for ~4× throughput vs. serial.

    Requires GOOGLE_APPLICATION_CREDENTIALS to point at a valid service-account
    JSON. Cost: ~$1.50 per 1000 images at Vision list price.
    """
    cfg = load_config(config, overrides=set_)
    _configure_logging(cfg)
    _run_vision_qa(
        cfg, dry_run=dry_run,
        batch_size=min(16, max(1, batch_size)),
        parallelism=max(1, parallelism),
        skip_already_rejected=skip_already_rejected,
    )


def _run_vision_qa(cfg: Config, *, dry_run: bool, batch_size: int,
                   parallelism: int = 4,
                   skip_already_rejected: bool = True) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .google_vision import GoogleVisionVerifier

    log = get_logger("vision_qa")

    verifier = GoogleVisionVerifier(
        credentials_json=cfg.google_vision.credentials_json,
        min_confidence=cfg.google_vision.min_fence_confidence,
        labels_to_accept=cfg.google_vision.labels_to_accept,
        sample_rate=1.0,
        batch_size=batch_size,
        batch_timeout_ms=500,
        use_uri_mode=False,   # local bytes
    )
    if not verifier.enabled:
        log.error("vision_not_initialized",
                  hint="set GOOGLE_APPLICATION_CREDENTIALS to a valid service-account JSON")
        return

    store = DedupStore(cfg.dedup.db_path,
                       phash_hamming_threshold=cfg.dedup.phash_hamming_threshold)
    rejected_dir = cfg.runtime.output_root / "rejected"
    if not dry_run:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    images_dir = cfg.storage.images_dir
    all_paths = sorted(p for p in images_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))

    # Resume support: if the rejected/ dir has files from a prior run, we
    # don't need to re-check them. Also skip kept-but-already-processed files
    # by checkpoint file (if present) to avoid re-billing on restart.
    skip_names: set[str] = set()
    if skip_already_rejected and rejected_dir.exists():
        skip_names.update(p.name for p in rejected_dir.iterdir())

    checkpoint_path = cfg.runtime.output_root / "vision_qa_processed.txt"
    if skip_already_rejected and checkpoint_path.exists():
        try:
            with checkpoint_path.open("r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        skip_names.add(name)
        except Exception:
            pass

    paths = [p for p in all_paths if p.name not in skip_names]
    skipped = len(all_paths) - len(paths)
    total = len(paths)
    log.info("qa_start", total=total, already_processed=skipped, dry_run=dry_run,
             parallelism=parallelism, batch_size=batch_size,
             min_conf=cfg.google_vision.min_fence_confidence,
             rejected_dir=str(rejected_dir))

    # Thread-safe checkpoint writer + counters
    ckpt_lock = threading.Lock()
    ckpt_file = None
    if not dry_run:
        ckpt_file = checkpoint_path.open("a", encoding="utf-8")

    # Build the list of batches up front — main thread processes results as
    # workers finish, so order doesn't matter.
    batches = [paths[i:i + batch_size] for i in range(0, total, batch_size)]

    def process_batch(chunk: list[Path]) -> tuple[list[Path], list]:
        """Worker: read bytes + call Vision batch_annotate. Returns (chunk, results).
        Thread-safe because google.cloud.vision.ImageAnnotatorClient is documented
        as thread-safe for concurrent calls from multiple threads."""
        items = []
        for p in chunk:
            try:
                items.append((p.read_bytes(), None))
            except Exception:
                items.append((b"", None))
        try:
            results = verifier._batch_annotate(items)
        except Exception as e:
            log.warn("batch_failed", error=f"{type(e).__name__}: {str(e)[:120]}",
                     size=len(chunk))
            # Be permissive on batch-level failure — don't mass-reject on API hiccup
            results = [(True, True, None, 0.0, []) for _ in chunk]
        return chunk, results

    kept = 0
    rejected = 0
    errors = 0
    processed = 0
    ten_batch_log = max(1, min(10, max(len(batches) // 100, 1)))

    try:
        with ThreadPoolExecutor(max_workers=parallelism,
                                thread_name_prefix="qa") as pool:
            futs = [pool.submit(process_batch, b) for b in batches]
            for batch_idx, fut in enumerate(as_completed(futs)):
                try:
                    chunk, results = fut.result()
                except Exception as e:
                    log.warn("batch_future_failed",
                             error=f"{type(e).__name__}: {str(e)[:120]}")
                    continue

                for p, (_checked, accepted, label, conf, _labels) in zip(chunk, results):
                    processed += 1
                    if accepted:
                        kept += 1
                    else:
                        rejected += 1
                        if dry_run:
                            log.info("would_reject", path=p.name,
                                     best_label=label, confidence=round(conf, 3))
                        else:
                            try:
                                p.rename(rejected_dir / p.name)
                                store.delete_by_path(str(p))
                            except Exception as e:
                                log.warn("move_failed", path=str(p),
                                         error=str(e)[:120])
                                errors += 1
                    # Checkpoint processed name (whether kept or rejected)
                    if ckpt_file is not None:
                        with ckpt_lock:
                            ckpt_file.write(p.name + "\n")

                if (batch_idx + 1) % ten_batch_log == 0 or batch_idx + 1 == len(batches):
                    if ckpt_file is not None:
                        ckpt_file.flush()
                    log.info("qa_progress",
                             processed=processed, total=total,
                             kept=kept, rejected=rejected, errors=errors,
                             batches_done=batch_idx + 1, batches_total=len(batches))
    except KeyboardInterrupt:
        log.warn("qa_interrupted",
                 processed=processed, kept=kept, rejected=rejected,
                 hint="run again to resume; already-processed images are skipped")
    finally:
        if ckpt_file is not None:
            ckpt_file.close()

    log.info("qa_done", total=total, kept=kept, rejected=rejected, errors=errors,
             pct_rejected=round(100 * rejected / max(total, 1), 1))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
