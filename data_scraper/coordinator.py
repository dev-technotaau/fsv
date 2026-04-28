"""Coordinator — orchestrates all sources, dedup, downloads, filters, logging.

Wires in all the enterprise-grade components:
  - StructuredLogger
  - DiskGuard (pre-flight + periodic)
  - PIL bomb hardening
  - BK-tree dedup
  - Circuit breakers + proxy rotation in Downloader
  - Adaptive rate limits in TokenBucket
  - Per-query progress (in DedupStore)
  - ContentFilter
  - QueryPriorityScheduler feedback
  - Optional Redis distributed coordination
  - Dry-run mode
"""
from __future__ import annotations

import asyncio
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .config import Config
from .content_filter import ContentFilter
from .dedup import DedupStore, sha256_of_bytes, dhash_of_bytes
from .disk_guard import DiskGuard
from .distributed import NullDistributedStore, build_distributed_store
from .downloader import Downloader
from .google_vision import GoogleVisionVerifier
from .logger import StructuredLogger, get_logger
from .proxy_rotator import ProxyRotator
from .quality import check_bytes, check_url, harden_pil
from .queries import build_queries
from .query_priority import PriorityQueryScheduler
from .sources.base import Source, URLCandidate, TokenBucket
from .storage import Storage


class Coordinator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # ---- logging ----
        StructuredLogger.configure(
            level=cfg.runtime.log_level,
            file_path=cfg.runtime.log_file,
            console_format=cfg.runtime.log_format,
        )
        self.log = get_logger("coordinator")

        # ---- PIL hardening ----
        harden_pil(max_megapixels=cfg.runtime.max_image_megapixels)

        self.stop_event = asyncio.Event()
        # Per-source queues + round-robin consumer prevents head-of-line blocking
        # where one fast source (e.g. Pexels) monopolizes the download pipeline.
        # Each queue is bounded so a runaway producer can't exhaust memory.
        self.queues: dict[str, asyncio.Queue[URLCandidate]] = {}
        self._queue_maxsize = 256   # per-source cap; small so producers slow down fairly
        self._rr_lock = asyncio.Lock()
        self._rr_idx = 0            # round-robin pointer for consumers

        self.dedup = DedupStore(
            cfg.dedup.db_path,
            phash_hamming_threshold=cfg.dedup.phash_hamming_threshold,
        )
        self.storage = Storage(
            cfg.storage.images_dir, cfg.storage.metadata_jsonl,
            cfg.storage.failed_log, cfg.storage.file_extension,
        )
        self.disk_guard = DiskGuard(
            cfg.storage.images_dir,
            min_free_gb=cfg.runtime.min_free_disk_gb,
            check_interval_s=cfg.runtime.disk_check_interval_s,
        )
        self.proxy_rotator = ProxyRotator(
            proxies=cfg.proxy.proxies if cfg.proxy.enabled else None,
            failure_cool_down_s=cfg.proxy.failure_cool_down_s,
        ) if cfg.proxy.enabled else None

        self.downloader: Optional[Downloader] = None
        self.vision = (
            GoogleVisionVerifier(
                cfg.google_vision.credentials_json,
                cfg.google_vision.min_fence_confidence,
                cfg.google_vision.labels_to_accept,
                sample_rate=cfg.google_vision.sample_rate,
                batch_size=cfg.google_vision.batch_size,
                batch_timeout_ms=cfg.google_vision.batch_timeout_ms,
                use_uri_mode=cfg.google_vision.use_uri_mode,
            ) if cfg.google_vision.enabled else None
        )
        self.content_filter = ContentFilter(
            extra_block_keywords=cfg.content_filter.extra_block_keywords,
            extra_block_domains=cfg.content_filter.extra_block_domains,
        ) if cfg.content_filter.enabled else None

        # Distributed (Redis) - optional
        self.dist = build_distributed_store(
            cfg.distributed.enabled, cfg.distributed.redis_url, cfg.distributed.key_prefix,
        )

        # Dedicated executor pools — prevents Vision (network I/O) and dHash (CPU)
        # from competing for the default executor's thread pool.
        # Sizing: vision > downloaders so no download ever waits on Vision capacity.
        n_dl = max(4, cfg.runtime.download_workers)
        self.vision_executor = ThreadPoolExecutor(
            max_workers=n_dl * 2, thread_name_prefix="vision",
        )
        self.hash_executor = ThreadPoolExecutor(
            max_workers=max(4, n_dl // 2), thread_name_prefix="dhash",
        )
        # Share Vision's batch-annotate executor with the verifier
        if self.vision is not None:
            self.vision.bind_executor(self.vision_executor)

        # Priority scheduler — populated after queries built
        self.priority: Optional[PriorityQueryScheduler] = None

        # stats
        self.saved = 0
        self.skipped_dup = 0
        self.skipped_quality = 0
        self.skipped_vision = 0
        self.skipped_content = 0
        self.fetch_failed = 0
        self.per_source_saved: dict[str, int] = {}

    async def _ensure_downloader(self) -> None:
        if self.downloader is None:
            self.downloader = Downloader(
                timeout_s=self.cfg.runtime.download_timeout_s,
                max_retries=self.cfg.runtime.max_retries,
                max_bytes=self.cfg.quality.max_bytes,
                proxy_rotator=self.proxy_rotator,
                use_circuit_breakers=self.cfg.circuit_breaker.enabled,
            )

    async def close(self) -> None:
        if self.downloader is not None:
            await self.downloader.close()
        await self.dist.close()
        # Shut down the dedicated executors
        self.vision_executor.shutdown(wait=False, cancel_futures=True)
        self.hash_executor.shutdown(wait=False, cancel_futures=True)

    def _build_sources(self, queries: list[str]) -> list[Source]:
        assert self.downloader is not None
        from .sources.google_cse import GoogleCSESource
        from .sources.pexels import PexelsSource
        from .sources.unsplash import UnsplashSource
        from .sources.pixabay import PixabaySource
        from .sources.flickr import FlickrSource
        from .sources.wikimedia import WikimediaSource
        from .sources.reddit import RedditSource
        from .sources.playwright_google import PlaywrightGoogleSource
        from .sources.playwright_bing import PlaywrightBingSource
        from .sources.playwright_ddg import PlaywrightDDGSource
        from .sources.playwright_pinterest import PlaywrightPinterestSource
        from .sources.playwright_houzz import PlaywrightHouzzSource
        from .sources.company_sites import CompanySitesSource

        sources: list[Source] = []
        specs = [
            (self.cfg.google_cse,  GoogleCSESource),
            (self.cfg.pexels,      PexelsSource),
            (self.cfg.unsplash,    UnsplashSource),
            (self.cfg.pixabay,     PixabaySource),
            (self.cfg.flickr,      FlickrSource),
            (self.cfg.wikimedia,   WikimediaSource),
            (self.cfg.reddit,      RedditSource),
            (self.cfg.pw_google,    PlaywrightGoogleSource),
            (self.cfg.pw_bing,      PlaywrightBingSource),
            (self.cfg.pw_ddg,       PlaywrightDDGSource),
            (self.cfg.pw_pinterest, PlaywrightPinterestSource),
            (self.cfg.pw_houzz,     PlaywrightHouzzSource),
            (self.cfg.company_sites, CompanySitesSource),
        ]
        for subcfg, cls in specs:
            if not subcfg.enabled:
                continue
            bucket = TokenBucket(subcfg.rate_limit_per_minute)
            # Re-order queries by shortage if priority scheduler is active
            q_list = queries
            if self.priority is not None:
                q_list = self.priority.reorder_for_shortage()
            sources.append(cls(q_list, self.downloader, self.dedup, subcfg, bucket=bucket))
        return sources

    async def run(self) -> None:
        # ---- disk preflight ----
        self.disk_guard.preflight()

        await self._ensure_downloader()
        queries = build_queries(
            static=self.cfg.queries.use_static,
            custom=self.cfg.queries.custom,
            gemini_extra=self.cfg.queries.gemini_target_extra if self.cfg.queries.use_gemini_expansion else 0,
            gemini_api_key=self.cfg.queries.gemini_api_key,
        )
        self.log.info("run_start", queries=len(queries),
                      target=self.cfg.runtime.target_total_images,
                      dry_run=self.cfg.runtime.dry_run,
                      workers=self.cfg.runtime.download_workers,
                      distributed=self.dist.enabled)

        if self.cfg.query_priority.enabled:
            self.priority = PriorityQueryScheduler(
                queries, target_per_query=self.cfg.query_priority.target_per_query,
            )

        # Resume: seed counters from dedup DB so target_total_images is an ABSOLUTE
        # target (total-on-disk), not "additional this run". Progress output shows
        # cumulative saved, matching what's actually in data_scraped/images/.
        try:
            existing = self.dedup.count()
            if existing > 0:
                self.saved = existing
                for src, n in self.dedup.counts_by_source().items():
                    self.per_source_saved[src] = n
                self.log.info("resume_from_existing", saved=existing,
                              target=self.cfg.runtime.target_total_images,
                              per_source=dict(self.per_source_saved))
        except Exception as e:
            self.log.warn("resume_seed_failed", error=str(e)[:200])

        sources = self._build_sources(queries)
        if not sources:
            self.log.warn("no_sources_enabled")
            return
        self.log.info("sources_enabled", names=[s.name for s in sources])

        # Create per-source queues (before any producer starts)
        for s in sources:
            self.queues.setdefault(s.name, asyncio.Queue(maxsize=self._queue_maxsize))

        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, self.stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass

        producers = [asyncio.create_task(self._run_source(s), name=f"producer_{s.name}")
                     for s in sources]
        downloaders = [
            asyncio.create_task(self._download_worker(i), name=f"downloader_{i}")
            for i in range(self.cfg.runtime.download_workers)
        ]
        progress = asyncio.create_task(self._progress_loop(), name="progress")
        disk_task = asyncio.create_task(
            self.disk_guard.monitor(lambda: self._should_stop()), name="disk_guard",
        )

        try:
            # Wait until ALL producers finish OR target hit. Previously used
            # FIRST_COMPLETED which exited as soon as any single producer ended —
            # broken when one producer (e.g. a fully-resumed company_sites) finishes
            # in <1s while slower sources haven't yielded their first URL yet.
            while not self._should_stop():
                if all(p.done() for p in producers):
                    break
                await asyncio.sleep(1.0)
            while any(not q.empty() for q in self.queues.values()) and not self._should_stop():
                await asyncio.sleep(0.5)
        finally:
            self.stop_event.set()
            for t in producers:
                if not t.done():
                    t.cancel()
            # Poison pill into each per-source queue, one per downloader
            for q in self.queues.values():
                for _ in range(len(downloaders)):
                    try:
                        q.put_nowait(URLCandidate(url="__STOP__", source="__stop__"))
                    except asyncio.QueueFull:
                        break
            for t in downloaders:
                try:
                    await asyncio.wait_for(t, timeout=30)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    t.cancel()
            progress.cancel()
            disk_task.cancel()
            for s in sources:
                try:
                    await s.close()
                except Exception:
                    pass
            await self.close()

        self._print_final_report()

    def _should_stop(self) -> bool:
        return (self.stop_event.is_set()
                or self.saved >= self.cfg.runtime.target_total_images
                or self.disk_guard.stop_signal.is_set())

    async def _run_source(self, source: Source) -> None:
        log = get_logger(f"source.{source.name}")
        try:
            async for cand in source.discover():
                if self._should_stop():
                    return
                reject = check_url(cand.url, self.cfg.quality)
                if reject:
                    continue
                if self.content_filter is not None:
                    cfr = self.content_filter.check(cand)
                    if cfr:
                        self.skipped_content += 1
                        log.debug("content_filter_rejected", url=cand.url[:100], reason=cfr)
                        continue
                # Distributed URL dedup (only matters if Redis mode on)
                if self.dist.enabled:
                    if await self.dist.url_seen(cand.url):
                        continue
                    await self.dist.mark_url(cand.url, cand.source)
                # Push to this source's own queue — producers only block on their own lane
                q = self.queues.setdefault(source.name, asyncio.Queue(maxsize=self._queue_maxsize))
                try:
                    await asyncio.wait_for(q.put(cand), timeout=30)
                except asyncio.TimeoutError:
                    continue
            log.info("source_done", name=source.name)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("source_crashed", exc=e, source=source.name)

    async def _next_candidate(self) -> Optional[URLCandidate]:
        """Round-robin pick across per-source queues. Returns None if all are empty.
        Rotating start index (guarded by a lock) ensures worker fairness across sources.
        """
        async with self._rr_lock:
            names = list(self.queues.keys())
            if not names:
                return None
            start = self._rr_idx % len(names)
            self._rr_idx += 1
            # Walk names starting at `start`, return first non-empty queue's item.
            # First pass: skip queues whose first item's host is in active cooldown.
            for skip_cool in (True, False):
                for i in range(len(names)):
                    q = self.queues[names[(start + i) % len(names)]]
                    if q.empty():
                        continue
                    try:
                        cand = q.get_nowait()
                    except asyncio.QueueEmpty:
                        continue
                    if skip_cool and self.downloader is not None:
                        host = self.downloader._host_of(cand.url)
                        import time
                        async with self.downloader._host_cool_lock:
                            resume_at = self.downloader._host_cool.get(host, 0)
                        if resume_at - time.monotonic() > 5:
                            # host is cooling — put back and try another queue this round
                            try:
                                q.put_nowait(cand)
                            except asyncio.QueueFull:
                                pass
                            continue
                    return cand
        # All empty — wait on the FIRST queue to get something (cheap busy-wait alternative)
        # Use asyncio.wait on queue.get for all queues, return first to resolve.
        gets = [asyncio.create_task(q.get()) for q in self.queues.values()]
        try:
            done, pending = await asyncio.wait(gets, return_when=asyncio.FIRST_COMPLETED, timeout=1.0)
            for t in pending:
                t.cancel()
            if done:
                task = next(iter(done))
                return task.result()
        except asyncio.CancelledError:
            for t in gets:
                t.cancel()
            raise
        return None

    async def _download_worker(self, idx: int) -> None:
        """Round-robin consumer: on each iteration, try each source queue in turn,
        starting from a rotating offset. First non-empty queue wins."""
        log = get_logger("worker")
        while True:
            if self._should_stop():
                return
            try:
                cand = await self._next_candidate()
                if cand is None:
                    await asyncio.sleep(0.1)
                    continue
                if cand.url == "__STOP__":
                    return
                if self._should_stop():
                    return
                await self._process_candidate(cand)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # NEVER let a worker die silently — log, sleep briefly, continue.
                # Previously an uncaught exception here was silently killing all
                # 16 workers after the initial burst, causing the queues-stay-full stall.
                log.exception("worker_iteration_failed", idx=idx,
                              error=f"{type(e).__name__}: {str(e)[:200]}")
                await asyncio.sleep(0.2)

    async def _process_candidate(self, cand: URLCandidate) -> None:
        log = get_logger("downloader")
        assert self.downloader is not None

        # Per-stage timing — emit if total pipeline > 5s, so we can spot slow stages
        import time
        t_start = time.monotonic()
        stages: list[tuple[str, float]] = []

        def stage(name: str) -> None:
            stages.append((name, round(time.monotonic() - t_start, 3)))

        def report_if_slow(reason: str) -> None:
            total = round(time.monotonic() - t_start, 2)
            if total > 5.0:
                log.warn("slow_pipeline", reason=reason, total_s=total,
                         source=cand.source, url=cand.url[:100], stages=dict(stages))

        if self.cfg.runtime.dry_run:
            # Simulate success — don't actually download
            self.saved += 1
            self.per_source_saved[cand.source] = self.per_source_saved.get(cand.source, 0) + 1
            if self.priority is not None:
                self.priority.record_saved(cand.query, 1)
            log.debug("dry_run_accepted", url=cand.url[:100], source=cand.source)
            return

        try:
            image_bytes = await self.downloader.fetch_image(cand.url)
        except Exception as e:
            self.fetch_failed += 1
            self.dedup.log_failure(cand.url, cand.source, f"fetch:{e}")
            report_if_slow("fetch_exception")
            return
        stage("fetch")
        if not image_bytes:
            self.fetch_failed += 1
            self.dedup.log_failure(cand.url, cand.source, "fetch:empty_or_non_image")
            report_if_slow("fetch_empty")
            return

        sha = sha256_of_bytes(image_bytes)
        if self.dedup.exists_sha256(sha):
            self.skipped_dup += 1
            report_if_slow("sha_dup")
            return
        if self.dist.enabled and await self.dist.sha_seen(sha):
            self.skipped_dup += 1
            report_if_slow("dist_sha_dup")
            return
        stage("sha")

        qc = check_bytes(image_bytes, self.cfg.quality)
        if not qc.ok:
            self.skipped_quality += 1
            self.dedup.log_failure(cand.url, cand.source, f"quality:{qc.reason}")
            report_if_slow("quality_fail")
            return
        stage("quality")

        dhash = await asyncio.get_running_loop().run_in_executor(
            self.hash_executor, dhash_of_bytes, image_bytes,
        )
        stage("dhash")
        if dhash is not None and self.dedup.near_duplicate(dhash):
            self.skipped_dup += 1
            report_if_slow("near_dup")
            return

        if self.vision is not None and self.vision.enabled:
            # Pass the URL — verifier sends URI to Vision (no upload bandwidth used).
            # Falls back to image_bytes if URI mode disabled.
            checked, ok, label, conf, _ = await self.vision.verify_batched(
                image_bytes, image_uri=cand.url,
            )
            stage("vision")
            if checked and not ok:
                self.skipped_vision += 1
                self.dedup.log_failure(cand.url, cand.source,
                                        f"vision_rejected:best_label={label}:conf={conf:.2f}")
                report_if_slow("vision_rejected")
                return
            # Tag metadata with Vision result if we actually called the API
            if checked:
                cand.extra["vision_label"] = label
                cand.extra["vision_conf"] = conf
                cand.extra["vision_checked"] = True
            else:
                cand.extra["vision_checked"] = False

        import functools as _ft
        save_fn = _ft.partial(
            self.storage.save_image,
            image_bytes=image_bytes, source=cand.source,
            query=cand.query or "", sha256=sha,
        )
        try:
            path = await asyncio.get_running_loop().run_in_executor(
                self.hash_executor, save_fn,
            )
        except Exception as e:
            self.dedup.log_failure(cand.url, cand.source, f"save:{e}")
            report_if_slow("save_fail")
            return
        stage("save")

        inserted = self.dedup.save_image(
            sha256=sha, dhash=dhash, path=str(path), source=cand.source,
            query=cand.query, origin_url=cand.url,
            width=qc.width, height=qc.height, bytes_=len(image_bytes),
        )
        if not inserted:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
            self.skipped_dup += 1
            return

        if self.dist.enabled:
            await self.dist.mark_sha(sha)

        self.storage.append_metadata({
            "path": str(path), "source": cand.source, "query": cand.query,
            "origin_url": cand.url, "origin_page": cand.origin_page, "title": cand.title,
            "sha256": sha, "dhash": dhash,
            "width": qc.width, "height": qc.height, "bytes": len(image_bytes),
            "extra": cand.extra,
        })
        self.saved += 1
        self.per_source_saved[cand.source] = self.per_source_saved.get(cand.source, 0) + 1
        if self.priority is not None:
            self.priority.record_saved(cand.query, 1)
        report_if_slow("saved_ok")

    async def _progress_loop(self) -> None:
        t0 = time.monotonic()
        log = get_logger("progress")
        last_saved = -1
        stall_ticks = 0
        while not self._should_stop():
            await asyncio.sleep(self.cfg.runtime.progress_interval_s)
            elapsed = time.monotonic() - t0
            rate = self.saved / max(elapsed, 1)
            log.info("progress",
                     elapsed_s=int(elapsed),
                     saved=self.saved, target=self.cfg.runtime.target_total_images,
                     rate_per_s=round(rate, 2),
                     skipped_dup=self.skipped_dup,
                     skipped_quality=self.skipped_quality,
                     skipped_vision=self.skipped_vision,
                     skipped_content=self.skipped_content,
                     fetch_failed=self.fetch_failed,
                     queue_size=sum(q.qsize() for q in self.queues.values()),
                     queues_by_source={n: q.qsize() for n, q in self.queues.items()},
                     urls_seen=self.dedup.url_seen_count(),
                     by_source=dict(self.per_source_saved))
            # Stall detection: if saved hasn't advanced for 4 ticks, dump task states.
            if self.saved == last_saved:
                stall_ticks += 1
            else:
                stall_ticks = 0
                last_saved = self.saved
            if stall_ticks == 4 or (stall_ticks > 0 and stall_ticks % 8 == 0):
                try:
                    import traceback
                    tasks = [t for t in asyncio.all_tasks()
                             if t.get_name().startswith(("downloader_", "producer_"))]
                    for t in tasks:
                        stack = t.get_stack(limit=6)
                        if not stack:
                            log.warn("task_state", name=t.get_name(),
                                     done=t.done(), coro="<no stack>")
                            continue
                        frames = [f"{f.f_code.co_filename.split(chr(92))[-1]}:"
                                  f"{f.f_lineno}:{f.f_code.co_name}"
                                  for f in stack]
                        log.warn("task_state", name=t.get_name(),
                                 done=t.done(), frames=frames)
                except Exception as e:
                    log.warn("task_state_dump_failed", error=str(e))

    def _print_final_report(self) -> None:
        self.log.info("run_done",
                      saved=self.saved,
                      skipped_dup=self.skipped_dup,
                      skipped_quality=self.skipped_quality,
                      skipped_vision=self.skipped_vision,
                      skipped_content=self.skipped_content,
                      fetch_failed=self.fetch_failed,
                      per_source=dict(self.per_source_saved))
