"""Google Cloud Vision API — optional verification gate.

Used AFTER download (not for URL discovery) to confirm the image actually contains
a fence. Cheap insurance against off-topic results from image search engines.

COST: ~$1.50 per 1000 API calls at list price. Use `sample_rate` < 1.0 to verify
only a random fraction of images — cost scales linearly:
  sample_rate=1.0  → $1.50/1000 images scraped
  sample_rate=0.2  → $0.30/1000 images scraped (most images bypass Vision)
  sample_rate=0.05 → $0.075/1000 images scraped (spot-check only)

Images that bypass the check are accepted unconditionally — this is a
quality-gate sampler, not a classifier. If you want 100% coverage, set 1.0.

Returns (from verify_with_sampling):
  (checked: bool, accepted: bool, best_label: str|None, confidence: float, labels)
"""
from __future__ import annotations

import asyncio
import hashlib
import random
from concurrent.futures import Executor
from pathlib import Path
from typing import Optional

from .logger import get_logger

_log = get_logger("vision")

_VisionResult = tuple[bool, bool, Optional[str], float, list[tuple[str, float]]]
# (checked, accepted, label, confidence, labels)


class GoogleVisionVerifier:
    def __init__(self, credentials_json: Optional[str], min_confidence: float,
                 labels_to_accept: list[str], sample_rate: float = 1.0,
                 batch_size: int = 10, batch_timeout_ms: int = 500,
                 use_uri_mode: bool = True):
        self.enabled = False
        self.client = None
        self.min_confidence = float(min_confidence)
        self.sample_rate = max(0.0, min(1.0, float(sample_rate)))
        self.labels_to_accept = set(l.lower() for l in labels_to_accept)
        self._rng = random.Random()

        # --- batch mode state ---
        self.batch_size = max(1, int(batch_size))
        self.batch_timeout_s = max(0.05, batch_timeout_ms / 1000.0)
        self.fut_watchdog_s = 60.0   # max a downloader can wait for Vision; then permissive
        # When True, send URL to Vision instead of uploading bytes — Google fetches
        # from inside their datacenter (gigabit). 10000× less local upload bandwidth.
        # Falls back to bytes if no URL is available.
        self.use_uri_mode = bool(use_uri_mode)
        # Buffer items are (image_bytes, image_uri, future) — bytes is a fallback
        self._batch: list[tuple[Optional[bytes], Optional[str], asyncio.Future]] = []
        self._batch_lock: Optional[asyncio.Lock] = None
        self._flush_task: Optional[asyncio.Task] = None
        # Strong references to spawned flush tasks so Python GC doesn't kill them
        self._flush_tasks_refs: set = set()
        self._executor: Optional[Executor] = None

        try:
            from google.cloud import vision   # type: ignore
            import os
            if credentials_json:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_json)
            self.client = vision.ImageAnnotatorClient()
            self._vision_mod = vision
            self.enabled = True
        except ImportError:
            _log.warn("vision_lib_missing")
        except Exception as e:
            _log.warn("vision_init_failed", error=str(e))

    def should_sample(self, image_bytes: bytes) -> bool:
        """Decide whether to call the Vision API for this image.

        Deterministic per-image via SHA-based hash so re-runs don't re-bill.
        Returns True if the image should be sent to Vision.
        """
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        # Use first 8 bytes of SHA to derive a deterministic 0..1 float
        h = int.from_bytes(hashlib.sha256(image_bytes).digest()[:8], "big")
        threshold = int(self.sample_rate * (1 << 64))
        return h < threshold

    def verify_with_sampling(
        self, image_bytes: bytes,
    ) -> tuple[bool, bool, Optional[str], float, list[tuple[str, float]]]:
        """Sample-aware verify. Returns (checked, accepted, label, conf, labels).

        - checked=False, accepted=True → skipped by sampler, auto-accepted
        - checked=True, accepted=?    → API called, gate applied normally
        """
        if not self.enabled or self.client is None:
            return False, True, None, 1.0, []
        if not self.should_sample(image_bytes):
            return False, True, None, 1.0, []
        accepted, label, conf, labels = self.verify(image_bytes)
        return True, accepted, label, conf, labels

    def verify(self, image_bytes: bytes) -> tuple[bool, Optional[str], float, list[tuple[str, float]]]:
        if not self.enabled or self.client is None:
            return True, None, 1.0, []
        try:
            image = self._vision_mod.Image(content=image_bytes)
            features = [
                {"type_": self._vision_mod.Feature.Type.LABEL_DETECTION, "max_results": 15},
                {"type_": self._vision_mod.Feature.Type.OBJECT_LOCALIZATION, "max_results": 15},
            ]
            response = self.client.annotate_image({"image": image, "features": features})
        except Exception as e:
            # API failure — be permissive (don't lose images due to transient errors)
            return True, None, 0.0, []

        labels: list[tuple[str, float]] = []
        for ann in response.label_annotations:
            labels.append((ann.description, float(ann.score)))
        for obj in response.localized_object_annotations:
            labels.append((obj.name, float(obj.score)))

        # Find best fence-related label
        best: tuple[str, float] = ("", 0.0)
        for name, score in labels:
            if name.lower() in self.labels_to_accept and score > best[1]:
                best = (name, score)

        accepted = best[1] >= self.min_confidence
        return accepted, best[0] or None, best[1], labels

    # ============================================================
    # BATCHED ASYNC API — 10× fewer HTTP round-trips to Vision
    # ============================================================

    def bind_executor(self, executor: Executor) -> None:
        """Coordinator calls this once to share its vision_executor pool."""
        self._executor = executor

    async def verify_batched(self, image_bytes: bytes,
                              image_uri: Optional[str] = None) -> _VisionResult:
        """Async batched verification. Buffers requests; flushes on batch-full
        OR batch_timeout. Returns (checked, accepted, label, conf, labels).

        When `image_uri` is provided AND `use_uri_mode=True`, Vision will fetch
        the image from the URI server-side (no upload). Otherwise sends the bytes.
        """
        # Fast paths: disabled or below sample rate → bypass the API entirely
        if not self.enabled or self.client is None:
            return (False, True, None, 1.0, [])
        if not self.should_sample(image_bytes):
            return (False, True, None, 1.0, [])

        # Lazy lock (event loop required)
        if self._batch_lock is None:
            self._batch_lock = asyncio.Lock()

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        # Decide payload: prefer URI when allowed (no upload bandwidth used)
        send_bytes: Optional[bytes]
        send_uri: Optional[str]
        if self.use_uri_mode and image_uri and image_uri.startswith(("http://", "https://")):
            send_bytes, send_uri = None, image_uri
        else:
            send_bytes, send_uri = image_bytes, None

        # The lock is held ONLY to mutate the batch list. We copy-out if the batch
        # is full and do the API call AFTER releasing the lock, so other coroutines
        # can keep adding to a fresh batch while Vision is in flight.
        batch_to_flush: list[tuple[Optional[bytes], Optional[str], asyncio.Future]] = []
        async with self._batch_lock:
            self._batch.append((send_bytes, send_uri, fut))
            if len(self._batch) >= self.batch_size:
                batch_to_flush = self._batch
                self._batch = []
            elif self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._timeout_flush())

        if batch_to_flush:
            # Run the API call WITHOUT holding the batch lock.
            # Keep a strong ref so Python GC doesn't kill the task; auto-cleanup on done.
            t = asyncio.create_task(self._do_flush(batch_to_flush))
            self._flush_tasks_refs.add(t)
            t.add_done_callback(self._on_flush_task_done)

        # Watchdog: if Vision never responds, resolve permissively so pipeline never hangs
        try:
            return await asyncio.wait_for(fut, timeout=self.fut_watchdog_s)
        except asyncio.TimeoutError:
            _log.error("vision_future_watchdog_fired", timeout_s=self.fut_watchdog_s)
            return (True, True, None, 0.0, [])

    def _on_flush_task_done(self, t: asyncio.Task) -> None:
        """Called when a flush task finishes — logs any unhandled exception and
        releases the strong reference so it can be garbage-collected."""
        self._flush_tasks_refs.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            _log.error("vision_flush_task_raised",
                       error=f"{type(exc).__name__}: {exc}")

    async def _timeout_flush(self) -> None:
        """Waits batch_timeout_s then flushes whatever's buffered."""
        try:
            await asyncio.sleep(self.batch_timeout_s)
        except asyncio.CancelledError:
            return
        if self._batch_lock is None:
            return
        batch_to_flush: list[tuple[Optional[bytes], Optional[str], asyncio.Future]] = []
        async with self._batch_lock:
            if self._batch:
                batch_to_flush = self._batch
                self._batch = []
        if batch_to_flush:
            try:
                await self._do_flush(batch_to_flush)
            except Exception as e:
                _log.error("vision_timeout_flush_raised",
                           error=f"{type(e).__name__}: {e}")
                # Resolve pending futures permissively so nothing hangs
                for _, _, fut in batch_to_flush:
                    if not fut.done():
                        fut.set_result((True, True, None, 0.0, []))

    async def _do_flush(self,
                         batch: list[tuple[Optional[bytes], Optional[str], asyncio.Future]],
                         ) -> None:
        """Execute the batch API call outside any lock. Fanout results to futures.
        Guaranteed to resolve EVERY future in `batch` — even on partial API response,
        API exception, or unexpected error. Never leave a downloader hanging."""
        if not batch:
            return
        permissive = (True, True, None, 0.0, [])
        try:
            items = [(b, u) for b, u, _ in batch]
            loop = asyncio.get_running_loop()
            try:
                results = await loop.run_in_executor(
                    self._executor, self._batch_annotate, items,
                )
            except Exception as e:
                _log.warn("vision_batch_failed", error=str(e)[:120], batch_size=len(batch))
                results = [permissive] * len(batch)
            # Pad / truncate results so every future gets a response
            if len(results) < len(batch):
                _log.warn("vision_batch_short_response",
                          expected=len(batch), got=len(results))
                results = list(results) + [permissive] * (len(batch) - len(results))
            for (_, _, fut), res in zip(batch, results):
                if not fut.done():
                    fut.set_result(res)
        except Exception as e:
            _log.error("vision_flush_unexpected", error=str(e)[:200],
                       batch_size=len(batch))
            for _, _, fut in batch:
                if not fut.done():
                    fut.set_result(permissive)

    def _batch_annotate(self,
                        items: list[tuple[Optional[bytes], Optional[str]]],
                        ) -> list[_VisionResult]:
        """Synchronous batch Vision call — runs in executor.
        items: list of (bytes_or_None, uri_or_None) — exactly one of each pair is set.
        Returns list of _VisionResult, one per input."""
        if not self.enabled or self.client is None:
            return [(False, True, None, 1.0, []) for _ in items]
        features = [
            {"type_": self._vision_mod.Feature.Type.LABEL_DETECTION, "max_results": 15},
            {"type_": self._vision_mod.Feature.Type.OBJECT_LOCALIZATION, "max_results": 15},
        ]
        requests = []
        for b, u in items:
            if u:
                # URI mode — Google fetches the image. Tiny request payload (just the URL).
                img = self._vision_mod.Image(source={"image_uri": u})
            else:
                img = self._vision_mod.Image(content=b)
            requests.append({"image": img, "features": features})
        try:
            response = self.client.batch_annotate_images(requests=requests)
        except Exception as e:
            _log.warn("vision_batch_api_error", error=str(e)[:120])
            # Permissive on API error — don't lose images due to transient failures
            return [(True, True, None, 0.0, []) for _ in images]

        out: list[_VisionResult] = []
        for r in response.responses:
            if getattr(r, "error", None) and r.error.message:
                out.append((True, True, None, 0.0, []))
                continue
            labels: list[tuple[str, float]] = []
            for ann in r.label_annotations:
                labels.append((ann.description, float(ann.score)))
            for obj in r.localized_object_annotations:
                labels.append((obj.name, float(obj.score)))
            best: tuple[str, float] = ("", 0.0)
            for name, score in labels:
                if name.lower() in self.labels_to_accept and score > best[1]:
                    best = (name, score)
            accepted = best[1] >= self.min_confidence
            out.append((True, accepted, best[0] or None, best[1], labels))
        return out
