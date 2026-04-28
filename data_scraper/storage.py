"""Image save + metadata JSONL append. Filenames include source + sha8 for traceability."""
from __future__ import annotations

import io
import json
import re
import threading
from pathlib import Path
from typing import Any


_safe_re = re.compile(r"[^a-z0-9_-]+")


def _slugify(s: str, maxlen: int = 40) -> str:
    s = (s or "unknown").lower()
    s = _safe_re.sub("_", s).strip("_")
    return s[:maxlen] or "x"


class Storage:
    """Writes images to disk + appends metadata lines to JSONL."""

    def __init__(self, images_dir: Path, metadata_jsonl: Path, failed_log: Path,
                 file_extension: str = "jpg"):
        self.images_dir = images_dir
        self.metadata_jsonl = metadata_jsonl
        self.failed_log = failed_log
        self.ext = file_extension.lstrip(".")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save_image(self, *, image_bytes: bytes, source: str, query: str, sha256: str) -> Path:
        """Save bytes to disk (re-encoded to target format). Returns the saved path."""
        safe_q = _slugify(query or "noquery")
        safe_src = _slugify(source)
        filename = f"{safe_src}__{safe_q}__{sha256[:8]}.{self.ext}"
        path = self.images_dir / filename
        # Re-encode to ensure consistent format and strip EXIF
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        save_kwargs: dict[str, Any] = {}
        if self.ext in ("jpg", "jpeg"):
            save_kwargs.update(quality=92, optimize=True, progressive=True)
        img.save(path, **save_kwargs)
        return path

    def append_metadata(self, record: dict) -> None:
        with self._lock, self.metadata_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def append_failure(self, record: dict) -> None:
        self.failed_log.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.failed_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
