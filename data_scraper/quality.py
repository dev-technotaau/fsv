"""Quality filter — resolution, aspect ratio, file size, blocked domains.

Also hardens PIL against decompression bombs:
  - PIL.Image.MAX_IMAGE_PIXELS caps decoded pixel count
  - LOAD_TRUNCATED_IMAGES = False rejects partial bytes
  - Warnings are converted to rejections (Image.DecompressionBombWarning)
"""
from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from .config import QualityConfig


# ---------- PIL hardening ----------
# Must be set BEFORE any PIL.Image.open / verify call in the process.
def harden_pil(max_megapixels: int = 100) -> None:
    """Call once at startup. Caps PIL-decoded pixel count; rejects truncated files."""
    try:
        from PIL import Image, ImageFile
        Image.MAX_IMAGE_PIXELS = max_megapixels * 1_000_000
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        # Elevate decompression bomb warning → exception
        warnings.simplefilter("error", Image.DecompressionBombWarning)
    except ImportError:
        pass


@dataclass
class QualityCheck:
    ok: bool
    reason: Optional[str] = None
    width: int = 0
    height: int = 0
    format: Optional[str] = None


def check_url(url: str, cfg: QualityConfig) -> Optional[str]:
    """Pre-download URL filter. Returns None if OK, or rejection reason."""
    try:
        parsed = urlparse(url)
    except Exception:
        return "unparseable_url"
    host = (parsed.netloc or "").lower()
    for dom in cfg.blocked_domains:
        if dom in host:
            return f"blocked_domain:{dom}"
    path_lower = (parsed.path or "").lower()
    if any(path_lower.endswith(ext) for ext in (".svg", ".gif", ".bmp", ".ico", ".tif", ".tiff")):
        return "bad_extension"
    return None


def check_bytes(b: bytes, cfg: QualityConfig) -> QualityCheck:
    """Post-download check. Decodes header to validate dims + format."""
    if len(b) < cfg.min_bytes:
        return QualityCheck(ok=False, reason="too_small_bytes")
    if len(b) > cfg.max_bytes:
        return QualityCheck(ok=False, reason="too_large_bytes")
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(b))
        try:
            img.verify()                                    # raises if corrupt / truncated / bomb
        except Image.DecompressionBombWarning:
            return QualityCheck(ok=False, reason="decompression_bomb")
        img = Image.open(io.BytesIO(b))                     # re-open after verify consumption
        w, h = img.size
        fmt = (img.format or "").lower()
    except Image.DecompressionBombWarning:
        return QualityCheck(ok=False, reason="decompression_bomb")
    except Image.DecompressionBombError:
        return QualityCheck(ok=False, reason="decompression_bomb")
    except Exception as e:
        return QualityCheck(ok=False, reason=f"bad_image:{type(e).__name__}")
    if w < cfg.min_width or h < cfg.min_height:
        return QualityCheck(ok=False, reason=f"too_small_res:{w}x{h}", width=w, height=h, format=fmt)
    if w > cfg.max_width or h > cfg.max_height:
        return QualityCheck(ok=False, reason=f"too_large_res:{w}x{h}", width=w, height=h, format=fmt)
    aspect = w / max(1, h)
    if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
        return QualityCheck(ok=False, reason=f"bad_aspect:{aspect:.2f}", width=w, height=h, format=fmt)
    return QualityCheck(ok=True, width=w, height=h, format=fmt)
