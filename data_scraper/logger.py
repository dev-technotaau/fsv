"""Structured JSON logger with log levels, correlation IDs, and rich console output.

Usage:
    from .logger import get_logger
    log = get_logger("coordinator")
    log.info("started", saved=0, target=12000)
    log.warn("rate_limited", source="pexels", retry_after=30)
    log.error("fetch_failed", url=url, error=str(e))

Configuration via env (overrides defaults; YAML `runtime.log_*` is the canonical source):
    SCRAPER_LOG_LEVEL=DEBUG|INFO|WARN|ERROR
    SCRAPER_LOG_FORMAT=json|plain          (console format; default: plain)

Log file path comes from YAML `runtime.log_file` only (no env var).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

try:
    from rich.console import Console
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


_console = Console(stderr=True, highlight=False) if _HAS_RICH else None
_file_handle = None
_file_lock = threading.Lock()
_context_id: threading.local = threading.local()


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Attach a correlation ID to this thread. Propagates into all subsequent logs."""
    _context_id.cid = cid or uuid.uuid4().hex[:12]
    return _context_id.cid


def get_correlation_id() -> Optional[str]:
    return getattr(_context_id, "cid", None)


def _level_int(name: str) -> int:
    return {"DEBUG": 10, "INFO": 20, "WARN": 30, "WARNING": 30, "ERROR": 40}.get(name.upper(), 20)


class StructuredLogger:
    """Logger that emits JSON for files and rich-styled text for console.

    Callable interface: log.info(message, **fields) — fields merge into the record.
    """
    _registry: dict[str, "StructuredLogger"] = {}
    _min_level: int = _level_int(os.environ.get("SCRAPER_LOG_LEVEL", "INFO"))
    _console_fmt: str = os.environ.get("SCRAPER_LOG_FORMAT", "plain").lower()
    _file_path: Optional[Path] = None

    def __init__(self, name: str):
        self.name = name

    @classmethod
    def configure(cls, *, level: Optional[str] = None, file_path: Optional[Path] = None,
                  console_format: Optional[str] = None) -> None:
        """Reconfigure global settings. Safe to call multiple times."""
        global _file_handle
        if level is not None:
            cls._min_level = _level_int(level)
        if console_format is not None:
            cls._console_fmt = console_format.lower()
        if file_path is not None:
            cls._file_path = Path(file_path)
            cls._file_path.parent.mkdir(parents=True, exist_ok=True)
            with _file_lock:
                if _file_handle is not None:
                    try:
                        _file_handle.close()
                    except Exception:
                        pass
                _file_handle = cls._file_path.open("a", encoding="utf-8", buffering=1)

    @classmethod
    def get(cls, name: str) -> "StructuredLogger":
        if name not in cls._registry:
            cls._registry[name] = cls(name)
        return cls._registry[name]

    # ---------- API ----------
    def debug(self, message: str, **fields: Any) -> None:
        self._emit(10, "DEBUG", message, fields)

    def info(self, message: str, **fields: Any) -> None:
        self._emit(20, "INFO", message, fields)

    def warn(self, message: str, **fields: Any) -> None:
        self._emit(30, "WARN", message, fields)

    # alias
    warning = warn

    def error(self, message: str, **fields: Any) -> None:
        self._emit(40, "ERROR", message, fields)

    def exception(self, message: str, exc: Optional[BaseException] = None, **fields: Any) -> None:
        import traceback
        if exc is None:
            exc_str = traceback.format_exc()
        else:
            exc_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._emit(40, "ERROR", message, {**fields, "exception": exc_str})

    # ---------- emit ----------
    def _emit(self, level_int: int, level_name: str, message: str, fields: dict[str, Any]) -> None:
        if level_int < self._min_level:
            return
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "level": level_name,
            "logger": self.name,
            "msg": message,
        }
        cid = get_correlation_id()
        if cid:
            record["cid"] = cid
        record.update(fields)

        # --- console ---
        self._emit_console(level_name, record)

        # --- file (JSON Lines) ---
        global _file_handle
        if _file_handle is not None:
            try:
                with _file_lock:
                    _file_handle.write(json.dumps(record, default=str) + "\n")
            except Exception:
                pass

    def _emit_console(self, level_name: str, record: dict[str, Any]) -> None:
        if self._console_fmt == "json":
            line = json.dumps(record, default=str)
            if _console is not None:
                _console.print(line, markup=False, highlight=False)
            else:
                print(line, file=sys.stderr)
            return

        # Plain / rich text format
        level_colors = {"DEBUG": "dim", "INFO": "cyan", "WARN": "yellow", "ERROR": "red"}
        extra_fields = {k: v for k, v in record.items()
                        if k not in ("ts", "level", "logger", "msg", "cid")}
        extras = " ".join(f"{k}={_compact(v)}" for k, v in extra_fields.items())
        cid = record.get("cid", "")
        cid_str = f"[{cid}] " if cid else ""
        prefix = f"{record['ts']} {record['level']:<5} {record['logger']:<14} {cid_str}"
        line = f"{prefix}{record['msg']}"
        if extras:
            line += f"  {extras}"

        if _console is not None:
            color = level_colors.get(level_name, "white")
            _console.print(f"[{color}]{line}[/{color}]", markup=True, highlight=False)
        else:
            print(line, file=sys.stderr)


def _compact(v: Any) -> str:
    if isinstance(v, str):
        s = v if len(v) < 80 else v[:77] + "..."
        return f'"{s}"' if " " in s else s
    if isinstance(v, (int, float, bool)) or v is None:
        return str(v)
    return json.dumps(v, default=str)[:120]


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger.get(name)


# ---------- bridge stdlib logging → structured logger ----------

class _StructuredHandler(logging.Handler):
    """Routes stdlib logging records through our structured logger."""
    def emit(self, record: logging.LogRecord) -> None:
        log = get_logger(record.name or "stdlib")
        fn = {logging.DEBUG: log.debug, logging.INFO: log.info,
              logging.WARNING: log.warn, logging.ERROR: log.error,
              logging.CRITICAL: log.error}.get(record.levelno, log.info)
        fn(record.getMessage())


def bridge_stdlib_logging() -> None:
    root = logging.getLogger()
    root.handlers = [_StructuredHandler()]
    root.setLevel(logging.DEBUG)
