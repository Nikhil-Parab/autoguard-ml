"""Structured logging: Rich console + JSON file output."""
from __future__ import annotations

import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


# ── Single shared UTF-8 wrappers (created once, never GC'd) ──────────────────
# Creating multiple TextIOWrapper instances over the same buffer causes the GC
# to close the buffer when any wrapper is collected → "I/O operation on closed
# file" crashes.  We cache one wrapper per stream and reuse it everywhere.
_UTF8_STDOUT: Optional[io.TextIOWrapper] = None
_UTF8_STDERR: Optional[io.TextIOWrapper] = None


def _get_utf8_stream(stderr: bool = False) -> io.TextIOWrapper:
    """Return (and cache) a UTF-8 TextIOWrapper over stdout or stderr."""
    global _UTF8_STDOUT, _UTF8_STDERR
    stream = sys.stderr if stderr else sys.stdout
    cached = _UTF8_STDERR if stderr else _UTF8_STDOUT
    if cached is None or cached.closed:
        try:
            wrapper = io.TextIOWrapper(
                stream.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            if stderr:
                _UTF8_STDERR = wrapper
            else:
                _UTF8_STDOUT = wrapper
            return wrapper
        except AttributeError:
            # No .buffer (e.g. pytest capture) — return original stream
            return stream  # type: ignore[return-value]
    return cached


# Module-level cached Console instances (shared across the whole process)
_STDOUT_CONSOLE: Optional[Console] = None
_STDERR_CONSOLE: Optional[Console] = None


def _make_utf8_console(stderr: bool = False) -> Console:
    """
    Return a shared Rich Console backed by a single UTF-8 TextIOWrapper.

    Re-using the same Console (and wrapper) avoids the GC closing the shared
    buffer when a locally-created wrapper goes out of scope, which was causing
    "I/O operation on closed file" errors on Windows.
    """
    global _STDOUT_CONSOLE, _STDERR_CONSOLE
    cached = _STDERR_CONSOLE if stderr else _STDOUT_CONSOLE
    if cached is not None:
        return cached
    try:
        utf8_stream = _get_utf8_stream(stderr)
        c = Console(file=utf8_stream, stderr=stderr, highlight=False)
    except Exception:
        c = Console(stderr=stderr, highlight=False)
    if stderr:
        _STDERR_CONSOLE = c
    else:
        _STDOUT_CONSOLE = c
    return c


_CONSOLE = _make_utf8_console(stderr=True)
_LOGGER_NAME = "autoguard"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(
    verbose: bool = True,
    log_file: Optional[str | Path] = None,
) -> None:
    logger = get_logger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    ch = RichHandler(console=_CONSOLE, show_time=True, show_path=False,
                     rich_tracebacks=True, markup=True)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(ch)

    if log_file:
        fh = _JsonLineHandler(Path(log_file))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    logger.propagate = False


class _JsonLineHandler(logging.FileHandler):
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(path, mode="a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
                "level": record.levelname,
                "msg": record.getMessage(),
                "module": record.module,
            }
            self.stream.write(json.dumps(payload) + "\n")
            self.flush()
        except Exception:
            self.handleError(record)
