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


def _make_utf8_console(stderr: bool = False) -> Console:
    """
    Create a Rich Console backed by a UTF-8 TextIOWrapper.

    This bypasses the Windows legacy console renderer (cp1252) so that
    all Unicode characters (arrows, box-drawing, emoji, etc.) are written
    correctly on any platform.
    """
    stream = sys.stderr if stderr else sys.stdout
    try:
        utf8_stream = io.TextIOWrapper(
            stream.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
        return Console(file=utf8_stream, stderr=stderr, highlight=False)
    except AttributeError:
        # No .buffer attribute (e.g. pytest capture) — fall back gracefully
        return Console(stderr=stderr, highlight=False)


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
