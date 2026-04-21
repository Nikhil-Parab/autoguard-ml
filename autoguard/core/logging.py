"""Structured logging: Rich console + JSON file output."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_CONSOLE = Console(stderr=True)
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
