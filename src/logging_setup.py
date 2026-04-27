"""Centralised structlog configuration.

Used by every CLI command. Output is JSON when stdout is not a TTY (so it
pipes cleanly into log aggregators in production), human-readable otherwise.
The default level is WARNING; pass `--verbose` to drop to INFO and `--verbose
--verbose` for DEBUG.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure(verbosity: int = 0) -> Any:
    """Configure structlog and return the module logger.

    Parameters
    ----------
    verbosity : 0 = WARNING (default), 1 = INFO, 2+ = DEBUG.
    """
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    is_tty = sys.stderr.isatty()

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any = (
        structlog.dev.ConsoleRenderer(colors=True) if is_tty else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    if level > logging.DEBUG:
        for name in ("tensorflow", "matplotlib", "sklearn", "absl"):
            logging.getLogger(name).setLevel(logging.WARNING)
    return structlog.get_logger()


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
