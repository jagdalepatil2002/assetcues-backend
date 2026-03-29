"""
Console logging for local / Uvicorn: structured fields + readable terminal lines.

Set LOG_LEVEL=DEBUG in the environment for more detail (default INFO).
"""
from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_terminal_logging() -> None:
    """Call once at process startup (before other Assetcues Invoice AI imports use loggers)."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False)
    shared_pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_pre_chain,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_pre_chain,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("httpx", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, level))
