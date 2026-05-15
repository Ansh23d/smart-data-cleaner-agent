"""Structured logging for the AI Data Agent.

Sets up a rotating file handler and structlog processors.
Every module should call get_logger(__name__) to obtain a logger.

Usage
-----
from utils.logger import get_logger
log = get_logger(__name__)
log.info("kpi_computed", kpi_name="Revenue", value=1234.5, session_id=sid)
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import structlog

_project_root = Path(__file__).resolve().parents[1]
_LOGS_DIR = _project_root / "data" / "logs"

_setup_done = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structlog + stdlib logging.  Idempotent — safe to call many times."""
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── structlog processors ────────────────────────────────────────────────
    shared_processors: list[Any] = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    # ── stdlib root logger ──────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        # Rotating file: 10 MB × 3 backups
        fh = logging.handlers.RotatingFileHandler(
            _LOGS_DIR / "agent.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

        # Console: errors only (avoid polluting Streamlit terminal output)
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.ERROR)
        ch.setFormatter(formatter)
        root.addHandler(ch)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to *name*.  Calls setup_logging() if needed."""
    setup_logging()
    return structlog.get_logger(name)
