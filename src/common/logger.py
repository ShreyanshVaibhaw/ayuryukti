"""Structured logging setup with rich console and file output."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


def setup_logger(name: str) -> logging.Logger:
    """Configure a logger with console + file handlers."""
    log_level = os.getenv("AYURYUKTI_LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    log_dir = Path("outputs") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "ayuryukti.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def get_performance_logger() -> logging.Logger:
    """Dedicated performance logger writing to outputs/logs/performance.log."""
    logger = logging.getLogger("AyurYuktiPerformance")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_dir = Path("outputs") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "performance.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    return logger


@contextmanager
def log_timing(logger: logging.Logger, operation: str, perf_logger: Optional[logging.Logger] = None):
    """Context manager to log operation duration."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        msg = f"{operation} completed in {duration:.3f}s"
        logger.info(msg)
        if perf_logger is not None:
            perf_logger.info(msg)
