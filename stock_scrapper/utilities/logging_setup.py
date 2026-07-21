"""Logging configuration helpers."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping


def setup_logging(config: Mapping[str, Any], run_id: str | None = None) -> logging.Logger:
    """Create console and rotating file handlers for the application."""
    logs_dir = Path(config["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stock_scrapper")
    logger.setLevel(getattr(logging, str(config.get("logging_level", "INFO")).upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_name = f"stock_scrapper_{run_id or 'run'}.log"
    file_handler = RotatingFileHandler(logs_dir / file_name, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
