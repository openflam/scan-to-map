"""Logging helpers for the segment3d project."""

from __future__ import annotations

import logging

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a console logger configured for INFO-level output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


__all__ = ["get_logger"]
