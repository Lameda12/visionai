"""Centralised logging configuration for VisionAI."""
from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DATE = "%H:%M:%S"

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
    root = logging.getLogger("visionai")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'visionai' namespace."""
    _configure()
    # Prefix non-visionai names so all project logs share the root handler
    if not name.startswith("visionai"):
        name = f"visionai.{name}"
    return logging.getLogger(name)
