"""Minimal logging: one logger per script, UTF-8 safe on Windows consoles."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str = "ccp", level: int = logging.INFO) -> logging.Logger:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(level)
    return log
