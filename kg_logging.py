#!/usr/bin/env python3
"""
Structured JSON-lines logging for extraction runs.

Usage in extract.py / apply_graph.py:
    from kg_logging import get_run_logger, NULL_LOGGER
    _log = get_run_logger(output_dir)          # writes output_dir/run.jsonl
    _log.info("doc_start", extra={"doc": "report.md", "stage": "start"})
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# ── Null logger (no-op sink used when log_dir is unavailable) ─────────────────
NULL_LOGGER: logging.Logger = logging.getLogger("kg._null")
NULL_LOGGER.addHandler(logging.NullHandler())
NULL_LOGGER.propagate = False

# Known extra fields forwarded to JSON output
_EXTRA_KEYS = (
    "doc", "stage", "chunks_total",
    "counts", "dropped", "total",
    "found",
    "triple_count", "color_dist",
    "run_path",
)


class _JsonlHandler(logging.Handler):
    """Appends one JSON object per log record to a .jsonl file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        for key in _EXTRA_KEYS:
            val = record.__dict__.get(key)
            if val is not None:
                entry[key] = val
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            self.handleError(record)


# ── Factory — one logger per log file path ────────────────────────────────────

_LOGGERS: dict[str, logging.Logger] = {}


def get_run_logger(log_dir: Path) -> logging.Logger:
    """Return a logger that appends structured JSON lines to log_dir/run.jsonl.

    Idempotent: calling twice with the same path returns the same logger.
    """
    log_path = log_dir / "run.jsonl"
    key = str(log_path.resolve())
    if key in _LOGGERS:
        return _LOGGERS[key]
    logger = logging.getLogger(f"kg.run.{key}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(_JsonlHandler(log_path))
    _LOGGERS[key] = logger
    return logger
