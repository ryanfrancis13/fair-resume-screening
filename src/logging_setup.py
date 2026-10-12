"""Console + JSONL logging.

Human-readable lines go to stderr; a machine-readable JSONL mirror goes into the
run directory.  The JSONL copy exists so that a rate-limit storm or a batch of
parse failures can be quantified after the fact ("how many calls retried, and at
which point in the run?") instead of being scrolled past.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False


class _JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ConsoleFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG": "\033[38;5;244m",
        "INFO": "\033[38;5;39m",
        "WARNING": "\033[38;5;214m",
        "ERROR": "\033[38;5;196m",
        "CRITICAL": "\033[48;5;196;38;5;231m",
    }
    _RESET = "\033[0m"

    def __init__(self, use_color: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self.use_color:
            return text
        color = self._COLORS.get(record.levelname, "")
        return f"{color}{text}{self._RESET}" if color else text


def setup_logging(level: str = "INFO", jsonl_path: Optional[Path] = None) -> None:
    """Idempotent root-logger configuration."""
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    if not _CONFIGURED:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(_ConsoleFormatter(use_color=sys.stderr.isatty()))
        root.addHandler(console)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("google").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        _CONFIGURED = True

    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        for handler in list(root.handlers):
            if getattr(handler, "_frs_jsonl", False):
                root.removeHandler(handler)
        file_handler = logging.FileHandler(jsonl_path, encoding="utf-8")
        file_handler.setFormatter(_JsonlFormatter())
        file_handler._frs_jsonl = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


#: Positional parameter names of :func:`log_event`. A structured field with one
#: of these names would collide with the parameter itself and raise a TypeError
#: at the call site, so such fields are renamed rather than allowed to break a
#: long-running pipeline stage on a logging call.
_RESERVED_FIELD_NAMES = ("logger", "level", "msg")


def log_event(logger: logging.Logger, level: int, msg: str, **fields: object) -> None:
    """Log with structured fields that survive into the JSONL mirror."""
    safe = {
        (f"field_{key}" if key in _RESERVED_FIELD_NAMES else key): value
        for key, value in fields.items()
    }
    logger.log(level, msg, extra={"extra_fields": safe})
