"""Centralized structured logging configuration for ControlPlane.ai.

Usage:
    from backend.shared.logging_config import configure_logging, request_id_var, trace_id_var
    configure_logging(level="INFO", json_logs=False)

When CONTROLPLANE_JSON_LOGS=true (default in containers), every log record
is emitted as a single JSON object with fields:
    time, level, logger, message, request_id, trace_id, + any extras.
"""

import json
import logging
import sys
from contextvars import ContextVar
from typing import Optional

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single compact JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get("-"),
            "trace_id": trace_id_var.get("-"),
        }
        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)

        _SKIP = frozenset({
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "id", "levelname", "levelno", "lineno", "module",
            "msecs", "message", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "thread", "threadName",
        })
        for key, val in record.__dict__.items():
            if key not in _SKIP:
                try:
                    json.dumps(val)
                    log_obj[key] = val
                except (TypeError, ValueError):
                    log_obj[key] = str(val)

        try:
            return json.dumps(log_obj, ensure_ascii=False)
        except Exception:
            return f'{{"level":"{record.levelname}","message":"{record.getMessage()}"}}'


class _PlainFormatter(logging.Formatter):
    """Plain text formatter injecting request_id from contextvars."""

    def format(self, record: logging.LogRecord) -> str:
        record.request_id = request_id_var.get("-")
        record.trace_id = trace_id_var.get("-")
        return super().format(record)


class _ContextFilter(logging.Filter):
    """Inject request_id and trace_id from context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")
        record.trace_id = trace_id_var.get("-")
        return True


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Install the root logging configuration."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())

    if json_logs:
        handler.setFormatter(_JSONFormatter())
    else:
        plain_format = "%(asctime)s %(name)s %(levelname)s [%(request_id)s] %(message)s"
        handler.setFormatter(_PlainFormatter(fmt=plain_format, datefmt="%Y-%m-%dT%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()
    root.addHandler(handler)

    for noisy in ("uvicorn.access", "httpx", "httpcore", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = []
    uvicorn_logger.propagate = True
