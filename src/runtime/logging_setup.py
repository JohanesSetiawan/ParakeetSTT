"""
Dated run-log configuration.

The terminal shows short progress lines; the log file keeps the detail needed
to reconstruct a run afterwards: run id, settings, device, per-stage timings,
and full exception tracebacks. The file name follows the project convention
``logs/log_<YYYY-MM-DD>.txt`` and appends when several runs share a day, so
every line carries the run id to keep runs apart.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s run=%(run_id)s %(name)s: %(message)s"
PACKAGE_LOGGER_NAME = "src"


class _RunIdFilter(logging.Filter):
    """Stamp every record with the current run id."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def log_file_path(log_dir: Path, now: datetime | None = None) -> Path:
    """Return ``<log_dir>/log_<YYYY-MM-DD>.txt`` for the given local time."""

    timestamp = now or datetime.now()
    return log_dir / f"log_{timestamp:%Y-%m-%d}.txt"


def configure_run_logging(log_dir: Path, level: str) -> tuple[str, Path]:
    """
    Attach a dated file handler to the package logger.

    Args:
        log_dir: Directory for log files; created when missing.
        level: Logging level name validated by settings.

    Returns:
        The new run id and the log file path.

    Side effects:
        Replaces handlers previously installed by this function, so calling it
        twice in one process does not duplicate every line.
    """

    run_id = uuid.uuid4().hex[:12]
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_file_path(log_dir)

    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(_RunIdFilter(run_id))

    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    for existing in list(package_logger.handlers):
        if getattr(existing, "_run_log_handler", False):
            package_logger.removeHandler(existing)
            existing.close()
    handler._run_log_handler = True  # type: ignore[attr-defined]
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    # Terminal output is owned by the command's reporter, not by logging.
    package_logger.propagate = False

    return run_id, path
