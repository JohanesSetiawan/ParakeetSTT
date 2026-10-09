"""
Plain-text terminal progress for long-running commands.

Reporting consumes real runtime state (completed/total units and wall-clock
time) and never duplicates or estimates what the pipeline itself measures.
Lines are rate-limited so a 300-batch run does not print 300 lines.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Callable, TextIO


def line_reporter(
    logger: logging.Logger,
    stream: TextIO | None = None,
    echo: bool = True,
) -> Callable[[str], None]:
    """
    A reporter that writes each line to the run log at INFO, and prints it.

    Args:
        logger: The command's logger (a literal name under ``src``).
        stream: Where to print; standard output by default. The worker
            passes standard error so standard output carries only its
            protocol.
        echo: Print as well as log. False keeps details in the log only.
    """

    def report(line: str) -> None:
        if echo:
            print(line, file=stream if stream is not None else sys.stdout)
        logger.info(line)

    return report


class ProgressReporter:
    """
    Print ``Batch: k / n, Progress: p percent, Elapsed, ETA`` lines.

    Args:
        label: Unit name shown at the start of each line.
        interval_seconds: Minimum wall-clock gap between two printed lines;
            the final unit is always printed.
        emit: Output function, ``print`` by default (tests inject a list).
        clock: Monotonic clock, injectable for deterministic tests.
    """

    def __init__(
        self,
        label: str,
        interval_seconds: float,
        emit: Callable[[str], None] = print,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.label = label
        self.interval_seconds = interval_seconds
        self.emit = emit
        self.clock = clock
        self.started = clock()
        self.last_emitted: float | None = None

    def __call__(self, completed: int, total: int) -> None:
        """Report ``completed`` of ``total`` units if the interval has elapsed."""

        now = self.clock()
        is_final = completed >= total
        interval_passed = (
            self.last_emitted is None
            or now - self.last_emitted >= self.interval_seconds
        )
        if not (is_final or interval_passed):
            return

        elapsed = now - self.started
        percent = 100.0 * completed / total if total else 100.0
        if 0 < completed < total:
            remaining = elapsed / completed * (total - completed)
            eta = f"{remaining:.1f} s"
        elif is_final:
            eta = "0.0 s"
        else:
            eta = "unknown"

        self.emit(
            f"{self.label}: {completed} / {total}, "
            f"Progress: {percent:.2f} percent, "
            f"Elapsed: {elapsed:.1f} s, ETA: {eta}"
        )
        self.last_emitted = now
