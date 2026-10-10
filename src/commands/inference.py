"""
Easy-to-use transcription command for the standalone Parakeet runtime.

Examples
--------
Single file:
    venv\\Scripts\\python.exe inference.py --transcribe path\\audio.wav

Folder batch (writes a CSV into the folder):
    venv\\Scripts\\python.exe inference.py --transcribe path\\audio_folder

The first invocation prepares weights and creates a checkpoint-local readiness
marker. Later invocations check the marker, load the prepared model, and
transcribe. All tunables live in config.toml; the command takes only the input.
"""

from __future__ import annotations

import csv
import io
import logging
import sys
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..audio.media import inspect_media
from ..configuration.settings import Settings, load_settings
from ..inference.offline import FileStatus, OfflineRunResult, OfflineTranscriber
from ..inference.planning import AudioMetadata
from ..runtime.filesystem import write_text_atomic
from ..runtime.logging_setup import configure_run_logging
from ..runtime.memory import peak_process_memory_bytes
from .model_loading import prepare_inference_model
from .reporting import ProgressReporter, line_reporter


# A literal name instead of __name__: under `python -m src.commands.inference`
# __name__ is "__main__", which is outside the "src" logger that owns the run
# log file, so this module's lines and tracebacks would never reach it.
logger = logging.getLogger("src.commands.inference")

CSV_COLUMNS = (
    "path_audio",
    "filename_audio",
    "duration_audio",
    "status",
    "transcription",
)


@dataclass(frozen=True)
class DiscoveredInput:
    """Readable audio to transcribe, plus files that were not audio."""

    audio: tuple[AudioMetadata, ...]
    unreadable: tuple[tuple[Path, str], ...]


@dataclass(frozen=True)
class CommandOutcome:
    """What a transcription command produced, for exit-code decisions."""

    result: OfflineRunResult
    unreadable: tuple[tuple[Path, str], ...]
    csv_path: Path | None
    csv_fallback_used: bool


# =============================================================================
# Input discovery and output persistence
# =============================================================================
# Discovery is deterministic so CSV row order is stable across runs, and every
# candidate is content-probed. Files that are not readable audio are reported
# (terminal, log, and a CSV row) instead of silently disappearing. Probed
# metadata is handed to the transcriber so no container is inspected twice.
# =============================================================================


def discover_audio_files(
    input_path: Path,
    extensions: tuple[str, ...],
    recursive: bool,
    excluded_names: frozenset[str] = frozenset(),
    require_audio: bool = True,
) -> DiscoveredInput:
    """
    Resolve one file, or probe every candidate under one directory.

    Args:
        input_path: Media file or directory.
        extensions: Optional lowercase suffix allow-list; empty means probe all.
        recursive: Whether to descend into subdirectories.
        excluded_names: File names never treated as input (the output CSV).
        require_audio: Raise when nothing readable is found. The worker passes
            False so it can report every file, readable or not.

    Returns:
        Readable audio metadata and ``(path, reason)`` for unreadable files,
        both sorted by path.

    Raises:
        FileNotFoundError: If the path does not exist, or (with
            ``require_audio``) no audio is found.
        ValueError: If a single explicitly named file is not readable audio
            (only with ``require_audio``).
        RuntimeError: If a codec path is configured but invalid; that is a
            setup error, not a property of one file.
    """

    resolved_input = input_path.expanduser().resolve()
    if resolved_input.is_file():
        try:
            if extensions and resolved_input.suffix.lower() not in extensions:
                raise ValueError(f"Unsupported media extension: {resolved_input.suffix!r}")
            return DiscoveredInput(audio=(inspect_media(resolved_input),), unreadable=())
        except ValueError as error:
            if require_audio:
                raise
            logger.warning("unreadable file %s: %s", resolved_input, error)
            return DiscoveredInput(audio=(), unreadable=((resolved_input, str(error)),))

    if not resolved_input.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {resolved_input}")

    pattern = "**/*" if recursive else "*"
    candidates = sorted(path for path in resolved_input.glob(pattern) if path.is_file())
    audio: list[AudioMetadata] = []
    unreadable: list[tuple[Path, str]] = []
    for path in candidates:
        if path.name in excluded_names:
            continue
        if extensions and path.suffix.lower() not in extensions:
            continue
        try:
            audio.append(inspect_media(path))
        except ValueError as error:
            unreadable.append((path, str(error)))
            logger.warning("unreadable file %s: %s", path, error)

    if not audio and require_audio:
        raise FileNotFoundError(
            f"No readable audio files found under {resolved_input} "
            f"({len(unreadable)} unreadable); "
            f"extension filter: {extensions or 'none (content probing)'}"
        )
    return DiscoveredInput(audio=tuple(audio), unreadable=tuple(unreadable))


def render_transcription_csv(rows: Iterable[dict[str, object]]) -> str:
    """Render rows with the fixed user-facing column order."""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_transcription_csv(output_path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write transcription rows atomically so a crash never leaves half a CSV."""

    write_text_atomic(output_path, render_transcription_csv(rows))


def persist_transcriptions(
    output_path: Path,
    fallback_path: Path,
    rows: list[dict[str, object]],
) -> tuple[Path, bool]:
    """
    Write the CSV, falling back to a second location if the first fails.

    On Windows a CSV still open in Excel cannot be replaced. Losing every
    transcript of a long run to that is worse than writing them elsewhere, so
    the fallback keeps the results and the caller reports the failure.

    Returns:
        The path actually written and whether it is the fallback.

    Raises:
        OSError: If both locations fail; the first error is chained.
    """

    try:
        write_transcription_csv(output_path, rows)
        return output_path, False
    except OSError as primary_error:
        logger.error("could not write %s: %s; trying %s", output_path, primary_error, fallback_path)
        try:
            write_transcription_csv(fallback_path, rows)
        except OSError as fallback_error:
            raise fallback_error from primary_error
        return fallback_path, True


def csv_rows(
    result: OfflineRunResult,
    unreadable: Iterable[tuple[Path, str]] = (),
) -> list[dict[str, object]]:
    """Convert results and unreadable files into CSV rows sorted by path."""

    rows: list[tuple[Path, dict[str, object]]] = [
        (
            file_result.path,
            {
                "path_audio": str(file_result.path),
                "filename_audio": file_result.path.name,
                "duration_audio": round(file_result.duration_seconds, 6),
                "status": file_result.status.value,
                "transcription": file_result.transcript,
            },
        )
        for file_result in result.files
    ]
    rows.extend(
        (
            path,
            {
                "path_audio": str(path),
                "filename_audio": path.name,
                "duration_audio": "",
                "status": FileStatus.UNREADABLE.value,
                "transcription": "",
            },
        )
        for path, _reason in unreadable
    )
    return [row for _path, row in sorted(rows, key=lambda pair: pair[0])]


# =============================================================================
# Terminal summaries
# =============================================================================


def _format_bytes(value: object) -> str:
    if not isinstance(value, int):
        return "not available"
    return f"{value / 2**20:.1f} MiB"


def _audio_per_second(result: OfflineRunResult, wall_seconds: float) -> float | None:
    """Seconds of audio transcribed per second of wall time, if both are known."""

    if result.total_audio_seconds <= 0 or wall_seconds <= 0:
        return None
    return result.total_audio_seconds / wall_seconds


def _speed_line(result: OfflineRunResult, wall_seconds: float) -> str:
    """One line every user sees: how long the transcription took."""

    speed = _audio_per_second(result, wall_seconds)
    if speed is None:
        return f"Processing seconds: {wall_seconds:.3f}"
    if speed >= 1.0:
        comparison = f"{speed:.1f}x faster than real time"
    else:
        # A CPU can be slower than the audio; never claim "0x faster".
        comparison = f"{1.0 / speed:.1f}x slower than real time"
    return f"Processing seconds: {wall_seconds:.3f} ({comparison})"


def _stage_details(result: OfflineRunResult, wall_seconds: float) -> list[str]:
    """
    Stage timings and memory peaks: the terminal shows them only on request.

    The wall time itself is on the speed line, which every run prints.
    """

    lines = [
        f"Media decode seconds: {result.media_decode_seconds:.3f}",
        f"Feature extraction seconds: {result.feature_seconds:.3f}",
        f"Model generation seconds: {result.generation_seconds:.3f}",
        f"Collapse recovery seconds: {result.recovery_seconds:.3f}",
    ]
    speed = _audio_per_second(result, wall_seconds)
    if speed is not None:
        lines.append(f"Real-time factor: {1.0 / speed:.6f}")
        lines.append(f"Throughput audio seconds per second: {speed:.3f}")
    lines.append(f"Work items: {len(result.plan.items)}, batches: {len(result.plan.batches)}")
    lines.append(f"Peak accelerator memory allocated: {_format_bytes(result.peak_memory.get('peak_allocated_bytes'))}")
    lines.append(f"Peak process memory: {_format_bytes(peak_process_memory_bytes())}")
    return lines


def _print_status_counts(result: OfflineRunResult, unreadable_count: int) -> None:
    counts: dict[str, int] = {}
    for file_result in result.files:
        counts[file_result.status.value] = counts.get(file_result.status.value, 0) + 1
    if unreadable_count:
        counts[FileStatus.UNREADABLE.value] = unreadable_count
    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    print(f"File statuses: {summary}")


# =============================================================================
# CLI workflow
# =============================================================================


def parse_arguments(argv: list[str] | None = None) -> Namespace:
    """Parse the small user-facing transcription command."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcribe",
        type=Path,
        required=True,
        help="One supported media file or a folder containing supported files",
    )
    return parser.parse_args(argv)


def transcribe_input(input_path: Path, settings: Settings, run_id: str) -> CommandOutcome:
    """
    Run single-file display or folder CSV transcription.

    The input is discovered before any checkpoint work, so a mistyped path or
    a folder without audio fails in milliseconds instead of after a download
    or a multi-second model load.
    """

    inference_settings = settings.inference
    resolved_input = input_path.expanduser().resolve()
    single_file = resolved_input.is_file()
    logger.info("input=%s mode=%s", resolved_input, "file" if single_file else "folder")

    discovered = discover_audio_files(
        resolved_input,
        inference_settings.audio_extensions,
        inference_settings.recursive,
        excluded_names=frozenset({inference_settings.output_filename}),
    )
    for path, reason in discovered.unreadable:
        print(f"Skipped unreadable file: {path.name} ({reason.splitlines()[0][:160]})")

    # Details (versions, precision, memory budget, stage timings) always go
    # to the run log; the terminal shows them only with terminal_details.
    details = line_reporter(logger, echo=settings.logging.terminal_details)
    notices = line_reporter(logger)
    prepared = prepare_inference_model(settings, details, notices)
    model, configuration, inference_settings = prepared.model, prepared.configuration, prepared.inference
    if not settings.logging.terminal_details:
        # The detailed lines went to the log only; the terminal still names
        # the device, precision, and library versions, on one line.
        print(prepared.runtime.summary(prepared.precision))

    print(f"Transcribing {len(discovered.audio)} file(s)")
    transcriber = OfflineTranscriber(model, configuration, inference_settings)
    reporter = ProgressReporter("Batch", inference_settings.progress_interval_seconds)
    started = time.perf_counter()
    result = transcriber.transcribe(
        [record.path for record in discovered.audio],
        metadata=discovered.audio,
        progress_callback=reporter,
    )
    wall_seconds = time.perf_counter() - started

    if single_file:
        file_result = result.files[0]
        print(f"Status: {file_result.status.value}")
        print(f"Transcript: {file_result.transcript}")
        print(f"Audio duration seconds: {file_result.duration_seconds:.3f}")
        notices(_speed_line(result, wall_seconds))
        for line in _stage_details(result, wall_seconds):
            details(line)
        return CommandOutcome(result, (), None, False)

    output_path = resolved_input / inference_settings.output_filename
    fallback_path = settings.paths.log_dir / f"transcriptions_{run_id}.csv"
    rows = csv_rows(result, discovered.unreadable)
    written_path, fallback_used = persist_transcriptions(output_path, fallback_path, rows)

    print(f"CSV: {written_path}")
    print(f"Files: {len(rows)}")
    _print_status_counts(result, len(discovered.unreadable))
    print(f"Total audio seconds: {result.total_audio_seconds:.3f}")
    notices(_speed_line(result, wall_seconds))
    for line in _stage_details(result, wall_seconds):
        details(line)
    logger.info("csv=%s rows=%d fallback=%s", written_path, len(rows), fallback_used)
    return CommandOutcome(result, discovered.unreadable, written_path, fallback_used)


def main(argv: list[str] | None = None) -> int:
    """
    Run the command and return the process exit code.

    0: every file transcribed (possibly with review statuses, which are
       warned about). 1: the run failed, or the CSV had to be written to the
       fallback location.
    """

    arguments = parse_arguments(argv)
    settings = load_settings()
    run_id, log_path = configure_run_logging(settings.paths.log_dir, settings.logging.level)
    if settings.logging.terminal_details:
        print(f"Run id: {run_id}")
    print(f"Log file: {log_path}")
    logger.info("command=transcribe argv=%s", sys.argv[1:] if argv is None else argv)

    try:
        outcome = transcribe_input(arguments.transcribe, settings, run_id)
    except Exception as error:
        logger.exception("run failed")
        print(f"Error: {error}", file=sys.stderr)
        print(f"Details: {log_path} (run {run_id})", file=sys.stderr)
        return 1

    review_count = len(outcome.unreadable) + sum(
        1
        for file_result in outcome.result.files
        if file_result.status not in (FileStatus.OK, FileStatus.NO_SPEECH)
    )
    if review_count:
        print(f"Warning: {review_count} file(s) need review; see the status column or log.")

    if outcome.csv_fallback_used:
        print(
            f"Error: the CSV could not be written to the input folder; "
            f"transcripts were saved to {outcome.csv_path}. Details: {log_path}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
