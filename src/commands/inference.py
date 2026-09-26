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
from pathlib import Path
from typing import Iterable

from ..audio.media import inspect_media
from ..checkpoint.bootstrap import ensure_first_run_ready
from ..configuration.settings import Settings, load_settings
from ..inference.offline import FileStatus, OfflineRunResult, OfflineTranscriber
from ..inference.planning import AudioMetadata
from ..models.parakeet import load_model
from ..runtime.device import describe_runtime
from ..runtime.filesystem import write_text_atomic
from ..runtime.logging_setup import configure_run_logging
from .reporting import ProgressReporter


logger = logging.getLogger(__name__)

CSV_COLUMNS = (
    "path_audio",
    "filename_audio",
    "duration_audio",
    "status",
    "transcription",
)


# =============================================================================
# Input discovery and output persistence
# =============================================================================
# Discovery is deterministic so CSV row order is stable across runs, and every
# candidate is content-probed so non-audio files are skipped rather than
# failing the run. Probed metadata is handed to the transcriber so no container
# is inspected twice.
# =============================================================================


def discover_audio_files(
    input_path: Path,
    extensions: tuple[str, ...],
    recursive: bool,
    excluded_names: frozenset[str] = frozenset(),
) -> list[AudioMetadata]:
    """
    Resolve one file, or probe every candidate under one directory.

    Args:
        input_path: Media file or directory.
        extensions: Optional lowercase suffix allow-list; empty means probe all.
        recursive: Whether to descend into subdirectories.
        excluded_names: File names never treated as input (the output CSV).

    Returns:
        Metadata for every readable audio file, sorted by path.

    Raises:
        FileNotFoundError: If the path does not exist or no audio is found.
        ValueError: If a single explicitly named file is not readable audio.
    """

    resolved_input = input_path.expanduser().resolve()
    if resolved_input.is_file():
        if extensions and resolved_input.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported media extension: {resolved_input.suffix!r}")
        return [inspect_media(resolved_input)]

    if not resolved_input.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {resolved_input}")

    pattern = "**/*" if recursive else "*"
    candidates = sorted(path for path in resolved_input.glob(pattern) if path.is_file())
    discovered: list[AudioMetadata] = []
    for path in candidates:
        if path.name in excluded_names:
            continue
        if extensions and path.suffix.lower() not in extensions:
            continue
        try:
            discovered.append(inspect_media(path))
        except ValueError as error:
            logger.info("skipped non-audio file %s: %s", path, error)

    if not discovered:
        raise FileNotFoundError(
            f"No supported audio files found under {resolved_input}; "
            f"extension filter: {extensions or 'none (content probing)'}"
        )
    return discovered


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


def csv_rows(result: OfflineRunResult) -> list[dict[str, object]]:
    """Convert run results into CSV rows in input order."""

    return [
        {
            "path_audio": str(file_result.path),
            "filename_audio": file_result.path.name,
            "duration_audio": round(file_result.duration_seconds, 6),
            "status": file_result.status.value,
            "transcription": file_result.transcript,
        }
        for file_result in result.files
    ]


# =============================================================================
# Terminal summaries
# =============================================================================


def _format_bytes(value: object) -> str:
    if not isinstance(value, int):
        return "not available"
    return f"{value / 2**20:.1f} MiB"


def _print_stage_summary(result: OfflineRunResult, wall_seconds: float) -> None:
    total_audio = result.total_audio_seconds
    print(f"Total audio seconds: {total_audio:.3f}")
    print(f"Wall-clock seconds: {wall_seconds:.3f}")
    print(f"Media decode seconds: {result.media_decode_seconds:.3f}")
    print(f"Feature extraction seconds: {result.feature_seconds:.3f}")
    print(f"Model generation seconds: {result.generation_seconds:.3f}")
    if total_audio > 0 and wall_seconds > 0:
        print(f"Real-time factor: {wall_seconds / total_audio:.6f}")
        print(f"Throughput audio seconds per second: {total_audio / wall_seconds:.3f}")
    print(f"Work items: {len(result.plan.items)}, batches: {len(result.plan.batches)}")
    print(f"Peak accelerator memory allocated: {_format_bytes(result.peak_memory.get('peak_allocated_bytes'))}")


def _print_status_counts(result: OfflineRunResult) -> None:
    counts: dict[str, int] = {}
    for file_result in result.files:
        counts[file_result.status.value] = counts.get(file_result.status.value, 0) + 1
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


def transcribe_input(input_path: Path, settings: Settings) -> OfflineRunResult:
    """Run single-file display or folder CSV transcription."""

    inference_settings = settings.inference
    resolved_input = input_path.expanduser().resolve()
    single_file = resolved_input.is_file()
    logger.info("input=%s mode=%s", resolved_input, "file" if single_file else "folder")

    bootstrap = ensure_first_run_ready(
        checkpoint_dir=settings.paths.weights_dir,
        checkpoint_settings=settings.checkpoint,
        progress_callback=print,
    )
    print(f"Weights: {bootstrap.action}")
    logger.info("weights action=%s marker=%s", bootstrap.action, bootstrap.marker_path)

    load_started = time.perf_counter()
    model, configuration, _metadata = load_model(settings.paths.weights_dir)
    load_seconds = time.perf_counter() - load_started
    runtime_report = describe_runtime(
        next(model.parameters()).device,
        next(model.parameters()).dtype,
    )
    for line in runtime_report.lines():
        print(line)
        logger.info(line)
    print(f"Model load seconds: {load_seconds:.3f}")
    logger.info("model load seconds=%.3f", load_seconds)

    metadata = discover_audio_files(
        resolved_input,
        inference_settings.audio_extensions,
        inference_settings.recursive,
        excluded_names=frozenset({inference_settings.output_filename}),
    )
    print(f"Transcribing {len(metadata)} file(s)")

    transcriber = OfflineTranscriber(model, configuration, inference_settings)
    reporter = ProgressReporter("Batch", inference_settings.progress_interval_seconds)
    started = time.perf_counter()
    result = transcriber.transcribe(
        [record.path for record in metadata],
        metadata=metadata,
        progress_callback=reporter,
    )
    wall_seconds = time.perf_counter() - started

    if single_file:
        file_result = result.files[0]
        print(f"Status: {file_result.status.value}")
        print(f"Transcript: {file_result.transcript}")
        print(f"Audio duration seconds: {file_result.duration_seconds:.3f}")
        _print_stage_summary(result, wall_seconds)
        return result

    output_path = resolved_input / inference_settings.output_filename
    write_transcription_csv(output_path, csv_rows(result))
    print(f"CSV: {output_path}")
    print(f"Files: {len(result.files)}")
    _print_status_counts(result)
    _print_stage_summary(result, wall_seconds)
    logger.info("csv=%s files=%d", output_path, len(result.files))
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the command; return the process exit code."""

    arguments = parse_arguments(argv)
    settings = load_settings()
    run_id, log_path = configure_run_logging(settings.paths.log_dir, settings.logging.level)
    print(f"Run id: {run_id}")
    print(f"Log file: {log_path}")
    logger.info("command=transcribe argv=%s", sys.argv[1:] if argv is None else argv)

    try:
        result = transcribe_input(arguments.transcribe, settings)
    except Exception as error:
        logger.exception("run failed")
        print(f"Error: {error}", file=sys.stderr)
        print(f"Details: {log_path} (run {run_id})", file=sys.stderr)
        return 1

    problem_files = [
        file_result
        for file_result in result.files
        if file_result.status not in (FileStatus.OK, FileStatus.NO_SPEECH)
    ]
    if problem_files:
        print(f"Warning: {len(problem_files)} file(s) need review; see the status column or log.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
