"""
Easy-to-use transcription command for the standalone Parakeet runtime.

Examples
--------
Single WAV:
    venv\\Scripts\\python.exe inference.py --transcribe path\\audio.wav

Folder batch:
    venv\\Scripts\\python.exe inference.py --transcribe path\\audio_folder

The first invocation prepares weights and creates a checkpoint-local readiness
marker. Later invocations check only that marker, load the already-prepared model,
and transcribe the requested input. Checkpoint repair remains an internal API
operation; it is intentionally not part of the simplified public command.
"""

from __future__ import annotations

import csv
import os
import tempfile
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable

from ..checkpoint.bootstrap import ensure_first_run_ready
from ..configuration.config import DEFAULT_WEIGHTS_DIR
from ..configuration.settings import load_inference_settings
from ..audio.media import inspect_media
from ..inference.offline import OfflineTranscriber
from ..models.parakeet import load_model

# =============================================================================
# Input discovery and output persistence
# =============================================================================
# Discovery is deterministic so CSV row order is stable across runs. Folder
# outputs use relative paths to avoid collisions when nested directories contain
# equal filenames. Atomic CSV replacement prevents an interrupted batch from
# leaving a misleading half-result file.
# =============================================================================


def discover_audio_files(
    input_path: Path,
    extensions: tuple[str, ...],
    recursive: bool,
) -> list[Path]:
    """Resolve one file or discover supported audio files under one directory."""

    resolved_input = input_path.expanduser().resolve()
    if resolved_input.is_file():
        if extensions and resolved_input.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported media extension: {resolved_input.suffix!r}")
        inspect_media(resolved_input)
        return [resolved_input]

    if not resolved_input.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {resolved_input}")

    pattern = "**/*" if recursive else "*"
    candidates = sorted(path for path in resolved_input.glob(pattern) if path.is_file())
    audio_paths: list[Path] = []
    for path in candidates:
        if extensions and path.suffix.lower() not in extensions:
            continue
        try:
            inspect_media(path)
        except (OSError, RuntimeError, ValueError):
            continue
        audio_paths.append(path)
    if not audio_paths:
        raise FileNotFoundError(
            f"No supported audio files found under {resolved_input}; "
            f"extensions: {extensions}"
        )

    return audio_paths


def write_transcription_csv(
    output_path: Path,
    rows: Iterable[dict[str, object]],
) -> None:
    """Write transcription rows atomically with the user-facing CSV schema."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output_path.parent,
        prefix=f"{output_path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            writer = csv.DictWriter(
                temporary_file,
                fieldnames=[
                    "path_audio",
                    "filename_audio",
                    "duration_audio",
                    "transcription",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    temporary_path.replace(output_path)


# =============================================================================
# CLI workflow
# =============================================================================
# Bootstrap is deliberately before model loading. On the first run it performs
# download, conversion, and strict validation. On later runs marker existence is
# the only weight-readiness check; model loading is still required for inference.
# =============================================================================


def parse_arguments() -> Namespace:
    """Parse the small user-facing transcription command."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcribe",
        type=Path,
        required=True,
        help="One supported media file or a folder containing supported files",
    )
    return parser.parse_args()


def transcribe_input(
    input_path: Path,
) -> None:
    """Run single-file display or folder CSV transcription."""

    settings = load_inference_settings()
    resolved_input = input_path.expanduser().resolve()
    single_file = resolved_input.is_file()

    def progress(message: str) -> None:
        print(message)

    bootstrap = ensure_first_run_ready(
        checkpoint_dir=DEFAULT_WEIGHTS_DIR,
        progress_callback=progress,
    )
    print(f"Weights: {bootstrap.action}")

    model, configuration, _metadata = load_model(DEFAULT_WEIGHTS_DIR)
    transcriber = OfflineTranscriber(model, configuration, settings)
    audio_paths = discover_audio_files(
        resolved_input,
        settings.audio_extensions,
        settings.recursive,
    )
    if not single_file:
        print(f"Transcribing {len(audio_paths)} files")

    started = time.perf_counter()

    def inference_progress(completed_batches: int, total_batches: int) -> None:
        """Print casual progress for folder work without printing transcripts."""

        if single_file:
            return
        ratio = completed_batches / total_batches if total_batches else 1.0
        width = 28
        filled = min(width, int(ratio * width))
        print(
            f"Inference progress: [{('#' * filled)}{('-' * (width - filled))}] "
            f"{completed_batches}/{total_batches} batches"
        )

    result = transcriber.transcribe(
        audio_paths,
        progress_callback=inference_progress,
    )
    elapsed = time.perf_counter() - started

    if single_file:
        file_result = result.files[0]
        print(f"Transcript status: {file_result.status}")
        print(f"Transcript: {file_result.transcript}")
        print(f"Audio duration seconds: {file_result.duration_seconds:.3f}")
        print(f"Inference time seconds: {result.elapsed_seconds:.3f}")
        print(f"Media decode seconds: {result.media_decode_seconds:.3f}")
        print(f"Feature extraction seconds: {result.feature_seconds:.3f}")
        print(f"Model generation seconds: {result.generation_seconds:.3f}")
        print(f"Real-time factor: {result.real_time_factor:.6f}")
        print(
            f"Throughput audio seconds/second: "
            f"{result.total_audio_seconds / result.elapsed_seconds:.3f}"
            if result.elapsed_seconds > 0
            else "Throughput audio seconds/second: 0.000"
        )
        return

    output_path = resolved_input / settings.output_filename
    rows: list[dict[str, object]] = []
    for file_result in result.files:
        rows.append(
            {
                "path_audio": str(file_result.path),
                "filename_audio": file_result.path.name,
                "duration_audio": round(file_result.duration_seconds, 6),
                "transcription": file_result.transcript,
            }
        )

    write_transcription_csv(output_path, rows)
    total_audio_seconds = result.total_audio_seconds
    print(f"CSV: {output_path}")
    print(f"Files: {len(result.files)}")
    print(f"Total audio duration seconds: {total_audio_seconds:.3f}")
    print(f"Total wall-clock seconds: {elapsed:.3f}")
    print(f"Media decode seconds: {result.media_decode_seconds:.3f}")
    print(f"Feature extraction seconds: {result.feature_seconds:.3f}")
    print(f"Model generation seconds: {result.generation_seconds:.3f}")
    print(
        f"End-to-end real-time factor: "
        f"{elapsed / total_audio_seconds:.6f}"
        if total_audio_seconds > 0
        else "End-to-end real-time factor: 0.000000"
    )
    print(
        f"End-to-end throughput audio seconds/second: "
        f"{total_audio_seconds / elapsed:.3f}"
        if elapsed > 0
        else "End-to-end throughput audio seconds/second: 0.000"
    )
    print(f"Planned work items: {len(result.plan.items)}")
    print(f"Planned batches: {len(result.plan.batches)}")
    print(f"Peak memory: {result.peak_memory}")


def main() -> None:
    """Run the user-facing transcription command."""

    arguments = parse_arguments()
    transcribe_input(
        input_path=arguments.transcribe,
    )


if __name__ == "__main__":
    main()
