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
and transcribe the requested input. Use ``--repair`` to explicitly remove the
marker and run full checkpoint preparation again.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable

from ..bootstrap import ensure_first_run_ready
from ..config import DEFAULT_WEIGHTS_DIR
from ..inference import Transcriber
from ..model.parakeet import load_model
from ..settings import load_inference_settings


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
        if resolved_input.suffix.lower() not in extensions:
            raise ValueError(
                f"Unsupported audio extension {resolved_input.suffix!r}; "
                f"supported extensions: {extensions}"
            )
        return [resolved_input]

    if not resolved_input.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {resolved_input}")

    pattern = "**/*" if recursive else "*"
    audio_paths = sorted(
        path
        for path in resolved_input.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    )
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
                fieldnames=["filename", "duration_audio", "transcribe"],
            )
            writer.writeheader()
            writer.writerows(rows)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    temporary_path.replace(output_path)


def display_name(audio_path: Path, input_path: Path, single_file: bool) -> str:
    """Return a stable CSV filename value for single-file or folder mode."""

    if single_file:
        return audio_path.name
    return str(audio_path.relative_to(input_path)).replace("\\", "/")


# =============================================================================
# CLI workflow
# =============================================================================
# Bootstrap is deliberately before model loading. On the first run it performs
# download, conversion, and strict validation. On later runs marker existence is
# the only weight-readiness check; model loading is still required for inference.
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse the small user-facing transcription command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcribe",
        type=Path,
        required=True,
        help="One WAV file or a folder containing WAV files",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Force first-run checkpoint preparation before transcription",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help="Checkpoint directory; normally the configured default is used",
    )
    return parser.parse_args()


def transcribe_input(
    input_path: Path,
    weights_dir: Path,
    repair: bool,
) -> None:
    """Run single-file display or folder CSV transcription."""

    settings = load_inference_settings()
    resolved_input = input_path.expanduser().resolve()
    single_file = resolved_input.is_file()

    def progress(message: str) -> None:
        print(message)

    bootstrap = ensure_first_run_ready(
        checkpoint_dir=weights_dir,
        force_repair=repair,
        progress_callback=progress,
    )
    print(f"Weights: {bootstrap.action}")

    model, configuration, _metadata = load_model(weights_dir)
    transcriber = Transcriber(model, configuration)
    audio_paths = discover_audio_files(
        resolved_input,
        settings.audio_extensions,
        settings.recursive,
    )

    if single_file:
        batch = transcriber.transcribe(audio_paths)
        print(f"Transcript: {batch.transcripts[0]}")
        print(f"Audio duration seconds: {batch.total_audio_seconds:.3f}")
        print(f"Inference time seconds: {batch.elapsed_seconds:.3f}")
        print(f"Real-time factor: {batch.real_time_factor:.6f}")
        print(f"Throughput audio seconds/second: {batch.audio_seconds_per_second:.3f}")
        return

    output_path = resolved_input / settings.output_filename
    rows: list[dict[str, object]] = []
    total_files = len(audio_paths)
    started = time.perf_counter()
    batch_metrics: list[tuple[int, float, float, float]] = []

    for start_index in range(0, total_files, settings.batch_size):
        batch_paths = audio_paths[start_index : start_index + settings.batch_size]
        batch = transcriber.transcribe(batch_paths)
        batch_metrics.append(
            (
                len(batch_paths),
                batch.elapsed_seconds,
                batch.real_time_factor,
                batch.audio_seconds_per_second,
            )
        )
        for audio_path, duration, transcript in zip(
            batch_paths,
            batch.audio_durations_seconds,
            batch.transcripts,
        ):
            rows.append(
                {
                    "filename": display_name(audio_path, resolved_input, False),
                    "duration_audio": round(duration, 6),
                    "transcribe": transcript,
                }
            )

        completed = min(start_index + len(batch_paths), total_files)
        print(
            f"Transcribed {completed}/{total_files} files | "
            f"inference={batch.elapsed_seconds:.3f}s | "
            f"rtf={batch.real_time_factor:.6f} | "
            f"throughput={batch.audio_seconds_per_second:.3f} audio-s/s"
        )

    write_transcription_csv(output_path, rows)
    elapsed = time.perf_counter() - started
    total_audio_seconds = sum(
        float(row["duration_audio"])
        for row in rows
    )
    print(f"CSV: {output_path}")
    print(f"Files: {total_files}")
    print(f"Total audio duration seconds: {total_audio_seconds:.3f}")
    print(f"Total wall-clock seconds: {elapsed:.3f}")
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
    fastest_batch = min(batch_metrics, key=lambda item: item[1])
    slowest_batch = max(batch_metrics, key=lambda item: item[1])
    print(f"Fastest batch inference seconds: {fastest_batch[1]:.3f}")
    print(f"Slowest batch inference seconds: {slowest_batch[1]:.3f}")


def main() -> None:
    """Run the user-facing transcription command."""

    arguments = parse_arguments()
    transcribe_input(
        input_path=arguments.transcribe,
        weights_dir=arguments.weights_dir,
        repair=arguments.repair,
    )


if __name__ == "__main__":
    main()
