"""
Development benchmark: repeatable timing of the real transcription pipeline.

    venv\\Scripts\\python.exe -m src.commands.benchmark --input path\\to\\audio_or_folder

The model is loaded once and its cold-start time is reported separately. Then
``[benchmark] warmup_rounds`` transcriptions run and are discarded (CUDA
kernel selection, file cache), and ``measured_rounds`` are timed. Each run
appends one JSON line to ``<metrics_dir>/benchmark_<YYYY-MM-DD>.jsonl`` with
the git commit, device, settings, every round's stage timings and memory
peaks, per-file metrics, and whether all rounds produced the same text. JSON
lines are the source of truth; compare them across commits to judge a
change.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..configuration.config import PROJECT_ROOT
from ..configuration.settings import Settings, load_settings
from ..inference.offline import OfflineRunResult, OfflineTranscriber
from ..runtime.logging_setup import configure_run_logging
from ..runtime.memory import peak_process_memory_bytes
from .inference import discover_audio_files
from .model_loading import prepare_inference_model
from .reporting import line_reporter


# Literal name: under `python -m` __name__ is "__main__", outside the run log.
logger = logging.getLogger("src.commands.benchmark")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """The input to benchmark; rounds and paths come from config.toml."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Audio file or folder")
    return parser.parse_args(argv)


def git_revision(timeout_seconds: float) -> dict[str, Any]:
    """
    Commit and dirty flag of the working tree, or nulls when git cannot say.

    A slow network share, an fsmonitor hook, or a lock wait can stall
    ``git status``; after ``timeout_seconds`` the value is recorded as unknown
    instead of blocking the benchmark after all its rounds have run.
    """

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def round_record(result: OfflineRunResult, wall_seconds: float) -> dict[str, Any]:
    """Everything measured in one timed round."""

    # Wall time measured here, around the whole transcribe call; None when the
    # input has no audio, so a ratio is never invented.
    real_time_factor = None
    if result.total_audio_seconds > 0:
        real_time_factor = wall_seconds / result.total_audio_seconds

    return {
        "wall_seconds": wall_seconds,
        "media_decode_seconds": result.media_decode_seconds,
        "feature_seconds": result.feature_seconds,
        "generation_seconds": result.generation_seconds,
        "recovery_seconds": result.recovery_seconds,
        "real_time_factor": real_time_factor,
        "work_items": len(result.plan.items),
        "batches": len(result.plan.batches),
        "peak_accelerator_allocated_bytes": result.peak_memory.get("peak_allocated_bytes"),
        "peak_accelerator_reserved_bytes": result.peak_memory.get("peak_reserved_bytes"),
        "files": [
            {
                "name": file_result.path.name,
                "duration_seconds": file_result.duration_seconds,
                "status": file_result.status.value,
                "chunks": len(file_result.chunks),
                "recovered_chunks": sum(chunk.recovered for chunk in file_result.chunks),
                "processing_seconds": file_result.processing_seconds,
                "real_time_factor": file_result.real_time_factor,
            }
            for file_result in result.files
        ],
    }


def format_optional(value: float | None, digits: int) -> str:
    """Fixed-point text for a measurement, or "not available" for None."""

    if value is None:
        return "not available"
    return f"{value:.{digits}f}"


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def run_benchmark(input_path: Path, settings: Settings, run_id: str) -> tuple[dict[str, Any], Path]:
    """
    Load once, warm up, measure, and append the JSON record.

    Returns:
        The record and the JSONL file it was appended to.
    """

    discovered = discover_audio_files(
        input_path,
        settings.inference.audio_extensions,
        settings.inference.recursive,
        excluded_names=frozenset({settings.inference.output_filename}),
    )
    paths = [record.path for record in discovered.audio]

    # Reported as it happens, not only in the final record, so a run that
    # fails in a later round still shows which device and versions it ran on.
    prepared = prepare_inference_model(settings, line_reporter(logger))
    parameter = next(prepared.model.parameters())
    inference_settings = prepared.inference
    transcriber = OfflineTranscriber(prepared.model, prepared.configuration, inference_settings)

    def timed_round() -> tuple[OfflineRunResult, float]:
        if parameter.device.type == "cuda":
            torch.cuda.synchronize(parameter.device)
        started = time.perf_counter()
        result = transcriber.transcribe(paths, metadata=discovered.audio)
        return result, time.perf_counter() - started

    for warmup_index in range(settings.benchmark.warmup_rounds):
        _result, seconds = timed_round()
        print(f"Warm-up round {warmup_index + 1}: {seconds:.3f} s")

    rounds: list[dict[str, Any]] = []
    transcripts: list[tuple[str, ...]] = []
    for round_index in range(settings.benchmark.measured_rounds):
        result, seconds = timed_round()
        rounds.append(round_record(result, seconds))
        transcripts.append(tuple(file_result.transcript for file_result in result.files))
        real_time_factor = format_optional(rounds[-1]["real_time_factor"], 5)
        print(
            f"Round {round_index + 1}: wall {seconds:.3f} s, "
            f"generation {result.generation_seconds:.3f} s, "
            f"real-time factor {real_time_factor}"
        )

    record: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_revision(settings.benchmark.git_timeout_seconds),
        "runtime": dataclasses.asdict(prepared.runtime),
        "encoder_precision": prepared.precision,
        "graph_decoding": prepared.graph_decoding,
        "settings": {
            "inference": dataclasses.asdict(inference_settings),
            "memory": dataclasses.asdict(settings.memory),
            "benchmark": dataclasses.asdict(settings.benchmark),
        },
        "input": {
            "path": str(input_path.expanduser().resolve()),
            "files": len(paths),
            "audio_seconds": sum(record.duration_seconds for record in discovered.audio),
            "unreadable_files": len(discovered.unreadable),
        },
        "cold_start": {
            "weights_action": prepared.weights_action,
            "bootstrap_and_load_seconds": prepared.load_seconds,
        },
        "memory_budget": dataclasses.asdict(prepared.memory_budget),
        "rounds": rounds,
        "summary": {
            "wall_seconds": summarize([entry["wall_seconds"] for entry in rounds]),
            "generation_seconds": summarize([entry["generation_seconds"] for entry in rounds]),
            "transcripts_identical_across_rounds": len(set(transcripts)) == 1,
        },
        "peak_process_memory_bytes": peak_process_memory_bytes(),
    }

    settings.paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.paths.metrics_dir / f"benchmark_{datetime.now():%Y-%m-%d}.jsonl"
    with output_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, default=str) + "\n")
    return record, output_path


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark; return the process exit code."""

    arguments = parse_arguments(argv)
    settings = load_settings()
    run_id, log_path = configure_run_logging(settings.paths.log_dir, settings.logging.level)
    print(f"Run id: {run_id}")
    print(f"Log file: {log_path}")

    try:
        record, output_path = run_benchmark(arguments.input, settings, run_id)
    except Exception as error:
        logger.exception("benchmark failed")
        print(f"Error: {error}", file=sys.stderr)
        print(f"Details: {log_path} (run {run_id})", file=sys.stderr)
        return 1

    summary = record["summary"]
    print(f"Cold start seconds: {record['cold_start']['bootstrap_and_load_seconds']:.3f}")
    print(
        f"Wall seconds: mean {summary['wall_seconds']['mean']:.3f}, "
        f"min {summary['wall_seconds']['min']:.3f}, max {summary['wall_seconds']['max']:.3f}, "
        f"stdev {summary['wall_seconds']['stdev']:.3f}"
    )
    print(f"Generation seconds: mean {summary['generation_seconds']['mean']:.3f}")
    print(f"Transcripts identical across rounds: {summary['transcripts_identical_across_rounds']}")
    peak = record["peak_process_memory_bytes"]
    print(f"Peak process memory: {peak / 2**20:.1f} MiB" if peak else "Peak process memory: not available")
    print(f"Results: {output_path}")
    logger.info("benchmark record appended to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
