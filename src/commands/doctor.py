"""
Check this machine: environment, startup decisions, accuracy, and speed.

    venv\\Scripts\\python.exe -m src.commands.doctor

Run it on any computer or GPU before trusting a transcription there. It
prints:

- what PyTorch sees: the accelerator, the GPU and its compute capability,
  whether the installed build has kernels for it, tensor cores, and memory;
- what startup decided for each "auto" setting and why;
- whether the committed LibriSpeech clips transcribe within the accuracy
  budget of the test suite;
- how fast the encoder runs on full 15-second chunks, a number comparable
  across GPUs.

It appends one JSON line to ``<metrics_dir>/doctor_<YYYY-MM-DD>.jsonl``, so
results from different machines can be collected and compared. The exit
code is 0 only when the accuracy check passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from ..audio.features import ParakeetFeatureExtractor
from ..configuration.config import PROJECT_ROOT
from ..configuration.settings import Settings, load_settings
from ..inference.offline import FileStatus, OfflineTranscriber
from ..runtime.device import (
    build_supports_device,
    cuda_capability,
    has_tensor_cores,
    select_device,
    synchronize,
)
from ..runtime.logging_setup import configure_run_logging
from ..text.scoring import word_error_rate
from .benchmark import git_revision
from .model_loading import PreparedModel, prepare_inference_model
from .reporting import line_reporter

# Literal name: under `python -m` __name__ would be "__main__".
logger = logging.getLogger("src.commands.doctor")

CLIPS_MANIFEST = PROJECT_ROOT / "tests" / "data" / "librispeech" / "manifest.json"

# The corpus word error rate the full test tier allows on these clips
# (tests/full/test_accuracy.py): 3.67 percent measured, room for one more
# wrong word.
ACCURACY_WER_BUDGET = 0.046

# Encoder throughput measurement: one warm-up batch, then this many timed
# batches. The CPU gets a small batch so the check stays quick there.
THROUGHPUT_TIMED_BATCHES = 3
CPU_THROUGHPUT_ROWS = 2

# Amplitude of the synthetic throughput input; the encoder's cost does not
# depend on the content.
THROUGHPUT_AMPLITUDE = 0.05

MIB = 2**20


# =============================================================================
# Environment
# =============================================================================


def environment_facts() -> dict[str, Any]:
    """What PyTorch and the machine report, before anything is loaded."""

    device = select_device()
    facts: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "accelerator": device.type,
    }
    if device.type == "cuda":
        capability = cuda_capability(device)
        free_bytes, total_bytes = torch.accelerator.get_memory_info(device)
        facts.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "build_architectures": torch.cuda.get_arch_list(),
                "build_supports_device": build_supports_device(device),
                "tensor_cores": has_tensor_cores(device),
                "memory_total_mib": round(total_bytes / MIB),
                "memory_free_mib": round(free_bytes / MIB),
            }
        )
    return facts


def environment_lines(facts: dict[str, Any]) -> list[str]:
    """Plain-text lines for the environment facts."""

    lines = [
        f"Python: {facts['python']} on {facts['platform']}",
        f"Logical CPUs: {facts['logical_cpus']}",
        f"PyTorch: {facts['pytorch']}, CUDA runtime: {facts['cuda_runtime'] or 'not available'}",
        f"Accelerator: {facts['accelerator']}",
    ]
    if facts["accelerator"] == "cuda":
        supported = "yes" if facts["build_supports_device"] else "NO"
        lines += [
            f"GPU: {facts['device_name']}, compute capability {facts['compute_capability']}",
            f"Kernels for this GPU in the PyTorch build: {supported} "
            f"(built for {', '.join(facts['build_architectures'])})",
            f"Tensor cores: {'yes' if facts['tensor_cores'] else 'no'}",
            f"GPU memory: {facts['memory_free_mib']} MiB free of {facts['memory_total_mib']} MiB",
        ]
    return lines


# =============================================================================
# Accuracy
# =============================================================================


def check_accuracy(prepared: PreparedModel, report: Callable[[str], None]) -> dict[str, Any]:
    """
    Transcribe the committed LibriSpeech clips and score them.

    The check passes when every clip is ``ok`` and the corpus word error
    rate is within the test suite's budget. Whether each transcript equals
    the one recorded on the reference GPU is reported too; float16 results
    can differ by a word between GPUs without being wrong.

    Returns:
        ``passed`` (False also when the clips are missing or damaged), the
        corpus WER, per-clip results, and the processing time of a second,
        warm run.
    """

    if not CLIPS_MANIFEST.is_file():
        report(f"Accuracy check: FAILED, the reference clips are missing ({CLIPS_MANIFEST})")
        return {"passed": False, "reason": f"missing {CLIPS_MANIFEST}"}

    manifest = json.loads(CLIPS_MANIFEST.read_text(encoding="utf-8"))
    clips = manifest["clips"]
    paths = [CLIPS_MANIFEST.parent / clip["file"] for clip in clips]
    for clip, path in zip(clips, paths, strict=True):
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]:
            report(f"Accuracy check: FAILED, {path.name} is missing or does not match its hash")
            return {"passed": False, "reason": f"{path.name} missing or modified"}

    transcriber = OfflineTranscriber(prepared.model, prepared.configuration, prepared.inference)
    transcriber.transcribe(paths)  # warm-up: first-use kernel selection
    started = time.perf_counter()
    result = transcriber.transcribe(paths)
    seconds = time.perf_counter() - started

    by_name = {file_result.path.name: file_result for file_result in result.files}
    per_clip = []
    for clip in clips:
        file_result = by_name[clip["file"]]
        same = file_result.transcript == clip["expected_transcript"]
        per_clip.append(
            {
                "id": clip["id"],
                "status": file_result.status.value,
                "same_as_reference_gpu": same,
                "transcript": file_result.transcript,
            }
        )
        report(
            f"Clip {clip['id']}: {file_result.status.value}, "
            f"{'same as on the reference GPU' if same else 'differs from the reference GPU'}"
        )

    wer = word_error_rate(
        [clip["reference_text"] for clip in clips],
        [by_name[clip["file"]].transcript for clip in clips],
    )
    all_ok = all(by_name[clip["file"]].status is FileStatus.OK for clip in clips)
    passed = all_ok and wer <= ACCURACY_WER_BUDGET
    report(
        f"Accuracy check: {'passed' if passed else 'FAILED'}, word error rate {100 * wer:.2f} percent "
        f"(budget {100 * ACCURACY_WER_BUDGET:.1f}), {sum(entry['same_as_reference_gpu'] for entry in per_clip)} "
        f"of {len(clips)} transcripts identical to the reference GPU"
    )
    audio_seconds = sum(clip["duration_seconds"] for clip in clips)
    report(f"Clips: {audio_seconds:.1f} s of audio transcribed in {seconds:.3f} s")
    return {
        "passed": passed,
        "word_error_rate": wer,
        "budget": ACCURACY_WER_BUDGET,
        "clips": per_clip,
        "audio_seconds": audio_seconds,
        "processing_seconds": seconds,
    }


# =============================================================================
# Speed
# =============================================================================


def measure_encoder_throughput(prepared: PreparedModel, report: Callable[[str], None]) -> dict[str, Any]:
    """
    Full 15-second chunks the encoder processes per second, at the run's batch rows.

    The encoder is most of the model's time, and this number does not depend
    on audio content or file layout, so it compares GPUs directly.
    """

    model = prepared.model
    configuration = prepared.configuration
    inference = prepared.inference
    device = next(model.parameters()).device
    rows = inference.batch_size if device.type != "cpu" else min(inference.batch_size, CPU_THROUGHPUT_ROWS)

    sample_rate = int(configuration.feature_extractor["sampling_rate"])
    hop_length = int(configuration.feature_extractor["hop_length"])
    samples = (inference.max_chunk_feature_frames - 1) * hop_length
    generator = torch.Generator().manual_seed(0)
    waveform = torch.randn(samples, generator=generator) * THROUGHPUT_AMPLITUDE
    features, mask = ParakeetFeatureExtractor(configuration)([waveform] * rows, [sample_rate] * rows, device)

    with torch.inference_mode():
        model.encode(features, mask)
        synchronize(device)
        started = time.perf_counter()
        for _ in range(THROUGHPUT_TIMED_BATCHES):
            model.encode(features, mask)
        synchronize(device)
        seconds = time.perf_counter() - started

    chunks_per_second = rows * THROUGHPUT_TIMED_BATCHES / seconds
    audio_per_second = chunks_per_second * samples / sample_rate
    report(
        f"Encoder throughput: {chunks_per_second:.1f} full chunks per second "
        f"({audio_per_second:.0f} s of audio per second) at {rows} rows"
    )
    return {
        "rows": rows,
        "chunks_per_second": chunks_per_second,
        "audio_seconds_per_second": audio_per_second,
    }


# =============================================================================
# Command
# =============================================================================


def run_doctor(settings: Settings, report: Callable[[str], None], run_id: str) -> tuple[dict[str, Any], Path]:
    """
    Run every check and append the JSON record.

    Returns:
        The record and the JSONL file it was appended to.
    """

    facts = environment_facts()
    for line in environment_lines(facts):
        report(line)

    prepared = prepare_inference_model(settings, report)
    accuracy = check_accuracy(prepared, report)
    speed = measure_encoder_throughput(prepared, report)

    record: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_revision(settings.benchmark.git_timeout_seconds),
        "environment": facts,
        "decisions": list(prepared.decisions),
        "memory_budget": list(prepared.memory_budget.lines()),
        "accuracy": accuracy,
        "encoder_throughput": speed,
    }
    settings.paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.paths.metrics_dir / f"doctor_{datetime.now():%Y-%m-%d}.jsonl"
    with output_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, default=str) + "\n")
    return record, output_path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """No options: everything comes from config.toml."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the checks; return 0 when the accuracy check passed, else 1."""

    parse_arguments(argv)
    settings = load_settings()
    run_id, log_path = configure_run_logging(settings.paths.log_dir, settings.logging.level)
    report = line_reporter(logger)
    report(f"Log file: {log_path}")

    try:
        record, output_path = run_doctor(settings, report, run_id)
    except Exception as error:
        logger.exception("doctor failed")
        print(f"Error: {error}", file=sys.stderr)
        print(f"Details: {log_path} (run {run_id})", file=sys.stderr)
        return 1

    report(f"Results: {output_path}")
    return 0 if record["accuracy"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
