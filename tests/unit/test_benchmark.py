"""Development benchmark command and host memory measurement."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import torch

from src.checkpoint.bootstrap import BootstrapResult
from src.commands.benchmark import format_optional, round_record, run_benchmark, summarize
from src.configuration.settings import BenchmarkSettings, load_settings
from src.runtime.memory import peak_process_memory_bytes
from support import SteadySpeechModel, inference_settings, write_float_wav


def test_peak_process_memory_is_reported() -> None:
    peak = peak_process_memory_bytes()

    assert peak is not None and peak > 10 * 2**20


def test_summary_statistics() -> None:
    summary = summarize([1.0, 2.0, 3.0])

    assert summary == {"mean": 2.0, "min": 1.0, "max": 3.0, "stdev": 1.0}
    assert summarize([4.0])["stdev"] == 0.0


def test_round_without_audio_reports_no_real_time_factor() -> None:
    """A ratio over zero audio seconds is not invented, and printing it does not crash."""

    from types import SimpleNamespace

    empty_round = SimpleNamespace(
        total_audio_seconds=0.0,
        media_decode_seconds=0.0,
        feature_seconds=0.0,
        generation_seconds=0.0,
        recovery_seconds=0.0,
        plan=SimpleNamespace(items=(), batches=()),
        peak_memory={},
        files=(),
    )

    record = round_record(empty_round, wall_seconds=0.5)

    assert record["real_time_factor"] is None
    assert format_optional(record["real_time_factor"], 5) == "not available"
    assert format_optional(0.0123456, 5) == "0.01235"


def test_benchmark_warms_up_measures_and_appends_one_json_line(tmp_path: Path, tiny_configuration) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    for index in range(2):
        write_float_wav(audio / f"clip_{index}.wav", torch.full((16000,), 0.1), 16000)
    base = load_settings()
    settings = dataclasses.replace(
        base,
        paths=dataclasses.replace(base.paths, metrics_dir=tmp_path / "metrics"),
        inference=inference_settings(),
        benchmark=BenchmarkSettings(warmup_rounds=1, measured_rounds=2),
    )
    model = SteadySpeechModel(3)
    bootstrap = BootstrapResult(action="ready", marker_path="marker", preparation=None)

    with patch(
        "src.commands.benchmark.ensure_first_run_ready",
        return_value=(bootstrap, (model.eval(), tiny_configuration, {})),
    ):
        record, output_path = run_benchmark(audio, settings, run_id="test-run")
        run_benchmark(audio, settings, run_id="second-run")

    lines = output_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert len(lines) == 2
    assert first["run_id"] == "test-run"
    assert first["input"]["files"] == 2
    assert len(first["rounds"]) == 2
    # Two benchmark runs x (1 warm-up + 2 measured rounds) x 2 files: these
    # 1 s clips do not fit one batch under the tiny budget, so each file is
    # its own generate call.
    assert model.calls == 2 * (1 + 2) * 2
    assert first["summary"]["transcripts_identical_across_rounds"] is True
    assert {file["name"] for file in first["rounds"][0]["files"]} == {"clip_0.wav", "clip_1.wav"}
    assert all(file["processing_seconds"] > 0 for file in first["rounds"][0]["files"])
    assert set(first["git"]) == {"commit", "dirty"}
    assert record["settings"]["benchmark"] == {"warmup_rounds": 1, "measured_rounds": 2}
