"""
Startup's automatic decisions on the real model and GPU, and the doctor command.

Both run in-process on the session model (the startup loader is patched to
return it), so the GPU never holds two models at once.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.audio.features import ParakeetFeatureExtractor
from src.checkpoint.bootstrap import BootstrapResult
from src.commands.model_loading import prepare_inference_model
from src.configuration.settings import load_settings
from src.inference.offline import enable_graph_decoding

LOADER = "src.commands.model_loading.ensure_first_run_ready"


@pytest.fixture
def restored_session_model(real_settings, real_model):
    """
    Startup changes process-wide state; put the session back afterwards.

    It caps the allocator, sets the float16 accumulation switch, and rebuilds
    the graph decoder for its own rows. The later full-tier tests expect the
    uncapped allocator, the switch off, and the fixed 16-row graph decoder.
    """

    model, _configuration = real_model
    device = next(model.parameters()).device
    accumulation = torch.backends.cuda.matmul.allow_fp16_accumulation
    # A freshly loaded model has no graph decoder when the budget is measured.
    model.graph_decoder = None
    yield
    torch.backends.cuda.matmul.allow_fp16_accumulation = accumulation
    model.graph_decoder = None
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.0, device)
    enable_graph_decoding(model, real_settings.inference)


def prepare_with_session_model(settings, real_model, lines: list[str]):
    model, configuration = real_model
    bootstrap = BootstrapResult(action="ready", marker_path="marker", preparation=None)
    with patch(LOADER, return_value=(bootstrap, (model, configuration, {}))):
        return prepare_inference_model(settings, lines.append)


@pytest.mark.cuda
def test_auto_settings_are_decided_for_this_gpu_and_fit(real_model, restored_session_model) -> None:
    settings = load_settings()  # the shipped "auto" configuration
    lines: list[str] = []

    prepared = prepare_with_session_model(settings, real_model, lines)

    inference = prepared.inference
    inference.require_resolved()
    budget = prepared.memory_budget
    assert budget.applied and budget.chunks_that_fit is not None
    assert 1 <= inference.batch_size <= budget.chunks_that_fit
    assert inference.max_batch_feature_frames == inference.batch_size * inference.max_chunk_feature_frames
    assert prepared.graph_decoding and prepared.model.graph_decoder.rows == inference.batch_size
    assert prepared.model.graph_decoder.graph is not None  # captured at startup
    # Captured in inference mode: no autograd state in the decoder's buffers.
    assert not prepared.model.graph_decoder.hidden_state.requires_grad
    assert inference.encoder_precision == "float16"
    assert any(line.startswith("Batch rows:") and "auto" in line for line in prepared.decisions)
    assert any(line.startswith("Float16 accumulation:") and "auto" in line for line in prepared.decisions)

    # A batch of the chosen rows of full chunks runs under the ceiling.
    model, configuration = prepared.model, prepared.configuration
    device = next(model.parameters()).device
    rows = inference.batch_size
    hop_length = int(configuration.feature_extractor["hop_length"])
    generator = torch.Generator().manual_seed(5)
    waveform = torch.randn((inference.max_chunk_feature_frames - 1) * hop_length, generator=generator) * 0.05
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        features, mask = ParakeetFeatureExtractor(configuration)([waveform] * rows, [16000] * rows, device)
        model.generate(features, mask)
    torch.cuda.synchronize(device)

    assert torch.cuda.max_memory_reserved(device) <= budget.ceiling_bytes


@pytest.mark.cuda
def test_configured_numbers_are_kept_within_the_budget(real_model, restored_session_model) -> None:
    base = load_settings()
    settings = dataclasses.replace(
        base,
        inference=dataclasses.replace(
            base.inference,
            batch_size=4,
            max_batch_feature_frames=None,
            float16_accumulation=False,
            decode_workers=2,
            encoder_precision="float16",
        ),
    )

    prepared = prepare_with_session_model(settings, real_model, [])

    assert prepared.inference.batch_size == 4
    assert prepared.inference.max_batch_feature_frames == 4 * base.inference.max_chunk_feature_frames
    assert prepared.inference.decode_workers == 2
    assert prepared.float16_accumulation == "off"
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is False


@pytest.mark.cuda
def test_doctor_passes_and_records_the_machine(real_model, restored_session_model, tmp_path: Path) -> None:
    from src.commands.doctor import run_doctor

    base = load_settings()
    settings = dataclasses.replace(base, paths=dataclasses.replace(base.paths, metrics_dir=tmp_path))
    model, configuration = real_model
    bootstrap = BootstrapResult(action="ready", marker_path="marker", preparation=None)
    lines: list[str] = []

    with patch(LOADER, return_value=(bootstrap, (model, configuration, {}))):
        record, output_path = run_doctor(settings, lines.append, run_id="doctor-test")

    assert record["accuracy"]["passed"] is True
    assert record["accuracy"]["word_error_rate"] <= record["accuracy"]["budget"]
    assert record["environment"]["build_supports_device"] is True
    assert record["encoder_throughput"]["chunks_per_second"] > 0
    assert any(line.startswith("Accuracy check: passed") for line in lines)
    assert any(line.startswith("Kernels for this GPU in the PyTorch build: yes") for line in lines)
    stored = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["run_id"] for entry in stored] == ["doctor-test"]


@pytest.mark.cuda
def test_a_decoder_captured_at_startup_matches_the_eager_loop(real_settings, real_model, speech_clips) -> None:
    """Capturing on a placeholder batch at startup must replay exactly what the eager loop computes."""

    import soundfile

    model, configuration = real_model
    device = next(model.parameters()).device
    waveforms = [torch.from_numpy(soundfile.read(str(clip.path), dtype="float32")[0]) for clip in speech_clips]
    features, mask = ParakeetFeatureExtractor(configuration)(waveforms, [16000] * len(waveforms), device)

    graph_decoder = model.graph_decoder
    assert graph_decoder is not None and graph_decoder.graph is not None
    with torch.inference_mode():
        graphed = model.generate(features, mask)
        model.graph_decoder = None
        try:
            eager = model.generate(features, mask)
        finally:
            model.graph_decoder = graph_decoder

    assert torch.equal(graphed.sequences, eager.sequences)
    assert torch.equal(graphed.durations, eager.durations)
    assert torch.equal(graphed.frame_starts, eager.frame_starts)
