"""
Shared pytest configuration.

Tiers are assigned by directory, so a test cannot land in the wrong tier by a
forgotten decorator:

    tests/unit/        -> unit
    tests/regression/  -> regression
    tests/full/        -> full (real checkpoint; skipped when not prepared)

Run one tier with ``-m unit``, ``-m regression``, ``-m full``, or combine
them, for example ``-m "not full"`` for the fast suite.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from src.configuration.config import ParakeetConfig, load_config
from src.configuration.settings import Settings, load_settings
from support import SpeechClip, load_speech_clips, write_tiny_checkpoint

TIER_DIRECTORIES = ("unit", "regression", "full")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every test with its tier and skip CUDA tests on machines without CUDA."""

    no_cuda = pytest.mark.skip(reason="needs a CUDA device")
    for item in items:
        parts = Path(str(item.path)).parts
        for tier in TIER_DIRECTORIES:
            if tier in parts:
                item.add_marker(getattr(pytest.mark, tier))
        if item.get_closest_marker("cuda") and not torch.cuda.is_available():
            item.add_marker(no_cuda)


# =============================================================================
# Tiny checkpoint fixtures (unit and regression tiers)
# =============================================================================


@pytest.fixture
def tiny_checkpoint_dir(tmp_path: Path) -> Path:
    """A schema-valid checkpoint directory for a model small enough for CPU tests."""

    return write_tiny_checkpoint(tmp_path / "checkpoint")


@pytest.fixture
def tiny_configuration(tiny_checkpoint_dir: Path) -> ParakeetConfig:
    """Validated configuration for the tiny checkpoint."""

    return load_config(tiny_checkpoint_dir)


# =============================================================================
# Real checkpoint fixtures (full tier)
# =============================================================================


@pytest.fixture(scope="session")
def real_settings() -> Settings:
    """
    Repository settings, or skip the full tier when the checkpoint is not ready.

    The full tier needs the prepared 2.5 GB checkpoint. It is prepared by the
    first `inference.py` run or by `python -m src.commands.prepare_checkpoint`;
    the tests never download it themselves.
    """

    settings = load_settings()
    if not (settings.paths.weights_dir / ".ready").is_file():
        pytest.skip(
            f"checkpoint not prepared in {settings.paths.weights_dir}; "
            "run `python -m src.commands.prepare_checkpoint` first"
        )
    return dataclasses.replace(settings, inference=fixed_full_tier_inference(settings.inference))


# Rows per batch for the full tier. The expected transcripts in
# tests/data/librispeech were recorded with 16-row batches, and float16
# results can move by a frame with the batch shape, so the tier pins the
# shape instead of taking what "auto" picks on the machine at hand. The
# automatic decisions have their own test (test_device_adaptation_gpu.py).
FULL_TIER_BATCH_ROWS = 16


def fixed_full_tier_inference(inference):
    """The repository settings with every "auto" replaced by a fixed value."""

    from src.runtime.device import default_decode_workers, encoder_dtype_for, select_device

    encoder_dtype, _description = encoder_dtype_for(inference.encoder_precision, select_device())
    rows = inference.batch_size if inference.batch_size is not None else FULL_TIER_BATCH_ROWS
    return dataclasses.replace(
        inference,
        batch_size=rows,
        max_batch_feature_frames=(
            inference.max_batch_feature_frames
            if inference.max_batch_feature_frames is not None
            else rows * inference.max_chunk_feature_frames
        ),
        encoder_precision="float16" if encoder_dtype == torch.float16 else "float32",
        # The expected transcripts predate float16 accumulation.
        float16_accumulation=bool(inference.float16_accumulation),
        decode_workers=(
            inference.decode_workers if inference.decode_workers is not None else default_decode_workers()
        ),
    )


@pytest.fixture(scope="session")
def real_model(real_settings: Settings):
    """Load the full-weight model once per test session on the auto-selected device."""

    from src.commands.model_loading import inference_model_loader
    from src.runtime.device import apply_float32_matmul_precision, encoder_dtype_for, select_device

    # The configured precision, so the full tier checks what a run would use.
    device = select_device()
    apply_float32_matmul_precision(real_settings.inference.float32_matmul_precision)
    encoder_dtype, _description = encoder_dtype_for(real_settings.inference.encoder_precision, device)
    loader = inference_model_loader(device, encoder_dtype)
    model, configuration, _metadata = loader(real_settings.paths.weights_dir)
    from src.inference.offline import enable_graph_decoding

    enable_graph_decoding(model, real_settings.inference)
    return model, configuration


@pytest.fixture(scope="session")
def speech_clips() -> tuple[SpeechClip, ...]:
    """Committed LibriSpeech clips, hash-verified against manifest.json."""

    return load_speech_clips()
