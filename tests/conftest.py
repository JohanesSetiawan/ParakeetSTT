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
    return settings


@pytest.fixture(scope="session")
def real_model(real_settings: Settings):
    """Load the full-weight model once per test session on the auto-selected device."""

    from src.models.parakeet import load_model

    model, configuration, _metadata = load_model(real_settings.paths.weights_dir)
    return model, configuration


@pytest.fixture(scope="session")
def speech_clips() -> tuple[SpeechClip, ...]:
    """Committed LibriSpeech clips, hash-verified against manifest.json."""

    return load_speech_clips()
