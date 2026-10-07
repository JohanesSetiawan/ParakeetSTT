"""The automatic batch budget on the real model: what it picks must actually fit."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from src.audio.features import ParakeetFeatureExtractor
from src.configuration.settings import MemorySettings
from src.inference.budget import resolve_memory_budget


@pytest.fixture
def uncapped_after(real_model):
    """The budget caps the allocator for the whole process; lift it after the test."""

    yield
    device = next(real_model[0].parameters()).device
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.0, device)


@pytest.mark.cuda
def test_chosen_budget_runs_under_the_ceiling(real_settings, real_model, uncapped_after) -> None:
    model, configuration = real_model
    device = next(model.parameters()).device
    max_chunk = real_settings.inference.max_chunk_feature_frames
    # Configure far more than a 4 GB card holds, so the budget has to lower it.
    oversized = dataclasses.replace(
        real_settings.inference,
        max_batch_feature_frames=64 * max_chunk,
        batch_size=64,
    )
    memory = MemorySettings(cap_to_free_memory=True, auto_batch_budget=True, reserve_mib=256)

    resolved, budget = resolve_memory_budget(model, configuration, oversized, memory)

    assert budget.applied
    assert budget.chunks_that_fit is not None and budget.chunks_that_fit >= 1
    assert resolved.max_batch_feature_frames == min(
        oversized.max_batch_feature_frames,
        budget.chunks_that_fit * max_chunk,
    )
    assert resolved.max_batch_feature_frames >= max_chunk

    # A batch of exactly the chosen number of full chunks must run without
    # running out of memory and stay under the ceiling.
    rows = resolved.max_batch_feature_frames // max_chunk
    hop_length = int(configuration.feature_extractor["hop_length"])
    generator = torch.Generator().manual_seed(3)
    waveform = torch.randn((max_chunk - 1) * hop_length, generator=generator) * 0.05
    extractor = ParakeetFeatureExtractor(configuration)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        features, mask = extractor([waveform] * rows, [16000] * rows, device)
        model.generate(features, mask)
    torch.cuda.synchronize(device)

    assert torch.cuda.max_memory_reserved(device) <= budget.ceiling_bytes


@pytest.mark.cuda
def test_budget_never_raises_the_configured_value(real_settings, real_model, uncapped_after) -> None:
    model, configuration = real_model
    max_chunk = real_settings.inference.max_chunk_feature_frames
    small = dataclasses.replace(real_settings.inference, max_batch_feature_frames=max_chunk)
    memory = MemorySettings(cap_to_free_memory=True, auto_batch_budget=True, reserve_mib=256)

    resolved, budget = resolve_memory_budget(model, configuration, small, memory)

    assert resolved.max_batch_feature_frames == max_chunk
    assert budget.configured_batch_feature_frames == max_chunk
