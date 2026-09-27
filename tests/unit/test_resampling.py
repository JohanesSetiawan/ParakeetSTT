"""Anti-aliased resampling on a global grid (src/audio/resampling.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.audio.resampling import plan_block, reduced_rates, resample_block

TARGET = 16000


def tone(frequency: float, rate: int, seconds: float = 1.0) -> torch.Tensor:
    time = torch.arange(int(rate * seconds), dtype=torch.float64) / rate
    return torch.sin(2 * math.pi * frequency * time).to(torch.float32)


def resample_whole(signal: torch.Tensor, rate: int) -> torch.Tensor:
    total = round(signal.numel() * TARGET / rate)
    plan = plan_block(0, total, rate, TARGET)
    block = torch.nn.functional.pad(signal, (-plan.block_start, plan.block_end - signal.numel()))
    return resample_block(block, plan, rate, TARGET)


def rms(signal: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(signal.double() ** 2)))


@pytest.mark.parametrize("rate", [48000, 44100, 22050])
def test_speech_band_passes_with_unity_gain(rate: int) -> None:
    output = resample_whole(tone(1000.0, rate), rate)

    middle = output[2000:-2000]  # away from the zero-padded file edges
    assert rms(middle) == pytest.approx(1 / math.sqrt(2), rel=0.01)


@pytest.mark.parametrize(("rate", "frequency"), [(48000, 10_000.0), (44100, 12_000.0), (48000, 20_000.0)])
def test_content_above_the_target_nyquist_is_removed_not_aliased(rate: int, frequency: float) -> None:
    """Linear interpolation folded these tones back into the 0-8 kHz speech band."""

    output = resample_whole(tone(frequency, rate), rate)

    attenuation_db = 20 * math.log10(rms(output[2000:-2000]) / (1 / math.sqrt(2)))
    assert attenuation_db < -40, f"{frequency} Hz only attenuated by {attenuation_db:.1f} dB"


@pytest.mark.parametrize("rate", [48000, 44100, 22050, 8000])
def test_any_interval_matches_the_whole_file(rate: int) -> None:
    signal = torch.from_numpy(np.random.default_rng(1).uniform(-0.5, 0.5, rate * 2).astype(np.float32))
    whole = resample_whole(signal, rate)

    for start, end in [(0, 1000), (777, 5432), (15_000, whole.numel())]:
        plan = plan_block(start, end, rate, TARGET)
        block = torch.nn.functional.pad(signal, (0, max(0, plan.block_end - signal.numel())))
        block = block[max(0, plan.block_start) : plan.block_end]
        block = torch.nn.functional.pad(block, (max(0, -plan.block_start), 0))

        piece = resample_block(block, plan, rate, TARGET)

        assert torch.allclose(piece, whole[start:end], atol=1e-6), (start, end)


def test_block_starts_on_the_polyphase_period() -> None:
    reduced_orig, _ = reduced_rates(44100, TARGET)

    for start in (0, 1, 159, 160, 12_345):
        assert plan_block(start, start + 500, 44100, TARGET).block_start % reduced_orig == 0


def test_wrong_block_length_is_rejected() -> None:
    plan = plan_block(0, 100, 48000, TARGET)

    with pytest.raises(ValueError, match="plan expects"):
        resample_block(torch.zeros(3), plan, 48000, TARGET)
