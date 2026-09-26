"""
Parakeet audio preprocessing implemented with native PyTorch.

The processor converts mono waveforms into normalized log-mel features and
attention masks. Every numeric processing parameter comes from
``ParakeetConfig.feature_extractor`` or its encoder configuration. This module
contains no tokenizer, model, checkpoint, or reporting responsibilities.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch

from ..configuration.config import ParakeetConfig


# Constants of the reference ParakeetFeatureExtractor. They are part of the
# checkpoint's input contract, not tunable settings: changing either shifts
# every feature the model was trained on.
LOG_GUARD = 2**-24
NORMALIZATION_EPSILON = 1e-5


# =============================================================================
# Slaney mel-scale utilities
# =============================================================================
# The original processor constructs its filter bank with librosa's default
# Slaney scale and Slaney area normalization. Reproducing these formulas here
# is required for feature parity; using the commonly seen HTK formula would
# produce similarly shaped but numerically different features.
# =============================================================================


def hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
    """Convert frequency in Hz to the piecewise Slaney mel scale."""

    minimum_log_frequency = 1000.0
    linear_step = 200.0 / 3.0
    log_step = 27.0 / math.log(6.4)
    linear_mel = frequency / linear_step
    log_mel = 15.0 + torch.log(frequency / minimum_log_frequency) * log_step
    return torch.where(frequency < minimum_log_frequency, linear_mel, log_mel)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """Convert Slaney mel values back to frequency in Hz."""

    minimum_log_frequency = 1000.0
    linear_step = 200.0 / 3.0
    log_step = 27.0 / math.log(6.4)
    linear_frequency = mel * linear_step
    log_frequency = minimum_log_frequency * torch.exp((mel - 15.0) / log_step)
    return torch.where(mel < 15.0, linear_frequency, log_frequency)


def build_mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    minimum_frequency: float,
    maximum_frequency: float,
) -> torch.Tensor:
    """
    Build Slaney-normalized triangular mel filters.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_fft: FFT size. The frequency axis has ``n_fft // 2 + 1`` bins.
        n_mels: Number of triangular filters.
        minimum_frequency: Lower frequency boundary in Hz.
        maximum_frequency: Upper frequency boundary in Hz.

    Returns:
        Float32 tensor with shape ``(n_mels, n_fft // 2 + 1)``.
    """

    frequency_grid = torch.linspace(
        0.0,
        sample_rate / 2.0,
        n_fft // 2 + 1,
        dtype=torch.float64,
    )
    mel_minimum = hz_to_mel(torch.tensor(minimum_frequency, dtype=torch.float64))
    mel_maximum = hz_to_mel(torch.tensor(maximum_frequency, dtype=torch.float64))
    mel_points = mel_to_hz(
        torch.linspace(mel_minimum, mel_maximum, n_mels + 2, dtype=torch.float64)
    )

    lower = mel_points[:-2, None]
    center = mel_points[1:-1, None]
    upper = mel_points[2:, None]
    rising = (frequency_grid[None, :] - lower) / (center - lower)
    falling = (upper - frequency_grid[None, :]) / (upper - center)
    filters = torch.maximum(torch.zeros_like(rising), torch.minimum(rising, falling))

    # Area normalization makes each filter integrate to one in frequency units.
    filters = filters * (2.0 / (upper - lower))
    return filters.to(torch.float32)


# =============================================================================
# Feature extraction
# =============================================================================
# The tensor path mirrors the reference processor: preemphasis, periodic-false
# Hann STFT with constant padding, squared magnitude, mel projection, guarded
# logarithm, valid-frame mask, and per-recording normalization. Padding never
# contributes to mean or variance statistics.
# =============================================================================


class ParakeetFeatureExtractor:
    """Compute padded normalized Parakeet log-mel features with torch only."""

    def __init__(self, configuration: ParakeetConfig):
        """Initialize extraction parameters from checkpoint JSON."""

        feature = configuration.feature_extractor
        self.feature_size = feature["feature_size"]
        self.sample_rate = feature["sampling_rate"]
        self.hop_length = feature["hop_length"]
        self.n_fft = feature["n_fft"]
        self.win_length = feature["win_length"]
        self.preemphasis = feature["preemphasis"]
        self.mel_filters_cpu = build_mel_filter_bank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.feature_size,
            minimum_frequency=0.0,
            maximum_frequency=self.sample_rate / 2.0,
        )
        self._window_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self._mel_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def _runtime_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return one cached Hann window for a device/dtype pair."""

        key = (str(device), dtype)
        window = self._window_cache.get(key)
        if window is None:
            window = torch.hann_window(
                self.win_length,
                periodic=False,
                dtype=dtype,
                device=device,
            )
            self._window_cache[key] = window
        return window

    def _runtime_mel_filters(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return one cached mel filter bank for a device/dtype pair."""

        key = (str(device), dtype)
        mel_filters = self._mel_cache.get(key)
        if mel_filters is None:
            mel_filters = self.mel_filters_cpu.to(device=device, dtype=dtype)
            self._mel_cache[key] = mel_filters
        return mel_filters

    def __call__(
        self,
        waveforms: Iterable[torch.Tensor],
        sample_rates: Iterable[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a padded batch.

        Args:
            waveforms: Iterable of one-dimensional waveform tensors.
            sample_rates: Sample rate for each waveform.
            device: Target device for preprocessing tensors.

        Returns:
            ``(features, attention_mask)`` where features have shape
            ``(B, T, M)`` and the boolean mask has shape ``(B, T)``.
        """

        waveform_list = [
            waveform.to(device=device, dtype=torch.float32)
            for waveform in waveforms
        ]
        sample_rate_list = list(sample_rates)
        if not waveform_list:
            raise ValueError("At least one waveform is required")
        if len(waveform_list) != len(sample_rate_list):
            raise ValueError("Each waveform must have one sample rate")
        if any(rate != self.sample_rate for rate in sample_rate_list):
            raise ValueError(f"All audio must use sample rate {self.sample_rate}")

        audio_lengths = torch.tensor(
            [waveform.numel() for waveform in waveform_list],
            dtype=torch.long,
            device=device,
        )
        # At least one (padding) sample keeps torch.stft defined when every
        # waveform in the batch is empty; the mask below still marks no frame
        # of such a row as valid.
        max_audio_length = max(1, int(audio_lengths.max().item()))
        padded_audio = torch.zeros(
            len(waveform_list),
            max_audio_length,
            dtype=torch.float32,
            device=device,
        )
        for index, waveform in enumerate(waveform_list):
            padded_audio[index, : waveform.numel()] = waveform

        # Preemphasis is applied before STFT, then padded samples are cleared so
        # the artificial tail cannot create energy in valid-frame statistics.
        time_mask = torch.arange(max_audio_length, device=device)[None, :] < audio_lengths[:, None]
        if self.preemphasis is not None:
            padded_audio = torch.cat(
                [
                    padded_audio[:, :1],
                    padded_audio[:, 1:] - self.preemphasis * padded_audio[:, :-1],
                ],
                dim=1,
            )
            padded_audio = padded_audio.masked_fill(~time_mask, 0.0)

        window = self._runtime_window(device, padded_audio.dtype)
        stft = torch.stft(
            padded_audio,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        magnitude = torch.view_as_real(stft)
        magnitude = torch.sqrt(magnitude.pow(2).sum(dim=-1)).pow(2)

        mel_features = self._runtime_mel_filters(device, magnitude.dtype) @ magnitude
        mel_features = torch.log(mel_features + LOG_GUARD)
        mel_features = mel_features.permute(0, 2, 1)

        feature_lengths = torch.floor_divide(
            audio_lengths + self.n_fft // 2 * 2 - self.n_fft,
            self.hop_length,
        )
        attention_mask = (
            torch.arange(mel_features.shape[1], device=device)[None, :]
            < feature_lengths[:, None]
        )

        # Normalize per recording, not across the padded batch. This preserves
        # the reference behavior for mixed-duration inference.
        #
        # The reference divides by n (mean) and n - 1 (variance). Rows with
        # fewer than two valid frames would divide by zero and send NaN into
        # the encoder, so the divisors are clamped to one. For n >= 2 the
        # result is bit-identical to the reference; for n <= 1 the features
        # become zeros, which the model decodes as silence.
        mask = attention_mask.unsqueeze(-1)
        mean_divisor = feature_lengths.clamp(min=1).unsqueeze(-1)
        variance_divisor = (feature_lengths - 1).clamp(min=1).unsqueeze(-1)

        masked_features = mel_features * mask
        mean = (masked_features.sum(dim=1) / mean_divisor).unsqueeze(1)
        squared_deviation = ((masked_features - mean) ** 2) * mask
        variance = squared_deviation.sum(dim=1) / variance_divisor
        standard_deviation = torch.sqrt(variance).unsqueeze(1)

        mel_features = (mel_features - mean) / (standard_deviation + NORMALIZATION_EPSILON)
        mel_features = mel_features * mask

        return mel_features, attention_mask
