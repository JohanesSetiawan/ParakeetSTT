"""
Application service for standalone Parakeet WAV transcription.

This module coordinates domain components but does not implement their math. It
loads audio through ``audio.py``, extracts features through ``processing.py``,
invokes ``ParakeetTDT.generate``, and decodes IDs through ``tokenization.py``.
Reporting, CLI parsing, and benchmark persistence remain in the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import torch

from ..audio.reader import read_wav
from ..audio.features import ParakeetFeatureExtractor
from ..configuration.config import ParakeetConfig
from ..models.parakeet import GenerationResult, ParakeetTDT
from ..text.tokenization import BpeTokenizer


@dataclass(frozen=True)
class TranscriptionBatch:
    """Measured tensors, text, and runtime metrics produced by one batch."""

    file_names: tuple[str, ...]
    audio_durations_seconds: tuple[float, ...]
    input_features: torch.Tensor
    attention_mask: torch.Tensor
    generation: GenerationResult
    transcripts: tuple[str, ...]
    elapsed_seconds: float

    @property
    def total_audio_seconds(self) -> float:
        """Return total source audio duration represented by this batch."""

        return sum(self.audio_durations_seconds)

    @property
    def real_time_factor(self) -> float:
        """
        Return wall-clock processing time divided by source audio duration.

        Values below 1.0 indicate faster-than-real-time end-to-end processing;
        the metric includes feature extraction, model generation, and decoding.
        """

        if self.total_audio_seconds <= 0:
            return 0.0
        return self.elapsed_seconds / self.total_audio_seconds

    @property
    def audio_seconds_per_second(self) -> float:
        """Return the inverse throughput metric in source-audio seconds/second."""

        if self.elapsed_seconds <= 0:
            return 0.0
        return self.total_audio_seconds / self.elapsed_seconds


class Transcriber:
    """Coordinate native processing, TDT generation, and detokenization."""

    def __init__(
        self,
        model: ParakeetTDT,
        configuration: ParakeetConfig,
    ) -> None:
        """Bind one loaded model to its checkpoint-specific processors."""

        self.model = model
        self.configuration = configuration
        self.feature_extractor = ParakeetFeatureExtractor(configuration)
        self.tokenizer = BpeTokenizer(
            configuration.tokenizer_path,
            configuration.pad_token_id,
            configuration.blank_token_id,
        )

    @property
    def device(self) -> torch.device:
        """Return the device of the loaded model parameters."""

        return next(self.model.parameters()).device

    def transcribe(self, audio_paths: Iterable[Path]) -> TranscriptionBatch:
        """
        Transcribe an explicit collection as one padded mixed-duration batch.

        Args:
            audio_paths: WAV files in the desired output order.

        Returns:
            Batch object containing input tensors, raw generation output, and
            decoded transcripts. Tensors remain on the model device so callers
            can inspect them without hidden host-device copies.
        """

        paths = tuple(audio_paths)
        if not paths:
            raise ValueError("At least one audio path is required")

        waveforms: list[torch.Tensor] = []
        sample_rates: list[int] = []
        durations: list[float] = []
        for audio_path in paths:
            waveform, sample_rate = read_wav(audio_path)
            waveforms.append(waveform)
            sample_rates.append(sample_rate)
            durations.append(waveform.numel() / sample_rate)

        started = time.perf_counter()
        input_features, attention_mask = self.feature_extractor(
            waveforms,
            sample_rates,
            self.device,
        )
        generation = self.model.generate(input_features, attention_mask)
        transcripts = tuple(self.tokenizer.batch_decode(generation.sequences))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_seconds = time.perf_counter() - started

        return TranscriptionBatch(
            file_names=tuple(path.name for path in paths),
            audio_durations_seconds=tuple(durations),
            input_features=input_features,
            attention_mask=attention_mask,
            generation=generation,
            transcripts=transcripts,
            elapsed_seconds=elapsed_seconds,
        )