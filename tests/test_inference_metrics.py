"""Regression tests for user-facing inference runtime metrics."""

from __future__ import annotations

import unittest

import torch

from src.inference.service import TranscriptionBatch
from src.models.parakeet import GenerationResult


class InferenceMetricTests(unittest.TestCase):
    """Verify metric formulas exposed by TranscriptionBatch."""

    def test_real_time_factor_and_throughput(self) -> None:
        """RTF is elapsed/audio duration and throughput is its reciprocal."""

        batch = TranscriptionBatch(
            file_names=("sample.wav",),
            audio_durations_seconds=(10.0,),
            input_features=torch.zeros(1, 1, 1),
            attention_mask=torch.ones(1, 1, dtype=torch.bool),
            generation=GenerationResult(
                sequences=torch.zeros(1, 1, dtype=torch.long),
                durations=torch.zeros(1, 1, dtype=torch.long),
            ),
            transcripts=("sample",),
            elapsed_seconds=2.0,
        )

        self.assertEqual(batch.total_audio_seconds, 10.0)
        self.assertEqual(batch.real_time_factor, 0.2)
        self.assertEqual(batch.audio_seconds_per_second, 5.0)


if __name__ == "__main__":
    unittest.main()
