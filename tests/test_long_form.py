"""Tests for offline workload planning invariants."""

from __future__ import annotations

import random
import unittest
from pathlib import Path

from src.inference.planning import (
    AudioMetadata,
    build_execution_plan,
    estimate_feature_frames,
)

HOP_LENGTH = 160


def _plan(frame_counts, max_chunk, overlap, batch_size=4, max_batch=None, padding=0.5):
    metadata = tuple(
        AudioMetadata(Path(f"file_{index}.wav"), 16000, 1, frames, "WAV")
        for index, frames in enumerate(frame_counts)
    )
    return build_execution_plan(
        metadata,
        target_sample_rate=16000,
        hop_length=HOP_LENGTH,
        max_chunk_feature_frames=max_chunk,
        overlap_feature_frames=overlap,
        batch_size=batch_size,
        max_batch_feature_frames=max_batch or 2 * max_chunk,
        max_padding_fraction=padding,
    )


class LongFormPlanningTests(unittest.TestCase):
    """Verify cost-aware planning without constructing the model."""

    def test_planner_uses_frame_budget_and_interleaves_files(self) -> None:
        """Mixed workloads are bounded and long files do not monopolize the queue."""

        plan = _plan([16000, 16000 * 120], max_chunk=500, overlap=10, batch_size=2, max_batch=700)

        self.assertGreater(len(plan.items), 2)
        self.assertEqual(plan.items[0].file_index, 0)
        self.assertEqual(plan.items[1].file_index, 1)
        for batch in plan.batches:
            self.assertLessEqual(sum(item.feature_frames for item in batch), 700)
        self.assertTrue(all(item.feature_frames <= 500 for item in plan.items))

    def test_feature_estimate_matches_centered_stft_frame_count(self) -> None:
        """samples // hop + 1 is the tensor length torch.stft allocates."""

        for samples in (0, 1, 159, 160, 161, 239_999, 240_000):
            with self.subTest(samples=samples):
                self.assertEqual(
                    estimate_feature_frames(samples, 16000, 16000, HOP_LENGTH),
                    samples // HOP_LENGTH + 1,
                )

    def test_chunks_tile_every_file_within_budget(self) -> None:
        """
        For random lengths and overlaps: cores tile [0, N) exactly, sources wrap
        cores with at most the configured overlap, and no chunk exceeds budget.
        """

        generator = random.Random(1234)
        for _trial in range(200):
            max_chunk = generator.randint(20, 400)
            overlap = generator.randint(0, (max_chunk - 1) // 2)
            frames = generator.randint(0, 200_000)
            plan = _plan([frames], max_chunk=max_chunk, overlap=overlap)
            items = sorted(plan.items, key=lambda item: item.chunk_index)
            overlap_samples = overlap * HOP_LENGTH

            with self.subTest(max_chunk=max_chunk, overlap=overlap, frames=frames):
                self.assertEqual(items[0].core_start_frame, 0)
                self.assertEqual(items[-1].core_end_frame, frames)
                for previous, current in zip(items, items[1:]):
                    self.assertEqual(previous.core_end_frame, current.core_start_frame)
                for item in items:
                    self.assertLessEqual(item.feature_frames, max_chunk)
                    self.assertGreaterEqual(item.core_start_frame - item.source_start_frame, 0)
                    self.assertLessEqual(item.core_start_frame - item.source_start_frame, overlap_samples)
                    self.assertLessEqual(item.source_end_frame - item.core_end_frame, overlap_samples)
                    self.assertEqual(
                        item.feature_frames,
                        item.source_frame_count // HOP_LENGTH + 1,
                    )

    def test_zero_overlap_is_a_valid_plan(self) -> None:
        """Zero overlap yields back-to-back chunks with identical source/core."""

        plan = _plan([100_000], max_chunk=100, overlap=0)

        for item in plan.items:
            self.assertEqual(item.source_start_frame, item.core_start_frame)
            self.assertEqual(item.source_end_frame, item.core_end_frame)

    def test_overlap_must_leave_room_for_a_core(self) -> None:
        with self.assertRaises(ValueError):
            _plan([100_000], max_chunk=100, overlap=50)


if __name__ == "__main__":
    unittest.main()
