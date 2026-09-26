"""Focused tests for offline workload planning."""

from __future__ import annotations

from pathlib import Path
import unittest

from src.inference.planning import (
    AudioMetadata,
    build_execution_plan,
)


class LongFormPlanningTests(unittest.TestCase):
    """Verify cost-aware planning without constructing the model checkpoint."""

    def test_planner_uses_frame_budget_and_interleaves_files(self) -> None:
        """Mixed workloads are bounded and long files do not monopolize the queue."""

        metadata = (
            AudioMetadata(Path("short.wav"), 16000, 1, 16000, "WAV"),
            AudioMetadata(Path("long.wav"), 16000, 1, 16000 * 120, "WAV"),
        )

        plan = build_execution_plan(
            metadata,
            target_sample_rate=16000,
            n_fft=512,
            hop_length=160,
            max_chunk_feature_frames=500,
            overlap_feature_frames=10,
            batch_size=2,
            max_batch_feature_frames=700,
            max_padding_fraction=0.5,
        )

        self.assertGreater(len(plan.items), 2)
        self.assertEqual(plan.items[0].file_index, 0)
        self.assertEqual(plan.items[1].file_index, 1)
        self.assertTrue(
            all(
                sum(item.feature_frames for item in batch)
                <= plan.max_batch_feature_frames
                for batch in plan.batches
            )
        )
        self.assertTrue(all(item.feature_frames <= 500 for item in plan.items))
        self.assertTrue(
            all(
                item.feature_frames <= 500
                for item in plan.items
            )
        )

if __name__ == "__main__":
    unittest.main()