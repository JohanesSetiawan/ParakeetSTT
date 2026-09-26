"""Planner invariants: chunk budget, core tiling, scheduling, padded batches."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.inference.planning import (
    AudioMetadata,
    ExecutionPlan,
    build_execution_plan,
    estimate_feature_frames,
    padded_batch_frames,
)

HOP_LENGTH = 160
SAMPLE_RATE = 16000


def make_plan(
    frame_counts: list[int],
    *,
    max_chunk: int,
    overlap: int,
    batch_size: int = 4,
    max_batch: int | None = None,
    padding: float = 0.5,
) -> ExecutionPlan:
    metadata = tuple(
        AudioMetadata(Path(f"file_{index}.wav"), SAMPLE_RATE, 1, frames, "WAV")
        for index, frames in enumerate(frame_counts)
    )
    return build_execution_plan(
        metadata,
        target_sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        max_chunk_feature_frames=max_chunk,
        overlap_feature_frames=overlap,
        batch_size=batch_size,
        max_batch_feature_frames=max_batch or 2 * max_chunk,
        max_padding_fraction=padding,
    )


@pytest.mark.parametrize("samples", [0, 1, 159, 160, 161, 239_999, 240_000])
def test_feature_estimate_is_the_centered_stft_frame_count(samples: int) -> None:
    assert estimate_feature_frames(samples, SAMPLE_RATE, SAMPLE_RATE, HOP_LENGTH) == samples // HOP_LENGTH + 1


def test_feature_estimate_accounts_for_resampling() -> None:
    # 48 kHz -> 16 kHz: 480_000 source samples become 160_000 target samples.
    assert estimate_feature_frames(480_000, 48_000, SAMPLE_RATE, HOP_LENGTH) == 1001


def test_mixed_workload_is_bounded_and_interleaved() -> None:
    plan = make_plan([16_000, 16_000 * 120], max_chunk=500, overlap=10, batch_size=2, max_batch=1000)

    assert len(plan.items) > 2
    # Round-robin: the long file cannot push the short file to the back.
    assert [item.file_index for item in plan.items[:2]] == [0, 1]
    assert all(item.feature_frames <= 500 for item in plan.items)
    assert all(padded_batch_frames(batch) <= 1000 for batch in plan.batches)


@pytest.mark.parametrize("seed", range(5))
def test_chunks_tile_every_file_within_budget(seed: int) -> None:
    """Cores tile [0, N) exactly; sources wrap cores by at most the overlap."""

    generator = random.Random(seed)
    for _trial in range(40):
        max_chunk = generator.randint(20, 400)
        overlap = generator.randint(0, (max_chunk - 1) // 2)
        frames = generator.randint(0, 200_000)
        plan = make_plan([frames], max_chunk=max_chunk, overlap=overlap)
        items = sorted(plan.items, key=lambda item: item.chunk_index)
        overlap_samples = overlap * HOP_LENGTH

        assert items[0].core_start_frame == 0
        assert items[-1].core_end_frame == frames
        for previous, current in zip(items, items[1:]):
            assert previous.core_end_frame == current.core_start_frame
        for item in items:
            assert item.feature_frames <= max_chunk
            assert 0 <= item.core_start_frame - item.source_start_frame <= overlap_samples
            assert 0 <= item.source_end_frame - item.core_end_frame <= overlap_samples
            assert item.feature_frames == item.source_frame_count // HOP_LENGTH + 1


def test_zero_overlap_gives_back_to_back_chunks() -> None:
    plan = make_plan([100_000], max_chunk=100, overlap=0)

    for item in plan.items:
        assert (item.source_start_frame, item.source_end_frame) == (item.core_start_frame, item.core_end_frame)


@pytest.mark.parametrize("seed", range(4))
def test_random_workloads_respect_every_batch_limit(seed: int) -> None:
    generator = random.Random(100 + seed)
    for _trial in range(25):
        frames = [generator.randint(1_000, 600_000) for _ in range(generator.randint(1, 12))]
        plan = make_plan(frames, max_chunk=300, overlap=20, batch_size=6, max_batch=900, padding=0.4)
        for batch in plan.batches:
            longest = max(item.feature_frames for item in batch)
            used = sum(item.feature_frames for item in batch)
            assert len(batch) <= 6
            assert len(batch) * longest <= 900
            assert (len(batch) * longest - used) / (len(batch) * longest) <= 0.4


def test_every_work_item_is_scheduled_exactly_once() -> None:
    plan = make_plan([50_000, 400_000, 7_000, 0], max_chunk=200, overlap=15, batch_size=3)

    batched = [item for batch in plan.batches for item in batch]
    assert sorted(batched, key=id) == sorted(plan.items, key=id)
    assert len(batched) == len(set(map(id, batched)))


def test_overlap_must_leave_room_for_a_core() -> None:
    with pytest.raises(ValueError, match="twice overlap"):
        make_plan([100_000], max_chunk=100, overlap=50)


def test_item_larger_than_batch_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeding batch limit"):
        make_plan([100_000], max_chunk=500, overlap=0, max_batch=400)
