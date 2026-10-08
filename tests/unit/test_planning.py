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
    rows_per_batch,
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
    max_open_files: int = 8,
) -> ExecutionPlan:
    metadata = tuple(
        AudioMetadata(Path(f"file_{index}.wav"), SAMPLE_RATE, 1, frames, "WAV", True)
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
        max_open_files=max_open_files,
        decode_workers=1,
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


# =============================================================================
# Bounded round-robin (open decoder limit)
# =============================================================================


def open_long_files_over_time(plan: ExecutionPlan) -> list[int]:
    """Multi-chunk files started but not finished, after each scheduled item."""

    chunk_counts: dict[int, int] = {}
    for item in plan.items:
        chunk_counts[item.file_index] = chunk_counts.get(item.file_index, 0) + 1

    seen: dict[int, int] = {}
    in_progress: set[int] = set()
    history = []
    for item in plan.items:
        seen[item.file_index] = seen.get(item.file_index, 0) + 1
        if chunk_counts[item.file_index] > 1:
            in_progress.add(item.file_index)
        if seen[item.file_index] == chunk_counts[item.file_index]:
            in_progress.discard(item.file_index)
        history.append(len(in_progress))
    return history


@pytest.mark.parametrize(("seed", "limit"), [(0, 1), (1, 2), (2, 3), (3, 5)])
def test_no_more_long_files_than_the_limit_are_open_at_once(seed: int, limit: int) -> None:
    generator = random.Random(seed)
    # A mix of single-chunk files (under 5 s) and long files (up to 3 minutes).
    frames = [generator.choice([16_000 * 2, 16_000 * generator.randint(20, 180)]) for _ in range(30)]
    plan = make_plan(frames, max_chunk=500, overlap=10, batch_size=4, max_batch=2000, max_open_files=limit)

    assert max(open_long_files_over_time(plan)) <= limit
    # Every chunk is still scheduled exactly once and each file stays in order.
    assert len({(item.file_index, item.chunk_index) for item in plan.items}) == len(plan.items)
    for file_index in set(item.file_index for item in plan.items):
        chunk_order = [item.chunk_index for item in plan.items if item.file_index == file_index]
        assert chunk_order == sorted(chunk_order)


def test_short_files_are_not_held_back_by_waiting_long_files() -> None:
    # Files 0-2 are long, 3-5 short; with one open long file allowed, the short
    # files must still be scheduled before the second long file starts.
    frames = [16_000 * 60] * 3 + [16_000 * 2] * 3
    plan = make_plan(frames, max_chunk=500, overlap=10, batch_size=4, max_batch=2000, max_open_files=1)

    order = [item.file_index for item in plan.items]
    first_of_file_1 = order.index(1)
    assert all(order.index(short_file) < first_of_file_1 for short_file in (3, 4, 5))


def test_within_the_limit_the_schedule_is_plain_round_robin() -> None:
    frames = [16_000 * 30, 16_000 * 2, 16_000 * 45]
    bounded = make_plan(frames, max_chunk=500, overlap=10, max_open_files=2)
    unbounded = make_plan(frames, max_chunk=500, overlap=10, max_open_files=100)

    assert [(item.file_index, item.chunk_index) for item in bounded.items] == [
        (item.file_index, item.chunk_index) for item in unbounded.items
    ]
    assert [item.file_index for item in bounded.items[:3]] == [0, 1, 2]


def test_open_file_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_open_files"):
        make_plan([16_000 * 30], max_chunk=500, overlap=10, max_open_files=0)


# =============================================================================
# Decode streams
# =============================================================================


# One chunk core: 100 frames without overlap, minus the planner's one-sample reserve.
CHUNK_SAMPLES = 100 * HOP_LENGTH - 1


def stream_plan(
    chunk_counts: list[int],
    decode_workers: int,
    max_open_files: int = 8,
    seekable: list[bool] | None = None,
) -> ExecutionPlan:
    """Files of exactly ``chunk_counts`` chunks, in batches of 8 rows."""

    seekable = seekable or [True] * len(chunk_counts)
    metadata = tuple(
        AudioMetadata(Path(f"file_{index}.wav"), SAMPLE_RATE, 1, chunks * CHUNK_SAMPLES, "WAV", can_seek)
        for index, (chunks, can_seek) in enumerate(zip(chunk_counts, seekable, strict=True))
    )
    return build_execution_plan(
        metadata,
        target_sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        max_chunk_feature_frames=100,
        overlap_feature_frames=0,
        batch_size=8,
        max_batch_feature_frames=800,
        max_padding_fraction=1.0,
        max_open_files=max_open_files,
        decode_workers=decode_workers,
    )


def streams_by_chunk(plan: ExecutionPlan, file_index: int = 0) -> list[int]:
    items = sorted(
        (item for item in plan.items if item.file_index == file_index),
        key=lambda item: item.chunk_index,
    )
    return [item.stream_index for item in items]


def test_a_long_file_is_dealt_to_the_workers_in_blocks() -> None:
    # 20 chunks, 8 rows per batch, 4 workers: blocks of 2 chunks.
    plan = stream_plan([20], decode_workers=4)

    streams = streams_by_chunk(plan)

    assert streams == [0, 0, 1, 1, 2, 2, 3, 3] * 2 + [0, 0, 1, 1]
    # Every full batch is read by all four streams, two chunks each.
    for batch in plan.batches[:2]:
        counts = sorted(
            sum(1 for item in batch if item.stream_index == stream)
            for stream in {item.stream_index for item in batch}
        )
        assert counts == [2, 2, 2, 2]


def test_one_worker_reads_each_file_as_one_stream() -> None:
    plan = stream_plan([20, 5, 1], decode_workers=1)

    assert set(streams_by_chunk(plan, 0)) == {0}
    assert set(streams_by_chunk(plan, 1)) == {1}
    assert streams_by_chunk(plan, 2) == [2]


def test_workers_are_shared_by_the_long_files_in_progress() -> None:
    two_files = stream_plan([20, 20], decode_workers=4)
    many_files = stream_plan([20] * 5, decode_workers=4)
    limited = stream_plan([20] * 5, decode_workers=4, max_open_files=2)

    assert [len(set(streams_by_chunk(two_files, index))) for index in range(2)] == [2, 2]
    assert all(len(set(streams_by_chunk(many_files, index))) == 1 for index in range(5))
    # Two files in rotation already hold the two decoders the limit allows.
    assert all(len(set(streams_by_chunk(limited, index))) == 1 for index in range(5))


def test_streams_never_hold_more_decoders_than_max_open_files() -> None:
    """Review finding: 8 workers gave one file 8 open decoders with max_open_files = 2."""

    plan = stream_plan([40], decode_workers=8, max_open_files=2)

    assert len(set(streams_by_chunk(plan))) == 2


def test_files_without_sample_accurate_seek_stay_one_stream() -> None:
    """FFmpeg-decoded files are read from the start, never split at estimated seeks."""

    plan = stream_plan([20, 20], decode_workers=4, seekable=[False, True])

    assert set(streams_by_chunk(plan, 0)) == {0}
    assert len(set(streams_by_chunk(plan, 1))) == 2


@pytest.mark.parametrize(
    ("batch_size", "max_batch", "max_chunk", "expected"),
    [(16, 24_000, 1500, 16), (16, 12_000, 1500, 8), (4, 24_000, 1500, 4), (16, 1500, 1500, 1)],
)
def test_rows_per_batch_takes_the_tighter_limit(batch_size, max_batch, max_chunk, expected) -> None:
    assert rows_per_batch(batch_size, max_batch, max_chunk) == expected


def test_short_files_never_get_more_streams_than_blocks() -> None:
    plan = stream_plan([3, 1], decode_workers=4)

    assert streams_by_chunk(plan, 0) == [0, 0, 1]
    assert streams_by_chunk(plan, 1) == [2]


def test_stream_indices_are_unique_across_files() -> None:
    plan = stream_plan([20, 20, 1], decode_workers=4)

    owners: dict[int, int] = {}
    for item in plan.items:
        assert owners.setdefault(item.stream_index, item.file_index) == item.file_index


def test_decode_workers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="decode_workers"):
        stream_plan([1], decode_workers=0)
