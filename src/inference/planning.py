"""Cost-aware work planning and boundary metadata for offline transcription."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AudioMetadata:
    """Measured source properties needed by the execution planner."""

    path: Path
    sample_rate: int
    channels: int
    frame_count: int
    format_name: str

    @property
    def duration_seconds(self) -> float:
        """Return source duration using the container's measured frame count."""

        if self.sample_rate <= 0:
            return 0.0
        return self.frame_count / self.sample_rate


@dataclass(frozen=True)
class WorkItem:
    """
    One bounded source interval scheduled for model inference.

    ``stream_index`` names the decoder that reads the item (see
    ``_decode_streams``): items of one stream are read in plan order by one
    decoder session, and different streams can be read in parallel.
    """

    file_index: int
    path: Path
    chunk_index: int
    source_start_frame: int
    source_end_frame: int
    core_start_frame: int
    core_end_frame: int
    feature_frames: int
    stream_index: int

    @property
    def source_frame_count(self) -> int:
        """Return the source frames requested, including overlap context."""

        return self.source_end_frame - self.source_start_frame

    @property
    def core_frame_count(self) -> int:
        """Return frames owned by this item after overlap reconciliation."""

        return self.core_end_frame - self.core_start_frame


@dataclass(frozen=True)
class ExecutionPlan:
    """Deterministic bounded work plan for one file or a mixed folder."""

    metadata: tuple[AudioMetadata, ...]
    items: tuple[WorkItem, ...]
    batches: tuple[tuple[WorkItem, ...], ...]
    target_sample_rate: int
    max_feature_frames: int
    max_batch_feature_frames: int


def target_frame_count(record: AudioMetadata, target_sample_rate: int) -> int:
    """Length of a file in samples at the checkpoint rate, as the planner cuts it."""

    return round(record.frame_count * target_sample_rate / record.sample_rate)


def estimate_feature_frames(
    sample_count: int,
    sample_rate: int,
    target_sample_rate: int,
    hop_length: int,
) -> int:
    """
    Return the STFT frame count the feature extractor will allocate.

    Centered ``torch.stft`` yields ``samples // hop + 1`` frames regardless of
    the FFT size. That tensor length (one more than the valid-frame count) is
    what occupies memory, so it is what the budget must bound.
    """

    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if min(sample_rate, target_sample_rate, hop_length) <= 0:
        raise ValueError("sample rates and hop length must be positive")

    target_samples = round(sample_count * target_sample_rate / sample_rate)
    return target_samples // hop_length + 1


def _chunk_ranges(
    frame_count: int,
    max_frames: int,
    overlap_frames: int,
) -> list[tuple[int, int, int, int]]:
    """Build source/core ranges while keeping overlap outside ownership."""

    if frame_count <= 0:
        return [(0, 0, 0, 0)]
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    # Overlap may exceed the core: the budget only requires core + 2 * overlap
    # to fit, and source ranges stay monotonic because starts clamp at zero.
    if overlap_frames < 0:
        raise ValueError("overlap_frames must be non-negative")

    ranges: list[tuple[int, int, int, int]] = []
    core_start = 0
    chunk_index = 0
    while core_start < frame_count:
        core_end = min(frame_count, core_start + max_frames)
        source_start = max(0, core_start - overlap_frames if chunk_index else core_start)
        source_end = min(frame_count, core_end + overlap_frames)
        ranges.append((source_start, source_end, core_start, core_end))
        core_start = core_end
        chunk_index += 1
    return ranges


def _padding_fraction(items: Iterable[WorkItem]) -> float:
    """Return the fraction of a batch occupied by frame padding."""

    item_list = list(items)
    if not item_list:
        return 0.0
    longest = max(item.feature_frames for item in item_list)
    capacity = longest * len(item_list)
    if capacity <= 0:
        return 0.0
    used = sum(item.feature_frames for item in item_list)
    return (capacity - used) / capacity


def _decode_streams(
    chunk_counts: list[int],
    rows_per_batch: int,
    decode_workers: int,
    max_open_files: int,
) -> list[list[int]]:
    """
    Assign every chunk to a decode stream so a batch can be read in parallel.

    A long file's chunks are dealt to a few streams in blocks of consecutive
    chunks: with 4 workers and 16-row batches, chunks 0-3 go to stream A, 4-7
    to B, 8-11 to C, 12-15 to D, 16-19 to A again, and so on. Each batch then
    holds one block per stream, decoded side by side, and every stream starts
    near the beginning of the file. Splitting the file into four regions
    instead made the first batch wait for a seek three quarters into the
    file (0.9 s for a 74-minute MP3, longer for longer files). Within a block
    each overlap is decoded once; between blocks a stream seeks forward.

    The workers are shared by the long files in progress, so with several of
    them each file gets fewer streams, and with at least as many files as
    workers every file is read by one stream, as before.

    Args:
        chunk_counts: Chunks of each file, in file order.
        rows_per_batch: Most rows a planned batch can hold.
        decode_workers: Threads available for decoding.
        max_open_files: Most multi-chunk files in progress at once.

    Returns:
        For each file, the stream index of each of its chunks. Indices are
        unique across files.
    """

    if decode_workers < 1:
        raise ValueError("decode_workers must be at least 1")
    long_files = sum(1 for count in chunk_counts if count > 1)
    files_in_progress = max(1, min(max_open_files, long_files))
    streams_per_file = max(1, decode_workers // files_in_progress)
    block_chunks = max(1, math.ceil(rows_per_batch / decode_workers))

    streams: list[list[int]] = []
    next_stream_index = 0
    for count in chunk_counts:
        blocks = math.ceil(count / block_chunks)
        stream_count = max(1, min(streams_per_file, blocks))
        streams.append(
            [next_stream_index + (chunk_index // block_chunks) % stream_count for chunk_index in range(count)]
        )
        next_stream_index += stream_count
    return streams


def _schedule_round_robin(items: Iterable[WorkItem], max_open_files: int) -> list[WorkItem]:
    """
    Interleave file queues so long files cannot monopolize the scheduler.

    A multi-chunk file keeps its decoder open (a file handle, or an ffmpeg
    process) and its overlap tail from its first chunk to its last, so at most
    ``max_open_files`` of them are in rotation at once; the next one joins when
    one finishes. Single-chunk files open and close within their own batch and
    are never held back, so short files still go ahead of long ones. With no
    more than ``max_open_files`` multi-chunk files this is plain round-robin.
    """

    if max_open_files < 1:
        raise ValueError("max_open_files must be at least 1")

    queues: dict[int, deque[WorkItem]] = {}
    for item in items:
        queues.setdefault(item.file_index, deque()).append(item)

    # Files with more than one chunk, the only ones that hold a decoder open
    # between batches.
    long_files = {file_index for file_index, queue in queues.items() if len(queue) > 1}

    rotation: deque[int] = deque()
    waiting_long_files: deque[int] = deque()
    admitted_long_files = 0
    for file_index in sorted(queues):
        if file_index not in long_files:
            rotation.append(file_index)
        elif admitted_long_files < max_open_files:
            rotation.append(file_index)
            admitted_long_files += 1
        else:
            waiting_long_files.append(file_index)

    scheduled: list[WorkItem] = []
    while rotation:
        file_index = rotation.popleft()
        queue = queues[file_index]
        scheduled.append(queue.popleft())
        if queue:
            rotation.append(file_index)
        elif file_index in long_files and waiting_long_files:
            # A long file finished and released its decoder: admit the next.
            rotation.append(waiting_long_files.popleft())
    return scheduled


def padded_batch_frames(items: Iterable[WorkItem]) -> int:
    """
    Return the frames a batch really allocates: rows x longest row.

    Features, encoder activations, and attention are padded to the longest
    item, so this (not the sum of item lengths) is what memory scales with.
    """

    item_list = list(items)
    if not item_list:
        return 0
    return len(item_list) * max(item.feature_frames for item in item_list)


def _batch_items(
    items: Iterable[WorkItem],
    batch_size: int,
    max_batch_feature_frames: int,
    max_padding_fraction: float,
) -> tuple[tuple[WorkItem, ...], ...]:
    """Pack scheduled work into deterministic micro-batches bounded by padded size."""

    if batch_size <= 0 or max_batch_feature_frames <= 0:
        raise ValueError("batch_size and max_batch_feature_frames must be positive")
    if not 0.0 <= max_padding_fraction <= 1.0:
        raise ValueError("max_padding_fraction must be between zero and one")

    batches: list[tuple[WorkItem, ...]] = []
    current: list[WorkItem] = []
    for item in items:
        if item.feature_frames > max_batch_feature_frames:
            raise ValueError(
                f"Work item requires {item.feature_frames} feature frames, "
                f"exceeding batch limit {max_batch_feature_frames}: {item.path}"
            )

        candidate = current + [item]
        exceeds_limits = (
            len(candidate) > batch_size
            or padded_batch_frames(candidate) > max_batch_feature_frames
            or _padding_fraction(candidate) > max_padding_fraction
        )
        if current and exceeds_limits:
            batches.append(tuple(current))
            candidate = [item]
        current = candidate

    if current:
        batches.append(tuple(current))
    return tuple(batches)


def build_execution_plan(
    metadata: Iterable[AudioMetadata],
    *,
    target_sample_rate: int,
    hop_length: int,
    max_chunk_feature_frames: int,
    overlap_feature_frames: int,
    batch_size: int,
    max_batch_feature_frames: int,
    max_padding_fraction: float,
    max_open_files: int,
    decode_workers: int = 1,
) -> ExecutionPlan:
    """
    Create an automatic cost-aware plan without a duration cutoff.

    Each file is cut into chunks whose STFT tensor never exceeds
    ``max_chunk_feature_frames``: a core interval the chunk owns, plus up to
    ``overlap_feature_frames`` of context on each side that it does not own.
    Chunks are interleaved round-robin across files, with at most
    ``max_open_files`` multi-chunk files in progress at once, and packed into
    micro-batches bounded by count, total frames, and padding fraction. Each
    chunk is assigned a decode stream for ``decode_workers`` threads.
    """

    metadata_list = tuple(metadata)
    if not metadata_list:
        raise ValueError("At least one audio metadata record is required")

    maximum_core_feature_frames = max_chunk_feature_frames - 2 * overlap_feature_frames
    if maximum_core_feature_frames <= 0:
        raise ValueError(
            "max_chunk_feature_frames must exceed twice overlap_feature_frames"
        )
    # Frames are samples // hop + 1, so a window of exactly N * hop samples
    # would need N + 1 frames. One sample less keeps the worst case at the
    # configured hard budget (the 1500 vs 1501 off-by-one).
    maximum_core_frames = maximum_core_feature_frames * hop_length - 1
    overlap_source_frames = overlap_feature_frames * hop_length

    file_ranges = [
        _chunk_ranges(
            target_frame_count(record, target_sample_rate),
            maximum_core_frames,
            overlap_source_frames,
        )
        for record in metadata_list
    ]
    rows_per_batch = max(1, min(batch_size, max_batch_feature_frames // max_chunk_feature_frames))
    file_streams = _decode_streams(
        [len(ranges) for ranges in file_ranges],
        rows_per_batch,
        decode_workers,
        max_open_files,
    )

    items: list[WorkItem] = []
    for file_index, (record, ranges) in enumerate(zip(metadata_list, file_ranges, strict=True)):
        for chunk_index, (source_start, source_end, core_start, core_end) in enumerate(ranges):
            feature_frames = estimate_feature_frames(
                source_end - source_start,
                target_sample_rate,
                target_sample_rate,
                hop_length,
            )
            items.append(
                WorkItem(
                    file_index=file_index,
                    path=record.path,
                    chunk_index=chunk_index,
                    source_start_frame=source_start,
                    source_end_frame=source_end,
                    core_start_frame=core_start,
                    core_end_frame=core_end,
                    feature_frames=feature_frames,
                    stream_index=file_streams[file_index][chunk_index],
                )
            )

    scheduled = _schedule_round_robin(items, max_open_files)
    batches = _batch_items(
        scheduled,
        batch_size,
        max_batch_feature_frames,
        max_padding_fraction,
    )
    return ExecutionPlan(
        metadata=metadata_list,
        items=tuple(scheduled),
        batches=batches,
        target_sample_rate=target_sample_rate,
        max_feature_frames=max_chunk_feature_frames,
        max_batch_feature_frames=max_batch_feature_frames,
    )


