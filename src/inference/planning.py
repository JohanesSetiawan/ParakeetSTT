"""Cost-aware work planning and boundary metadata for offline transcription."""

from __future__ import annotations

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
    """One bounded source interval scheduled for model inference."""

    file_index: int
    path: Path
    chunk_index: int
    source_start_frame: int
    source_end_frame: int
    core_start_frame: int
    core_end_frame: int
    feature_frames: int

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

    items: tuple[WorkItem, ...]
    batches: tuple[tuple[WorkItem, ...], ...]
    target_sample_rate: int
    max_feature_frames: int
    max_batch_feature_frames: int



def estimate_feature_frames(
    sample_count: int,
    sample_rate: int,
    target_sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> int:
    """Estimate valid mel frames using the same padded-STFT contract as runtime."""

    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if min(sample_rate, target_sample_rate, n_fft, hop_length) <= 0:
        raise ValueError("sample rates, FFT size, and hop length must be positive")

    target_samples = round(sample_count * target_sample_rate / sample_rate)
    # torch.stft with centered padding produces one additional edge frame for
    # the valid lengths used by the feature extractor. The planner must use the
    # same contract or its resource budget can undercount every work item.
    del n_fft
    return max(1, target_samples // hop_length + 1)


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
    if overlap_frames < 0 or overlap_frames >= max_frames:
        raise ValueError("overlap_frames must be non-negative and smaller than max_frames")

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


def _schedule_round_robin(items: Iterable[WorkItem]) -> list[WorkItem]:
    """Interleave file queues so long files cannot monopolize the scheduler."""

    queues: dict[int, deque[WorkItem]] = {}
    for item in items:
        queues.setdefault(item.file_index, deque()).append(item)

    scheduled: list[WorkItem] = []
    active_queues = deque(sorted(queues))
    while active_queues:
        file_index = active_queues.popleft()
        queue = queues[file_index]
        scheduled.append(queue.popleft())
        if queue:
            active_queues.append(file_index)
    return scheduled


def _batch_items(
    items: Iterable[WorkItem],
    batch_size: int,
    max_batch_feature_frames: int,
    max_padding_fraction: float,
) -> tuple[tuple[WorkItem, ...], ...]:
    """Pack scheduled work into deterministic bounded microbatches."""

    if batch_size <= 0 or max_batch_feature_frames <= 0:
        raise ValueError("batch_size and max_batch_feature_frames must be positive")
    if not 0.0 <= max_padding_fraction <= 1.0:
        raise ValueError("max_padding_fraction must be between zero and one")

    batches: list[tuple[WorkItem, ...]] = []
    current: list[WorkItem] = []
    current_frames = 0
    for item in items:
        if item.feature_frames > max_batch_feature_frames:
            raise ValueError(
                f"Work item requires {item.feature_frames} feature frames, "
                f"exceeding batch limit {max_batch_feature_frames}: {item.path}"
            )

        candidate = current + [item]
        candidate_frames = current_frames + item.feature_frames
        exceeds_limits = (
            len(candidate) > batch_size
            or candidate_frames > max_batch_feature_frames
            or _padding_fraction(candidate) > max_padding_fraction
        )
        if current and exceeds_limits:
            batches.append(tuple(current))
            current = []
            current_frames = 0
            candidate = [item]
            candidate_frames = item.feature_frames

        current = candidate
        current_frames = candidate_frames

    if current:
        batches.append(tuple(current))
    return tuple(batches)


def build_execution_plan(
    metadata: Iterable[AudioMetadata],
    *,
    target_sample_rate: int,
    n_fft: int,
    hop_length: int,
    max_chunk_feature_frames: int,
    overlap_feature_frames: int,
    batch_size: int,
    max_batch_feature_frames: int,
    max_padding_fraction: float,
) -> ExecutionPlan:
    """Create an automatic cost-aware plan without a duration cutoff."""

    metadata_list = tuple(metadata)
    if not metadata_list:
        raise ValueError("At least one audio metadata record is required")
    if max_chunk_feature_frames <= overlap_feature_frames:
        raise ValueError("max_chunk_feature_frames must exceed overlap_feature_frames")

    items: list[WorkItem] = []
    for file_index, record in enumerate(metadata_list):
        target_frames = round(record.frame_count * target_sample_rate / record.sample_rate)
        target_feature_frames = estimate_feature_frames(
            record.frame_count,
            record.sample_rate,
            target_sample_rate,
            n_fft,
            hop_length,
        )
        maximum_core_feature_frames = max_chunk_feature_frames - 2 * overlap_feature_frames
        if maximum_core_feature_frames <= 0:
            raise ValueError(
                "max_chunk_feature_frames must leave room for both overlap sides"
            )
        # Valid feature frames are floor(samples / hop) + 1. Reserve one
        # sample so a source window at the exact boundary cannot become one
        # frame larger than the configured hard budget after centered padding.
        maximum_core_frames = maximum_core_feature_frames * hop_length - 1
        overlap_source_frames = overlap_feature_frames * hop_length
        ranges = _chunk_ranges(
            target_frames,
            maximum_core_frames,
            overlap_source_frames,
        )
        for chunk_index, (source_start, source_end, core_start, core_end) in enumerate(ranges):
            feature_frames = estimate_feature_frames(
                source_end - source_start,
                target_sample_rate,
                target_sample_rate,
                n_fft,
                hop_length,
            )
            if target_feature_frames <= max_chunk_feature_frames:
                feature_frames = target_feature_frames
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
                )
            )

    scheduled = _schedule_round_robin(items)
    batches = _batch_items(
        scheduled,
        batch_size,
        max_batch_feature_frames,
        max_padding_fraction,
    )
    return ExecutionPlan(
        items=tuple(scheduled),
        batches=batches,
        target_sample_rate=target_sample_rate,
        max_feature_frames=max_chunk_feature_frames,
        max_batch_feature_frames=max_batch_feature_frames,
    )


