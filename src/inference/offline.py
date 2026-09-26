"""
Offline automatic short/long transcription orchestration.

The service creates a frame-budgeted execution plan, decodes only bounded media
segments, runs one full-weight model instance through bounded microbatches, and
merges token output by timestamp ownership. It stops on model OOM and does not
retry, reduce the plan, fall back to CPU, or write a successful partial result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Iterable

import torch

from ..configuration.config import ParakeetConfig
from .planning import (
    AudioMetadata,
    ExecutionPlan,
    WorkItem,
    build_execution_plan,
)
from ..audio.media import MediaSession, DecodedSegment, inspect_media, open_media_session
from ..models.parakeet import GenerationResult, ParakeetTDT
from ..audio.features import ParakeetFeatureExtractor
from ..configuration.settings import InferenceSettings
from ..text.tokenization import BpeTokenizer


@dataclass(frozen=True)
class QualityAssessment:
    """Input and output anomaly flags that prevent silent false success."""

    status: str
    rms: float
    peak: float
    clipping_ratio: float
    finite: bool
    token_count: int
    blank_or_special_count: int
    repeated_adjacent_tokens: int


@dataclass(frozen=True)
class ChunkResult:
    """Small CPU-side result retained after one bounded model work unit."""

    item: WorkItem
    transcript: str
    token_ids: tuple[int, ...]
    durations: tuple[int, ...]
    frame_starts: tuple[int, ...]
    frame_ends: tuple[int, ...]
    encoder_length: int
    elapsed_seconds: float
    quality: QualityAssessment


@dataclass(frozen=True)
class OfflineFileResult:
    """Final output and measurements for one source file."""

    path: Path
    duration_seconds: float
    transcript: str
    status: str
    chunks: tuple[ChunkResult, ...]


@dataclass(frozen=True)
class OfflineRunResult:
    """Final deterministic folder result with plan and end-to-end metrics."""

    files: tuple[OfflineFileResult, ...]
    plan: ExecutionPlan
    elapsed_seconds: float
    peak_memory: dict[str, object]
    media_decode_seconds: float
    feature_seconds: float
    generation_seconds: float

    @property
    def total_audio_seconds(self) -> float:
        """Return total source duration across completed files."""

        return sum(result.duration_seconds for result in self.files)

    @property
    def real_time_factor(self) -> float:
        """Return end-to-end elapsed seconds divided by source seconds."""

        if self.total_audio_seconds <= 0:
            return 0.0
        return self.elapsed_seconds / self.total_audio_seconds

    @property
    def throughput_audio_seconds_per_second(self) -> float:
        """Return source-audio throughput over the complete run."""

        if self.elapsed_seconds <= 0:
            return 0.0
        return self.total_audio_seconds / self.elapsed_seconds


def _quality_assessment(
    segment: DecodedSegment,
    token_ids: tuple[int, ...],
    blank_token_id: int,
    pad_token_id: int,
) -> QualityAssessment:
    """Classify only obvious anomalies; this is not an accuracy oracle."""

    content_ids = [token for token in token_ids if token not in {blank_token_id, pad_token_id}]
    repeated = sum(
        left == right
        for left, right in zip(content_ids, content_ids[1:])
    )
    if not segment.finite:
        status = "decode_failed"
    elif segment.waveform.numel() == 0 or segment.rms == 0.0:
        status = "no_speech"
    elif repeated >= 3 and len(content_ids) >= 4:
        status = "possible_gibberish"
    else:
        status = "accepted"
    return QualityAssessment(
        status=status,
        rms=segment.rms,
        peak=segment.peak,
        clipping_ratio=segment.clipping_ratio,
        finite=segment.finite,
        token_count=len(content_ids),
        blank_or_special_count=len(token_ids) - len(content_ids),
        repeated_adjacent_tokens=repeated,
    )


class OfflineTranscriber:
    """Run automatic bounded offline inference for files or mixed folders."""

    def __init__(
        self,
        model: ParakeetTDT,
        configuration: ParakeetConfig,
        settings: InferenceSettings,
    ) -> None:
        """Bind model, checkpoint processor, planner settings, and tokenizer."""

        self.model = model
        self.configuration = configuration
        self.settings = settings
        self.feature_extractor = ParakeetFeatureExtractor(configuration)
        self.tokenizer = BpeTokenizer(
            configuration.tokenizer_path,
            configuration.pad_token_id,
            configuration.blank_token_id,
        )

    @property
    def device(self) -> torch.device:
        """Return the loaded model parameter device."""

        return next(self.model.parameters()).device

    def _plan(self, paths: tuple[Path, ...]) -> ExecutionPlan:
        """Inspect all containers and build a deterministic bounded plan."""

        metadata = tuple(inspect_media(path) for path in paths)
        feature = self.configuration.feature_extractor
        plan = build_execution_plan(
            metadata,
            target_sample_rate=int(feature["sampling_rate"]),
            n_fft=int(feature["n_fft"]),
            hop_length=int(feature["hop_length"]),
            max_chunk_feature_frames=self.settings.max_chunk_feature_frames,
            overlap_feature_frames=self.settings.overlap_feature_frames,
            batch_size=self.settings.batch_size,
            max_batch_feature_frames=self.settings.max_batch_feature_frames,
            max_padding_fraction=self.settings.max_padding_fraction,
        )
        return plan

    def _transcribe_batch(
        self,
        items: tuple[WorkItem, ...],
        sessions: dict[Path, MediaSession],
        sequential_state: dict[Path, tuple[int, torch.Tensor]],
    ) -> tuple[tuple[ChunkResult, ...], dict[str, float]]:
        """Decode and infer one bounded batch, stopping immediately on OOM."""

        target_rate = self.settings_target_sample_rate
        decode_started = time.perf_counter()
        segments: list[DecodedSegment] = []
        for item in items:
            previous = sequential_state.get(item.path)
            segment, waveform = sessions[item.path].read_sequential_segment(
                item.source_start_frame,
                item.source_end_frame,
                target_rate,
                previous_source_end_frame=previous[0] if previous else None,
                previous_waveform=previous[1] if previous else None,
            )
            sequential_state[item.path] = (item.source_end_frame, waveform)
            segments.append(segment)
        segments = tuple(segments)
        decode_seconds = time.perf_counter() - decode_started
        waveforms = [segment.waveform for segment in segments]
        feature_started = time.perf_counter()
        features, attention_mask = self.feature_extractor(
            waveforms,
            [target_rate] * len(waveforms),
            self.device,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        feature_seconds = time.perf_counter() - feature_started

        generation_started = time.perf_counter()
        try:
            generation = self.model.generate(features, attention_mask)
        except torch.OutOfMemoryError as error:
            if self.device.type == "cuda":
                memory = {
                    "allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "reserved_bytes": torch.cuda.memory_reserved(self.device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                }
            else:
                memory = {"device": str(self.device)}
            raise RuntimeError(
                f"Model inference ran out of memory for batch items {[item.path.name for item in items]}; "
                f"device={self.device}, feature_frames={[item.feature_frames for item in items]}, "
                f"memory={memory}. No fallback or retry was attempted."
            ) from error
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - generation_started

        results = tuple(
            self._chunk_result(item, segment, generation, row_index)
            for row_index, (item, segment) in enumerate(zip(items, segments))
        )
        return results, {
            "decode": decode_seconds,
            "feature": feature_seconds,
            "generation": generation_seconds,
        }

    @property
    def settings_target_sample_rate(self) -> int:
        """Return the checkpoint feature sample rate."""

        return int(self.configuration.feature_extractor["sampling_rate"])

    def _chunk_result(
        self,
        item: WorkItem,
        segment: DecodedSegment,
        generation: GenerationResult,
        row_index: int,
    ) -> ChunkResult:
        """Move one generated row to CPU and record detailed quality evidence."""

        sequence = generation.sequences[row_index].detach().cpu().tolist()
        durations = (
            generation.durations[row_index].detach().cpu().tolist()
        )
        frame_starts = (
            generation.frame_starts[row_index].detach().cpu().tolist()
            if generation.frame_starts is not None
            else [0] * len(sequence)
        )
        frame_ends = (
            generation.frame_ends[row_index].detach().cpu().tolist()
            if generation.frame_ends is not None
            else durations
        )
        encoder_length = int(
            generation.encoder_lengths[row_index].detach().cpu().item()
            if generation.encoder_lengths is not None
            else 0
        )
        token_ids = tuple(int(token) for token in sequence)
        transcript = self.tokenizer.decode(token_ids)
        quality = _quality_assessment(
            segment,
            token_ids,
            self.configuration.blank_token_id,
            self.configuration.pad_token_id,
        )
        result = ChunkResult(
            item=item,
            transcript=transcript,
            token_ids=token_ids,
            durations=tuple(int(value) for value in durations),
            frame_starts=tuple(int(value) for value in frame_starts),
            frame_ends=tuple(int(value) for value in frame_ends),
            encoder_length=encoder_length,
            elapsed_seconds=0.0,
            quality=quality,
        )
        return result

    def _merge_file(
        self,
        path: Path,
        metadata: AudioMetadata,
        chunks: tuple[ChunkResult, ...],
    ) -> OfflineFileResult:
        """Keep token pieces owned by core intervals and decode them once."""

        ordered_chunks = sorted(chunks, key=lambda chunk: chunk.item.chunk_index)
        merged_tokens: list[int] = []
        for chunk in ordered_chunks:
            scale = self.configuration.feature_extractor["hop_length"] * self.configuration.encoder["subsampling_factor"]
            for token, start, end in zip(
                chunk.token_ids,
                chunk.frame_starts,
                chunk.frame_ends,
            ):
                if token in {
                    self.configuration.blank_token_id,
                    self.configuration.pad_token_id,
                }:
                    continue
                token_start = chunk.item.source_start_frame + start * scale
                token_end = chunk.item.source_start_frame + end * scale
                midpoint = (token_start + token_end) // 2
                if chunk.item.core_start_frame <= midpoint < chunk.item.core_end_frame:
                    merged_tokens.append(token)

        transcript = self.tokenizer.decode(merged_tokens)
        statuses = {chunk.quality.status for chunk in ordered_chunks}
        status_priority = (
            "decode_failed",
            "possible_gibberish",
            "unstable_boundary",
            "low_confidence",
            "no_speech",
            "accepted",
        )
        status = next(
            candidate
            for candidate in status_priority
            if candidate in statuses
        )
        result = OfflineFileResult(
            path=path,
            duration_seconds=metadata.duration_seconds,
            transcript=transcript,
            status=status,
            chunks=ordered_chunks,
        )
        return result

    def transcribe(
        self,
        paths: Iterable[Path],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OfflineRunResult:
        """Transcribe a mixed file collection with automatic bounded planning."""

        path_tuple = tuple(path.expanduser().resolve() for path in paths)
        if not path_tuple:
            raise ValueError("At least one media path is required")
        started = time.perf_counter()
        plan = self._plan(path_tuple)
        sessions = {
            path: open_media_session(path)
            for path in path_tuple
        }
        sequential_state: dict[Path, tuple[int, torch.Tensor]] = {}
        chunk_results: dict[int, list[ChunkResult]] = {
            index: [] for index in range(len(path_tuple))
        }
        stage_totals = {"decode": 0.0, "feature": 0.0, "generation": 0.0}
        try:
            for batch_index, batch in enumerate(plan.batches):
                results, timings = self._transcribe_batch(batch, sessions, sequential_state)
                for name, elapsed in timings.items():
                    stage_totals[name] += elapsed
                for result in results:
                    chunk_results[result.item.file_index].append(result)
                if progress_callback is not None:
                    progress_callback(batch_index + 1, len(plan.batches))
        finally:
            for session in sessions.values():
                session.close()

        metadata = {record.path: record for record in (inspect_media(path) for path in path_tuple)}
        file_results = tuple(
            self._merge_file(
                path,
                metadata[path],
                tuple(chunk_results[index]),
            )
            for index, path in enumerate(path_tuple)
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_memory = {
                "available": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            }
        else:
            peak_memory = {"available": False, "device": str(self.device)}
        result = OfflineRunResult(
            files=file_results,
            plan=plan,
            elapsed_seconds=time.perf_counter() - started,
            peak_memory=peak_memory,
            media_decode_seconds=stage_totals["decode"],
            feature_seconds=stage_totals["feature"],
            generation_seconds=stage_totals["generation"],
        )
        return result