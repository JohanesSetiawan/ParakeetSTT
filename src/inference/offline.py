"""
Offline automatic short/long transcription orchestration.

The service creates a frame-budgeted execution plan, decodes only bounded media
segments, runs one full-weight model instance through bounded micro-batches,
and merges token output by timestamp ownership. It stops on OOM and does not
retry, reduce the plan, fall back to CPU, or write a partial result.

Every file gets an explicit status so that anomalies (non-finite input,
numerical failure, decoder guard, empty output) reach the caller instead of
looking like a normal transcript.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterable

import torch

from ..audio.features import ParakeetFeatureExtractor
from ..audio.media import DecodedSegment, MediaSession, inspect_media, open_media_session
from ..configuration.config import ParakeetConfig
from ..configuration.settings import InferenceSettings
from ..models.parakeet import GenerationResult, ParakeetTDT
from ..text.tokenization import BpeTokenizer
from .planning import AudioMetadata, ExecutionPlan, WorkItem, build_execution_plan


logger = logging.getLogger(__name__)


class FileStatus(StrEnum):
    """
    Per-file outcome, ordered from most to least severe.

    OK: transcript produced from finite input without decoder intervention.
    NUMERICAL_FAILURE: features or encoder states were non-finite for at least
        one chunk; that chunk's tokens were discarded.
    INPUT_NONFINITE: the decoded audio contained NaN/Inf samples, which were
        replaced by zeros before inference.
    DECODER_FORCED_ADVANCE: the per-frame symbol guard fired, so part of the
        transcript comes from a degenerate decoding loop.
    NO_SPEECH: every chunk was digital silence and nothing was transcribed.
    EMPTY_TRANSCRIPT: audio was not silent but no token was produced.
    """

    NUMERICAL_FAILURE = "numerical_failure"
    INPUT_NONFINITE = "input_nonfinite"
    DECODER_FORCED_ADVANCE = "decoder_forced_advance"
    NO_SPEECH = "no_speech"
    EMPTY_TRANSCRIPT = "empty_transcript"
    OK = "ok"


@dataclass(frozen=True)
class ChunkResult:
    """Small CPU-side result retained after one bounded model work unit."""

    item: WorkItem
    token_ids: tuple[int, ...]
    durations: tuple[int, ...]
    frame_starts: tuple[int, ...]
    frame_ends: tuple[int, ...]
    encoder_length: int
    input_finite: bool
    silent: bool
    features_finite: bool
    encoder_finite: bool
    forced_advances: int

    @property
    def numerically_valid(self) -> bool:
        """Return whether this chunk's tokens can be trusted at all."""

        return self.features_finite and self.encoder_finite


@dataclass(frozen=True)
class OfflineFileResult:
    """Final output and measurements for one source file."""

    path: Path
    duration_seconds: float
    transcript: str
    status: FileStatus
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


def classify_file(transcript: str, chunks: tuple[ChunkResult, ...]) -> FileStatus:
    """
    Derive one file status from objective chunk facts.

    Problems anywhere in the file win over the transcript state, because a
    single bad chunk makes the whole transcript incomplete or suspect.
    """

    if any(not chunk.numerically_valid for chunk in chunks):
        return FileStatus.NUMERICAL_FAILURE
    if any(not chunk.input_finite for chunk in chunks):
        return FileStatus.INPUT_NONFINITE
    if any(chunk.forced_advances > 0 for chunk in chunks):
        return FileStatus.DECODER_FORCED_ADVANCE
    if not transcript:
        if all(chunk.silent for chunk in chunks):
            return FileStatus.NO_SPEECH
        return FileStatus.EMPTY_TRANSCRIPT
    return FileStatus.OK


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

        feature = configuration.feature_extractor
        self.target_sample_rate = int(feature["sampling_rate"])
        self.hop_length = int(feature["hop_length"])
        # One encoder frame covers hop * subsampling input samples; converts
        # decoder frame indices back to source sample positions.
        self.samples_per_encoder_frame = self.hop_length * int(
            configuration.encoder["subsampling_factor"]
        )
        self.special_token_ids = {
            configuration.blank_token_id,
            configuration.pad_token_id,
        }

    @property
    def device(self) -> torch.device:
        """Return the loaded model parameter device."""

        return next(self.model.parameters()).device

    # -------------------------------------------------------------------------
    # Planning
    # -------------------------------------------------------------------------

    def _plan(self, metadata: tuple[AudioMetadata, ...]) -> ExecutionPlan:
        """Build a deterministic bounded plan from already inspected files."""

        return build_execution_plan(
            metadata,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            max_chunk_feature_frames=self.settings.max_chunk_feature_frames,
            overlap_feature_frames=self.settings.overlap_feature_frames,
            batch_size=self.settings.batch_size,
            max_batch_feature_frames=self.settings.max_batch_feature_frames,
            max_padding_fraction=self.settings.max_padding_fraction,
        )

    # -------------------------------------------------------------------------
    # One micro-batch
    # -------------------------------------------------------------------------

    def _decode_batch_audio(
        self,
        items: tuple[WorkItem, ...],
        sessions: dict[Path, MediaSession],
        sequential_state: dict[Path, tuple[int, torch.Tensor]],
        remaining_chunks: dict[Path, int],
    ) -> tuple[DecodedSegment, ...]:
        """Decode each item in plan order, reusing per-file overlap tails."""

        segments: list[DecodedSegment] = []
        for item in items:
            previous = sequential_state.get(item.path)
            segment, waveform = sessions[item.path].read_sequential_segment(
                item.source_start_frame,
                item.source_end_frame,
                self.target_sample_rate,
                previous_source_end_frame=previous[0] if previous else None,
                previous_waveform=previous[1] if previous else None,
            )
            segments.append(segment)

            # Keep the overlap tail only while the file still has chunks left,
            # so finished files release their last waveform immediately.
            remaining_chunks[item.path] -= 1
            if remaining_chunks[item.path] > 0:
                sequential_state[item.path] = (item.source_end_frame, waveform)
            else:
                sequential_state.pop(item.path, None)
                sessions[item.path].close()
        return tuple(segments)

    def _out_of_memory(self, stage: str, items: tuple[WorkItem, ...]) -> RuntimeError:
        """Build the stop-the-run error for an accelerator OOM."""

        if self.device.type == "cuda":
            memory = {
                "allocated_bytes": torch.cuda.memory_allocated(self.device),
                "reserved_bytes": torch.cuda.memory_reserved(self.device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            }
        else:
            memory = {"device": str(self.device)}
        message = (
            f"Out of memory during {stage} for batch items "
            f"{[f'{item.path.name}#{item.chunk_index}' for item in items]}; "
            f"device={self.device}, feature_frames={[item.feature_frames for item in items]}, "
            f"memory={memory}. No fallback or retry was attempted; lower "
            "inference.max_batch_feature_frames or inference.max_chunk_feature_frames."
        )
        logger.error(message)
        return RuntimeError(message)

    def _transcribe_batch(
        self,
        items: tuple[WorkItem, ...],
        sessions: dict[Path, MediaSession],
        sequential_state: dict[Path, tuple[int, torch.Tensor]],
        remaining_chunks: dict[Path, int],
    ) -> tuple[tuple[ChunkResult, ...], dict[str, float]]:
        """Decode, extract, and infer one bounded batch, stopping on OOM."""

        decode_started = time.perf_counter()
        segments = self._decode_batch_audio(items, sessions, sequential_state, remaining_chunks)
        decode_seconds = time.perf_counter() - decode_started

        feature_started = time.perf_counter()
        try:
            features, attention_mask = self.feature_extractor(
                [segment.waveform for segment in segments],
                [self.target_sample_rate] * len(segments),
                self.device,
            )
        except torch.OutOfMemoryError as error:
            raise self._out_of_memory("feature extraction", items) from error
        features_finite = torch.isfinite(features).all(dim=2).all(dim=1)  # (B,)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        feature_seconds = time.perf_counter() - feature_started

        generation_started = time.perf_counter()
        try:
            generation = self.model.generate(features, attention_mask)
        except torch.OutOfMemoryError as error:
            raise self._out_of_memory("model generation", items) from error
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - generation_started

        features_finite_rows = features_finite.cpu().tolist()
        results = tuple(
            self._chunk_result(item, segment, generation, row_index, features_finite_rows[row_index])
            for row_index, (item, segment) in enumerate(zip(items, segments))
        )
        timings = {
            "decode": decode_seconds,
            "feature": feature_seconds,
            "generation": generation_seconds,
        }
        logger.debug(
            "batch items=%s feature_frames=%s decode=%.3fs feature=%.3fs generation=%.3fs",
            [f"{item.path.name}#{item.chunk_index}" for item in items],
            [item.feature_frames for item in items],
            decode_seconds,
            feature_seconds,
            generation_seconds,
        )
        return results, timings

    def _chunk_result(
        self,
        item: WorkItem,
        segment: DecodedSegment,
        generation: GenerationResult,
        row_index: int,
        features_finite: bool,
    ) -> ChunkResult:
        """Move one generated row to CPU together with its health facts."""

        return ChunkResult(
            item=item,
            token_ids=tuple(generation.sequences[row_index].cpu().tolist()),
            durations=tuple(generation.durations[row_index].cpu().tolist()),
            frame_starts=tuple(generation.frame_starts[row_index].cpu().tolist()),
            frame_ends=tuple(generation.frame_ends[row_index].cpu().tolist()),
            encoder_length=int(generation.encoder_lengths[row_index].cpu()),
            input_finite=segment.finite,
            silent=segment.rms == 0.0,
            features_finite=features_finite,
            encoder_finite=bool(generation.encoder_finite[row_index].cpu()),
            forced_advances=int(generation.forced_advances[row_index].cpu()),
        )

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    def _merge_file(
        self,
        metadata: AudioMetadata,
        chunks: tuple[ChunkResult, ...],
    ) -> OfflineFileResult:
        """
        Keep each token only in the chunk whose core owns its midpoint.

        Overlap context is decoded by two neighboring chunks; ownership by
        token midpoint assigns every position to exactly one of them. Repeated
        words are never collapsed, because natural speech repeats words.
        """

        ordered_chunks = tuple(sorted(chunks, key=lambda chunk: chunk.item.chunk_index))
        merged_tokens: list[int] = []
        for chunk in ordered_chunks:
            if not chunk.numerically_valid:
                continue
            for token, start, end in zip(chunk.token_ids, chunk.frame_starts, chunk.frame_ends):
                if token in self.special_token_ids:
                    continue
                token_start = chunk.item.source_start_frame + start * self.samples_per_encoder_frame
                token_end = chunk.item.source_start_frame + end * self.samples_per_encoder_frame
                midpoint = (token_start + token_end) // 2
                if chunk.item.core_start_frame <= midpoint < chunk.item.core_end_frame:
                    merged_tokens.append(token)

        transcript = self.tokenizer.decode(merged_tokens)
        return OfflineFileResult(
            path=metadata.path,
            duration_seconds=metadata.duration_seconds,
            transcript=transcript,
            status=classify_file(transcript, ordered_chunks),
            chunks=ordered_chunks,
        )

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def transcribe(
        self,
        paths: Iterable[Path],
        *,
        metadata: Iterable[AudioMetadata] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OfflineRunResult:
        """
        Transcribe a mixed file collection with automatic bounded planning.

        Args:
            paths: Media files in output order.
            metadata: Optional already-inspected metadata for ``paths`` (same
                order); avoids probing each container a second time.
            progress_callback: Called with ``(completed_batches, total_batches)``.

        Returns:
            One result per input file plus plan and stage timings.
        """

        path_tuple = tuple(path.expanduser().resolve() for path in paths)
        if not path_tuple:
            raise ValueError("At least one media path is required")
        if len(set(path_tuple)) != len(path_tuple):
            raise ValueError("Each media path may appear only once per run")

        started = time.perf_counter()
        if metadata is None:
            metadata_tuple = tuple(inspect_media(path) for path in path_tuple)
        else:
            metadata_tuple = tuple(metadata)
            if tuple(record.path.resolve() for record in metadata_tuple) != path_tuple:
                raise ValueError("metadata must describe paths in the same order")
        plan = self._plan(metadata_tuple)
        logger.info(
            "plan files=%d work_items=%d batches=%d audio_seconds=%.3f",
            len(path_tuple),
            len(plan.items),
            len(plan.batches),
            sum(record.duration_seconds for record in metadata_tuple),
        )

        remaining_chunks: dict[Path, int] = {path: 0 for path in path_tuple}
        for item in plan.items:
            remaining_chunks[item.path] += 1

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        sessions = {path: open_media_session(path) for path in path_tuple}
        sequential_state: dict[Path, tuple[int, torch.Tensor]] = {}
        chunk_results: dict[int, list[ChunkResult]] = {index: [] for index in range(len(path_tuple))}
        stage_totals = {"decode": 0.0, "feature": 0.0, "generation": 0.0}
        try:
            for batch_index, batch in enumerate(plan.batches):
                results, timings = self._transcribe_batch(
                    batch,
                    sessions,
                    sequential_state,
                    remaining_chunks,
                )
                for name, elapsed in timings.items():
                    stage_totals[name] += elapsed
                for result in results:
                    chunk_results[result.item.file_index].append(result)
                if progress_callback is not None:
                    progress_callback(batch_index + 1, len(plan.batches))
        finally:
            for session in sessions.values():
                session.close()

        file_results = tuple(
            self._merge_file(record, tuple(chunk_results[index]))
            for index, record in enumerate(metadata_tuple)
        )
        for file_result in file_results:
            log_level = logging.INFO if file_result.status is FileStatus.OK else logging.WARNING
            logger.log(
                log_level,
                "file=%s status=%s duration_seconds=%.3f chunks=%d words=%d",
                file_result.path.name,
                file_result.status.value,
                file_result.duration_seconds,
                len(file_result.chunks),
                len(file_result.transcript.split()),
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_memory: dict[str, object] = {
                "available": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            }
        else:
            peak_memory = {"available": False, "device": str(self.device)}

        return OfflineRunResult(
            files=file_results,
            plan=plan,
            elapsed_seconds=time.perf_counter() - started,
            peak_memory=peak_memory,
            media_decode_seconds=stage_totals["decode"],
            feature_seconds=stage_totals["feature"],
            generation_seconds=stage_totals["generation"],
        )
