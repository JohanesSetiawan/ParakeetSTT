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

import dataclasses
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
from .merging import ChunkWords, TimedToken, group_words, merge_chunks
from .planning import (
    AudioMetadata,
    ExecutionPlan,
    WorkItem,
    build_execution_plan,
    estimate_feature_frames,
)
from .recovery import Window, longest_untranscribed_gap, recovery_windows


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
    UNTRANSCRIBED_GAP: a long non-silent stretch produced no words, even
        after re-decoding with shifted windows. Either the decoder collapsed
        there or the audio is not speech (music, noise).
    NO_SPEECH: every chunk was digital silence and nothing was transcribed.
    EMPTY_TRANSCRIPT: audio was not silent but no token was produced.
    UNREADABLE: the file could not be decoded as audio and was not
        transcribed; assigned by the command layer during discovery.
    """

    NUMERICAL_FAILURE = "numerical_failure"
    INPUT_NONFINITE = "input_nonfinite"
    DECODER_FORCED_ADVANCE = "decoder_forced_advance"
    UNTRANSCRIBED_GAP = "untranscribed_gap"
    NO_SPEECH = "no_speech"
    EMPTY_TRANSCRIPT = "empty_transcript"
    UNREADABLE = "unreadable"
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
    gap_start_sample: int = 0
    gap_end_sample: int = 0
    gap_rms: float = 0.0
    recovered: bool = False
    processing_seconds: float = 0.0

    @property
    def gap_samples(self) -> int:
        """Length of the longest stretch of the core with no word."""

        return self.gap_end_sample - self.gap_start_sample

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

    @property
    def processing_seconds(self) -> float:
        """
        Time spent on this file: its own decode time plus its share of each
        batch's feature and generation time (by feature frames), plus any
        collapse re-decodes. Batches mix files, so this is an attribution,
        not a separately timed run.
        """

        return sum(chunk.processing_seconds for chunk in self.chunks)

    @property
    def real_time_factor(self) -> float:
        """Attributed processing seconds per second of audio."""

        if self.duration_seconds <= 0:
            return 0.0
        return self.processing_seconds / self.duration_seconds


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
    recovery_seconds: float = 0.0

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


@dataclass(frozen=True)
class _GenerationOnHost:
    """A batch's generation output copied to Python lists, one transfer each."""

    sequences: list[list[int]]
    durations: list[list[int]]
    frame_starts: list[list[int]]
    frame_ends: list[list[int]]
    encoder_lengths: list[int]
    forced_advances: list[int]
    encoder_finite: list[bool]
    features_finite: list[bool]

    @classmethod
    def from_device(
        cls,
        generation: GenerationResult,
        features_finite: torch.Tensor,
    ) -> _GenerationOnHost:
        return cls(
            sequences=generation.sequences.cpu().tolist(),
            durations=generation.durations.cpu().tolist(),
            frame_starts=generation.frame_starts.cpu().tolist(),
            frame_ends=generation.frame_ends.cpu().tolist(),
            encoder_lengths=generation.encoder_lengths.cpu().tolist(),
            forced_advances=generation.forced_advances.cpu().tolist(),
            encoder_finite=generation.encoder_finite.cpu().tolist(),
            features_finite=features_finite.cpu().tolist(),
        )


def classify_file(
    transcript: str,
    chunks: tuple[ChunkResult, ...],
    untranscribed_gap: bool = False,
) -> FileStatus:
    """
    Derive one file status from objective chunk facts.

    Problems anywhere in the file win over the transcript state, because a
    single bad chunk makes the whole transcript incomplete or suspect.

    Args:
        transcript: The merged transcript.
        chunks: All chunks of the file.
        untranscribed_gap: Whether a chunk still has a long non-silent
            stretch without words after recovery. It is reported only for
            non-empty transcripts; an empty one is covered by NO_SPEECH or
            EMPTY_TRANSCRIPT.
    """

    if any(not chunk.numerically_valid for chunk in chunks):
        return FileStatus.NUMERICAL_FAILURE
    if any(not chunk.input_finite for chunk in chunks):
        return FileStatus.INPUT_NONFINITE
    if any(chunk.forced_advances > 0 for chunk in chunks):
        return FileStatus.DECODER_FORCED_ADVANCE
    if transcript and untranscribed_gap:
        return FileStatus.UNTRANSCRIBED_GAP
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
        self.merge_tolerance_samples = settings.merge_tolerance_feature_frames * self.hop_length
        self.gap_threshold_samples = round(settings.untranscribed_gap_seconds * self.target_sample_rate)
        self.recovery_offsets_samples = tuple(
            offset * self.hop_length for offset in settings.recovery_start_offsets_feature_frames
        )
        # Same one-sample reserve as the planner, so a recovery window never
        # needs one more feature frame than the configured chunk budget.
        self.max_chunk_samples = settings.max_chunk_feature_frames * self.hop_length - 1
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
            max_open_files=self.settings.max_open_files,
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
    ) -> tuple[tuple[DecodedSegment, ...], tuple[float, ...]]:
        """
        Decode each item in plan order, reusing per-file overlap tails.

        Returns the segments and each item's own decode time.

        A file's decoder session is opened at its first chunk and closed after
        its last one, so single-chunk files never hold a handle beyond their
        own batch and the number of open handles tracks the files that are
        actually in progress, not the size of the folder.
        """

        segments: list[DecodedSegment] = []
        decode_seconds: list[float] = []
        for item in items:
            item_started = time.perf_counter()
            session = sessions.get(item.path)
            if session is None:
                session = open_media_session(item.path)
                sessions[item.path] = session

            previous = sequential_state.get(item.path)
            segment, waveform = session.read_sequential_segment(
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
                sessions.pop(item.path).close()
            decode_seconds.append(time.perf_counter() - item_started)
        return tuple(segments), tuple(decode_seconds)

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
            "inference.max_batch_feature_frames or inference.max_chunk_feature_frames, "
            "or raise memory.reserve_mib so the automatic budget leaves more room."
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
        segments, item_decode_seconds = self._decode_batch_audio(
            items,
            sessions,
            sequential_state,
            remaining_chunks,
        )
        decode_seconds = time.perf_counter() - decode_started

        results, feature_seconds, generation_seconds = self._infer_segments(items, segments)
        results = self._attribute_time(results, item_decode_seconds, feature_seconds + generation_seconds)
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

    @staticmethod
    def _attribute_time(
        results: tuple[ChunkResult, ...],
        item_decode_seconds: tuple[float, ...],
        shared_seconds: float,
    ) -> tuple[ChunkResult, ...]:
        """Give each chunk its decode time plus a frame-weighted share of the batch."""

        total_frames = sum(result.item.feature_frames for result in results) or 1
        return tuple(
            dataclasses.replace(
                result,
                processing_seconds=decode_seconds + shared_seconds * result.item.feature_frames / total_frames,
            )
            for result, decode_seconds in zip(results, item_decode_seconds, strict=True)
        )

    def _infer_segments(
        self,
        items: tuple[WorkItem, ...],
        segments: tuple[DecodedSegment, ...],
    ) -> tuple[tuple[ChunkResult, ...], float, float]:
        """Run features and generation on decoded segments; stop on OOM."""

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

        # One device-to-host copy per tensor for the whole batch; indexing
        # rows on the device first would cost one synchronizing copy per row
        # and per field.
        host = _GenerationOnHost.from_device(generation, features_finite)
        results = tuple(
            self._chunk_result(item, segment, host, row_index)
            for row_index, (item, segment) in enumerate(zip(items, segments))
        )
        return results, feature_seconds, generation_seconds

    def _chunk_result(
        self,
        item: WorkItem,
        segment: DecodedSegment,
        host: _GenerationOnHost,
        row_index: int,
    ) -> ChunkResult:
        """Assemble one row's tokens, health facts, and longest word gap."""

        chunk = ChunkResult(
            item=item,
            token_ids=tuple(host.sequences[row_index]),
            durations=tuple(host.durations[row_index]),
            frame_starts=tuple(host.frame_starts[row_index]),
            frame_ends=tuple(host.frame_ends[row_index]),
            encoder_length=host.encoder_lengths[row_index],
            input_finite=segment.finite,
            silent=segment.rms == 0.0,
            features_finite=host.features_finite[row_index],
            encoder_finite=host.encoder_finite[row_index],
            forced_advances=host.forced_advances[row_index],
        )
        word_starts = [word.start_sample for word in self._chunk_words(chunk).words]
        gap = longest_untranscribed_gap(word_starts, item.core_start_frame, item.core_end_frame)
        gap_waveform = segment.waveform[
            max(0, gap.start - item.source_start_frame) : max(0, gap.end - item.source_start_frame)
        ]
        gap_rms = float(torch.sqrt(torch.mean(gap_waveform.square()))) if gap_waveform.numel() else 0.0
        return dataclasses.replace(
            chunk,
            gap_start_sample=gap.start,
            gap_end_sample=gap.end,
            gap_rms=gap_rms,
        )

    # -------------------------------------------------------------------------
    # Decoder collapse recovery
    # -------------------------------------------------------------------------

    def _has_untranscribed_gap(self, chunk: ChunkResult) -> bool:
        """A long stretch of the core with no word whose audio is not silence."""

        return (
            chunk.numerically_valid
            and chunk.gap_samples >= self.gap_threshold_samples
            and chunk.gap_rms >= self.settings.gap_silence_rms
        )

    def _recover_chunk(self, chunk: ChunkResult, metadata: AudioMetadata) -> ChunkResult:
        """
        Re-decode a collapsed chunk with shifted windows; keep the first that
        no longer has an untranscribed gap, otherwise the smallest gap seen.
        """

        item = chunk.item
        file_end = round(metadata.frame_count * self.target_sample_rate / metadata.sample_rate)
        windows = recovery_windows(
            Window(item.source_start_frame, item.source_end_frame),
            Window(item.core_start_frame, item.core_end_frame),
            self.recovery_offsets_samples,
            file_end,
            self.max_chunk_samples,
        )
        best = chunk
        recovery_started = time.perf_counter()
        with open_media_session(item.path) as session:
            for window in windows:
                shifted_item = dataclasses.replace(
                    item,
                    source_start_frame=window.start,
                    source_end_frame=window.end,
                    feature_frames=estimate_feature_frames(
                        window.end - window.start,
                        self.target_sample_rate,
                        self.target_sample_rate,
                        self.hop_length,
                    ),
                )
                segment = session.read_segment(window.start, window.end, self.target_sample_rate)
                (candidate,), _feature_seconds, _generation_seconds = self._infer_segments(
                    (shifted_item,),
                    (segment,),
                )
                candidate = dataclasses.replace(
                    candidate,
                    recovered=True,
                    processing_seconds=chunk.processing_seconds + time.perf_counter() - recovery_started,
                )
                if not self._has_untranscribed_gap(candidate):
                    logger.info(
                        "recovered collapsed chunk %s#%d with window start shifted by %.3f s",
                        item.path.name,
                        item.chunk_index,
                        (window.start - item.source_start_frame) / self.target_sample_rate,
                    )
                    return candidate
                if candidate.gap_samples < best.gap_samples:
                    best = candidate
        best = dataclasses.replace(
            best,
            processing_seconds=chunk.processing_seconds + time.perf_counter() - recovery_started,
        )
        logger.warning(
            "chunk %s#%d keeps an untranscribed gap of %.1f s after %d re-decodes",
            item.path.name,
            item.chunk_index,
            best.gap_samples / self.target_sample_rate,
            len(windows),
        )
        return best

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    def _chunk_words(self, chunk: ChunkResult) -> ChunkWords:
        """Convert one chunk's content tokens to timed words in source samples."""

        origin = chunk.item.source_start_frame
        timed_tokens = [
            TimedToken(
                token_id=token,
                start_sample=origin + start * self.samples_per_encoder_frame,
                end_sample=origin + end * self.samples_per_encoder_frame,
            )
            for token, start, end in zip(chunk.token_ids, chunk.frame_starts, chunk.frame_ends)
            if token not in self.special_token_ids
        ]
        return ChunkWords(
            words=group_words(timed_tokens, self.tokenizer.id_to_token.get),
            source_start=chunk.item.source_start_frame,
            source_end=chunk.item.source_end_frame,
            core_start=chunk.item.core_start_frame,
            core_end=chunk.item.core_end_frame,
        )

    def _merge_file(
        self,
        metadata: AudioMetadata,
        chunks: tuple[ChunkResult, ...],
    ) -> OfflineFileResult:
        """
        Merge a file's chunk transcripts at word level (see ``merging``).

        Neighbouring chunks both transcribe their overlap. Words they agree
        on inside the overlap are taken once; otherwise whole words go to the
        chunk whose core holds their start. Words are never split between
        chunks, and repeated words inside a chunk are never collapsed.
        """

        ordered_chunks = tuple(sorted(chunks, key=lambda chunk: chunk.item.chunk_index))
        chunk_words = tuple(
            self._chunk_words(chunk)
            for chunk in ordered_chunks
            if chunk.numerically_valid
        )
        merged_tokens = merge_chunks(chunk_words, self.merge_tolerance_samples)

        transcript = self.tokenizer.decode(merged_tokens)
        return OfflineFileResult(
            path=metadata.path,
            duration_seconds=metadata.duration_seconds,
            transcript=transcript,
            status=classify_file(
                transcript,
                ordered_chunks,
                untranscribed_gap=any(self._has_untranscribed_gap(chunk) for chunk in ordered_chunks),
            ),
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

        sessions: dict[Path, MediaSession] = {}
        sequential_state: dict[Path, tuple[int, torch.Tensor]] = {}
        chunk_results: dict[int, list[ChunkResult]] = {index: [] for index in range(len(path_tuple))}
        stage_totals = {"decode": 0.0, "feature": 0.0, "generation": 0.0, "recovery": 0.0}
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

        recovery_started = time.perf_counter()
        for index, record in enumerate(metadata_tuple):
            chunk_results[index] = [
                self._recover_chunk(chunk, record) if self._has_untranscribed_gap(chunk) else chunk
                for chunk in chunk_results[index]
            ]
        stage_totals["recovery"] = time.perf_counter() - recovery_started

        file_results = tuple(
            self._merge_file(record, tuple(chunk_results[index]))
            for index, record in enumerate(metadata_tuple)
        )
        for file_result in file_results:
            log_level = logging.INFO if file_result.status is FileStatus.OK else logging.WARNING
            logger.log(
                log_level,
                "file=%s status=%s duration_seconds=%.3f chunks=%d words=%d "
                "processing_seconds=%.3f real_time_factor=%.5f",
                file_result.path.name,
                file_result.status.value,
                file_result.duration_seconds,
                len(file_result.chunks),
                len(file_result.transcript.split()),
                file_result.processing_seconds,
                file_result.real_time_factor,
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
            recovery_seconds=stage_totals["recovery"],
        )
