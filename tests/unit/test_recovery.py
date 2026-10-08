"""Decoder-collapse detection and recovery (src/inference/recovery.py and offline wiring)."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import torch

from src.audio.media import DecodedSegment
from src.inference.offline import ChunkResult, FileStatus, OfflineTranscriber
from src.inference.planning import WorkItem
from src.inference.recovery import Window, longest_untranscribed_gap, recovery_windows
from support import SteadySpeechModel, inference_settings, write_float_wav

CORE = Window(10_000, 30_000)


# =============================================================================
# Pure functions
# =============================================================================


def test_longest_gap_is_found_between_words_and_core_edges() -> None:
    gap = longest_untranscribed_gap([12_000, 14_000, 25_000, 40_000], CORE.start, CORE.end)

    assert gap == Window(14_000, 25_000)


def test_chunk_without_words_is_one_gap_over_the_whole_core() -> None:
    assert longest_untranscribed_gap([], CORE.start, CORE.end) == CORE


def test_words_outside_the_core_do_not_close_a_gap() -> None:
    gap = longest_untranscribed_gap([5_000, 29_000, 31_000], CORE.start, CORE.end)

    assert gap == Window(CORE.start, 29_000)


def test_middle_chunk_gets_both_shift_directions() -> None:
    windows = recovery_windows(Window(8_000, 32_000), CORE, [-1_000, 1_000, -3_000], 100_000, 24_000)

    # -3000 would end at 29_000, before the core ends, so it is skipped.
    assert windows == [Window(7_000, 31_000), Window(9_000, 33_000)]


def test_first_chunk_cannot_move_its_start() -> None:
    first_core = Window(0, 20_000)

    windows = recovery_windows(Window(0, 22_000), first_core, [-1_000, 1_000], 100_000, 24_000)

    assert windows == []


def test_last_chunk_may_only_start_later() -> None:
    last_core = Window(10_000, 30_000)

    windows = recovery_windows(Window(8_000, 30_000), last_core, [-1_000, 1_000], 30_000, 24_000)

    assert windows == [Window(9_000, 30_000)]


def test_windows_never_exceed_the_chunk_budget() -> None:
    windows = recovery_windows(Window(8_000, 32_000), CORE, [-1_000, 1_000], 100_000, 23_999)

    assert windows == []


# =============================================================================
# Offline wiring
# =============================================================================


def recovery_settings(**overrides: object):
    values: dict[str, object] = {
        "batch_size": 1,
        "max_chunk_feature_frames": 100,
        "overlap_feature_frames": 5,
        "untranscribed_gap_seconds": 0.5,
        "recovery_start_offsets_feature_frames": (-3, 3),
    }
    values.update(overrides)
    return inference_settings(**values)


def transcribe(tmp_path: Path, tiny_configuration, model, **overrides: object):
    path = tmp_path / "speech.wav"
    write_float_wav(path, torch.full((48_000,), 0.1), 16000)
    transcriber = OfflineTranscriber(model.eval(), tiny_configuration, recovery_settings(**overrides))
    return transcriber.transcribe([path]).files[0]


def test_collapsed_middle_chunk_is_recovered(tmp_path, tiny_configuration) -> None:
    file_result = transcribe(tmp_path, tiny_configuration, SteadySpeechModel(3, collapse_calls=frozenset({1})))

    assert len(file_result.chunks) >= 3
    assert file_result.chunks[1].recovered
    assert file_result.status is FileStatus.OK


def test_unrecoverable_collapse_is_reported(tmp_path, tiny_configuration) -> None:
    """The first chunk has no left overlap to shift into, so it cannot be re-decoded."""

    file_result = transcribe(tmp_path, tiny_configuration, SteadySpeechModel(3, collapse_calls=frozenset({0})))

    assert not file_result.chunks[0].recovered
    assert file_result.status is FileStatus.UNTRANSCRIBED_GAP


def test_recovery_disabled_by_empty_offsets(tmp_path, tiny_configuration) -> None:
    file_result = transcribe(
        tmp_path,
        tiny_configuration,
        SteadySpeechModel(3, collapse_calls=frozenset({1})),
        recovery_start_offsets_feature_frames=(),
    )

    assert not any(chunk.recovered for chunk in file_result.chunks)
    assert file_result.status is FileStatus.UNTRANSCRIBED_GAP


def test_steady_speech_is_never_re_decoded(tmp_path, tiny_configuration) -> None:
    model = SteadySpeechModel(3)

    file_result = transcribe(tmp_path, tiny_configuration, model)

    assert model.calls == len(file_result.chunks)
    assert file_result.status is FileStatus.OK


def test_a_window_that_works_first_costs_one_re_decode(tmp_path, tiny_configuration) -> None:
    model = SteadySpeechModel(3, collapse_calls=frozenset({1}))

    file_result = transcribe(
        tmp_path,
        tiny_configuration,
        model,
        recovery_start_offsets_feature_frames=(-3, 3, -5, 5),
    )

    assert file_result.chunks[1].recovered
    assert model.calls == len(file_result.chunks) + 1


def test_windows_are_tried_in_order_until_one_works(tmp_path, tiny_configuration) -> None:
    # Call 1 is chunk 1, call 2 its first window (-3 frames): both collapse.
    model = SteadySpeechModel(3, collapse_calls=frozenset({1, 2}))

    file_result = transcribe(
        tmp_path,
        tiny_configuration,
        model,
        recovery_start_offsets_feature_frames=(-3, 3, -5, 5),
    )

    chunk = file_result.chunks[1]
    original_start = chunk.item.core_start_frame - 5 * 160  # overlap of 5 frames
    assert chunk.recovered
    assert chunk.item.source_start_frame == original_start + 3 * 160
    assert model.calls == len(file_result.chunks) + 2
    assert file_result.status is FileStatus.OK


# =============================================================================
# Recovery batches
# =============================================================================


SOURCE = Window(15_000, 30_999)  # 15999 samples, the 100-frame chunk budget
CHUNK_CORE = Window(15_800, 30_199)


def collapsed_item(chunk_index: int = 1) -> WorkItem:
    return WorkItem(0, Path("a.wav"), chunk_index, SOURCE.start, SOURCE.end, CHUNK_CORE.start, CHUNK_CORE.end, 100, 0)


def result_with_gap(item: WorkItem, gap_samples: int) -> ChunkResult:
    """A row whose longest wordless stretch is ``gap_samples`` of clearly audible audio."""

    return ChunkResult(
        item=item,
        token_ids=(),
        durations=(),
        frame_starts=(),
        frame_ends=(),
        encoder_length=0,
        input_finite=True,
        silent=False,
        features_finite=True,
        encoder_finite=True,
        forced_advances=0,
        gap_start_sample=item.core_start_frame,
        gap_end_sample=item.core_start_frame + gap_samples,
        gap_rms=0.1,
    )


def span_around(item: WorkItem, reach: int) -> DecodedSegment:
    start = item.source_start_frame - reach
    end = item.source_end_frame + reach
    return DecodedSegment(torch.full((end - start,), 0.1), 16000, start, end, 16000, 1, 0.1, True)


def recovery_transcriber(tiny_configuration, scripted_gaps: dict[int, int], rows_seen: list[int]):
    """
    A transcriber with 4-row recovery batches whose model is replaced by a
    script: the gap of each window, keyed by its start shift in samples.
    """

    transcriber = OfflineTranscriber(
        SteadySpeechModel(3).eval(),
        tiny_configuration,
        recovery_settings(
            batch_size=4,
            max_batch_feature_frames=400,
            recovery_start_offsets_feature_frames=(-3, 3, -5, 5),
        ),
    )

    def scripted(items, segments):
        rows_seen.append(len(items))
        for item, segment in zip(items, segments, strict=True):
            # Windows are cut from the span the chunk was decoded with.
            assert (segment.source_start_frame, segment.source_end_frame) == (
                item.source_start_frame,
                item.source_end_frame,
            )
        results = tuple(
            result_with_gap(item, scripted_gaps[item.source_start_frame - SOURCE.start]) for item in items
        )
        return results, 0.0, 0.0

    transcriber._infer_segments = scripted
    return transcriber


def test_a_lone_chunk_tries_all_windows_in_one_batch_and_keeps_the_first_that_works(
    tiny_configuration,
) -> None:
    rows_seen: list[int] = []
    gaps = {-480: 12_000, 480: 0, -800: 0, 800: 0}
    transcriber = recovery_transcriber(tiny_configuration, gaps, rows_seen)
    item = collapsed_item()
    waiting = transcriber._start_recovery(result_with_gap(item, 14_000), span_around(item, 800), 100_000)
    queue = deque([waiting])

    finished, _seconds = transcriber._run_recovery_batch(queue)

    assert rows_seen == [4]
    assert not queue
    assert len(finished) == 1
    assert finished[0].recovered
    assert finished[0].item.source_start_frame == SOURCE.start + 480


def test_unrecovered_chunks_keep_the_smallest_gap_after_every_window(tiny_configuration) -> None:
    rows_seen: list[int] = []
    gaps = {-480: 12_000, 480: 9_000, -800: 11_000, 800: 10_000}
    transcriber = recovery_transcriber(tiny_configuration, gaps, rows_seen)
    first, second = collapsed_item(1), collapsed_item(2)
    queue = deque(
        transcriber._start_recovery(result_with_gap(item, 14_000), span_around(item, 800), 100_000)
        for item in (first, second)
    )

    finished: list[ChunkResult] = []
    while queue:
        batch_finished, _seconds = transcriber._run_recovery_batch(queue)
        finished.extend(batch_finished)

    # Two chunks share 4 rows: two windows each per batch, two batches.
    assert rows_seen == [4, 4]
    assert sorted(chunk.item.chunk_index for chunk in finished) == [1, 2]
    for chunk in finished:
        assert chunk.recovered
        assert chunk.gap_samples == 9_000
        assert chunk.item.source_start_frame == SOURCE.start + 480


def test_more_waiting_chunks_than_rows_wait_for_the_next_batch(tiny_configuration) -> None:
    rows_seen: list[int] = []
    gaps = {-480: 0, 480: 0, -800: 0, 800: 0}
    transcriber = recovery_transcriber(tiny_configuration, gaps, rows_seen)
    items = [collapsed_item(index) for index in range(1, 7)]
    queue = deque(
        transcriber._start_recovery(result_with_gap(item, 14_000), span_around(item, 800), 100_000)
        for item in items
    )

    finished, _seconds = transcriber._run_recovery_batch(queue)

    assert rows_seen == [4]
    assert [chunk.item.chunk_index for chunk in finished] == [1, 2, 3, 4]
    assert [entry.chunk.item.chunk_index for entry in queue] == [5, 6]
