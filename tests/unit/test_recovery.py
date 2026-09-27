"""Decoder-collapse detection and recovery (src/inference/recovery.py and offline wiring)."""

from __future__ import annotations

from pathlib import Path

import torch

from src.inference.offline import FileStatus, OfflineTranscriber
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
