"""
Long-form behavior on real speech: chunking at scale, memory bound, accuracy.

Long inputs are built by concatenating the committed clips with 0.5 s of
silence between them, so the reference text is known exactly. Set
PARAKEET_TEST_LONG_AUDIO to one or more extra files (separated by the OS path
separator) to also run your own long recordings through the same checks.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from src.audio.media import inspect_media
from src.inference.offline import FileStatus, OfflineTranscriber
from support import SpeechClip, word_error_rate

SAMPLE_RATE = 16000
GAP = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)

# Measured with word-level merging and collapse recovery: 4.59 percent WER at
# 4 passes (196 s, 15 chunks), against 3.67 percent for the same clips
# transcribed one by one (titles normalized). Before those fixes it was 13.30
# percent. The budget allows about four extra word errors in 436 words.
LONG_FORM_WER_BUDGET = 0.055

# How much worse long-form transcription may be than clip-by-clip, in WER
# points. Chunk seams and a different context per chunk cost a little; more
# than this means seams are losing or repeating words again.
CHUNKING_COST_BUDGET = 0.015


def write_concatenation(path: Path, clips: tuple[SpeechClip, ...], repeats: int) -> str:
    """Write the clips `repeats` times with gaps; return the reference text."""

    pieces: list[np.ndarray] = []
    for clip in clips:
        audio, rate = soundfile.read(str(clip.path), dtype="float32")
        assert rate == SAMPLE_RATE
        pieces += [audio, GAP]
    soundfile.write(str(path), np.tile(np.concatenate(pieces), repeats), SAMPLE_RATE, subtype="PCM_16")
    return " ".join([clip.reference_text for clip in clips] * repeats)


@pytest.fixture(scope="module")
def transcriber(real_model, real_settings) -> OfflineTranscriber:
    model, configuration = real_model
    return OfflineTranscriber(model, configuration, real_settings.inference)


def test_long_recording_is_chunked_and_stays_accurate(transcriber, speech_clips, tmp_path: Path) -> None:
    path = tmp_path / "long.wav"
    reference = write_concatenation(path, speech_clips, repeats=4)

    file_result = transcriber.transcribe([path]).files[0]
    wer = word_error_rate([reference], [file_result.transcript])

    assert file_result.status is FileStatus.OK
    assert len(file_result.chunks) >= 10
    assert wer <= LONG_FORM_WER_BUDGET, f"long-form WER {wer:.4f}"


def test_chunking_costs_almost_no_accuracy(transcriber, speech_clips, tmp_path: Path) -> None:
    path = tmp_path / "long.wav"
    reference = write_concatenation(path, speech_clips, repeats=4)
    per_clip = transcriber.transcribe([clip.path for clip in speech_clips])
    per_clip_wer = word_error_rate(
        [clip.reference_text for clip in speech_clips],
        [file_result.transcript for file_result in per_clip.files],
    )

    long_form = transcriber.transcribe([path]).files[0]

    long_form_wer = word_error_rate([reference], [long_form.transcript])
    assert long_form_wer <= per_clip_wer + CHUNKING_COST_BUDGET, (
        f"long-form {long_form_wer:.4f} vs clip-by-clip {per_clip_wer:.4f}"
    )


def test_no_speech_chunk_comes_back_empty_in_full_batches(transcriber, speech_clips, tmp_path: Path) -> None:
    """
    Every chunk of real speech decoded in full-size batches has tokens.

    cuDNN's float16 depthwise convolution returned garbage for some inputs in
    batches of 7 or more rows, and whole chunks came back without a single
    token (305 of 316 on a podcast). That input is not in the repository, so
    this checks the general property; the structural fix (depthwise
    convolutions stay float32) is checked in test_derived_checkpoint, and
    the labeled long-form test limits how many chunks may need recovery.
    """

    path = tmp_path / "long.wav"
    write_concatenation(path, speech_clips, repeats=8)
    special_tokens = {transcriber.configuration.blank_token_id, transcriber.configuration.pad_token_id}

    result = transcriber.transcribe([path])

    rows = max(len(batch) for batch in result.plan.batches)
    empty = [
        chunk.item.chunk_index
        for chunk in result.files[0].chunks
        if not any(token not in special_tokens for token in chunk.token_ids)
    ]
    assert rows >= min(8, transcriber.settings.batch_size), f"batches only reached {rows} rows"
    assert empty == [], f"chunks without tokens: {empty}"


@pytest.mark.cuda
def test_peak_memory_does_not_grow_with_duration(transcriber, speech_clips, tmp_path: Path) -> None:
    # Both recordings must fill at least one whole batch; below that, memory
    # rightly grows with the rows in use. Past it, the batch budget alone sets
    # the peak, whatever the duration.
    settings = transcriber.settings
    rows_per_batch = max(1, settings.max_batch_feature_frames // settings.max_chunk_feature_frames)

    def written_with_chunks(repeats: int) -> tuple[Path, int]:
        path = tmp_path / f"long_{repeats}.wav"
        write_concatenation(path, speech_clips, repeats=repeats)
        chunks = len(transcriber._plan((inspect_media(path),)).items)
        return path, chunks

    repeats = 1
    shorter, chunks = written_with_chunks(repeats)
    while chunks < rows_per_batch:
        repeats *= 2
        shorter, chunks = written_with_chunks(repeats)
    longer, _ = written_with_chunks(3 * repeats)

    peaks = []
    for path in (shorter, longer):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        transcriber.transcribe([path])

        peaks.append(torch.cuda.max_memory_allocated())

    growth_mib = (peaks[1] - peaks[0]) / 2**20
    assert growth_mib <= 16, f"peak grew by {growth_mib:.1f} MiB for 3x the audio"


def extra_long_recordings() -> list[Path]:
    value = os.environ.get("PARAKEET_TEST_LONG_AUDIO", "")
    return [Path(part) for part in value.split(os.pathsep) if part]


@pytest.mark.skipif(not extra_long_recordings(), reason="set PARAKEET_TEST_LONG_AUDIO to run on your own long files")
@pytest.mark.parametrize("path", extra_long_recordings(), ids=lambda path: path.name)
def test_user_supplied_long_recording(transcriber, path: Path) -> None:
    result = transcriber.transcribe([path])
    file_result = result.files[0]

    assert file_result.status is FileStatus.OK
    assert len(file_result.transcript.split()) > 0
    # Real-time factor is reported, not asserted: it depends on the hardware.
    print(f"{path.name}: {file_result.duration_seconds:.1f} s audio, RTF {result.real_time_factor:.4f}")
