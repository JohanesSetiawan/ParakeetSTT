"""
Accuracy on real speech with the real checkpoint.

The clips and their expected transcripts live in tests/data/librispeech. The
expected transcripts were recorded from this runtime and cross-checked against
Hugging Face ParakeetForTDT (identical for every single-chunk clip).
"""

from __future__ import annotations

import pytest

from src.inference.offline import FileStatus, OfflineTranscriber
from support import SpeechClip, word_error_rate

PAD_TOKEN_ID = 2

# Measured corpus WER over the five clips is 3.67 percent (4 errors in 109
# words, titles normalized), all from the model itself on LibriSpeech text.
# The budget leaves room for one extra word error; anything worse is a real
# regression.
CORPUS_WER_BUDGET = 0.046

# The chunked clip differs from the unchunked reference by one word ("and"
# for "an", 1.49 percent), a context effect rather than a seam artifact.
# Seam fragments such as "smile atile at one" pushed this above 4 percent.
FULL_CONTEXT_WER_BUDGET = 0.03


@pytest.fixture(scope="module")
def transcriber(real_model, real_settings) -> OfflineTranscriber:
    model, configuration = real_model
    return OfflineTranscriber(model, configuration, real_settings.inference)


def single_chunk(clips: tuple[SpeechClip, ...]) -> list[SpeechClip]:
    return [clip for clip in clips if clip.chunks == 1]


def test_single_chunk_clips_match_expected_transcripts(transcriber, speech_clips) -> None:
    for clip in single_chunk(speech_clips):
        file_result = transcriber.transcribe([clip.path]).files[0]

        assert file_result.transcript == clip.expected_transcript, clip.clip_id
        assert file_result.status.value == clip.expected_status
        assert len(file_result.chunks) == 1


def test_chunked_clip_matches_its_recorded_transcript(transcriber, speech_clips) -> None:
    """
    Locks the current multi-chunk output, so any change to chunking or merging
    shows up as a deliberate diff. It includes the known boundary artifacts;
    see test_chunked_clip_matches_full_context_reference.
    """

    for clip in speech_clips:
        if clip.chunks == 1:
            continue
        file_result = transcriber.transcribe([clip.path]).files[0]

        assert len(file_result.chunks) == clip.chunks
        assert file_result.transcript == clip.expected_transcript


def test_chunked_clip_stays_close_to_the_full_context_reference(transcriber, speech_clips) -> None:
    """
    Word-level merging keeps seams clean: no split or repeated words where
    the unchunked reference has none.
    """

    for clip in speech_clips:
        if clip.chunks == 1:
            continue
        file_result = transcriber.transcribe([clip.path]).files[0]
        wer = word_error_rate([clip.full_context_transcript], [file_result.transcript])

        assert wer <= FULL_CONTEXT_WER_BUDGET, f"{clip.clip_id}: {wer:.4f} against the unchunked reference"


def test_corpus_word_error_rate_within_budget(transcriber, speech_clips) -> None:
    result = transcriber.transcribe([clip.path for clip in speech_clips])
    transcripts = {file_result.path.name: file_result.transcript for file_result in result.files}

    wer = word_error_rate(
        [clip.reference_text for clip in speech_clips],
        [transcripts[clip.path.name] for clip in speech_clips],
    )

    assert wer <= CORPUS_WER_BUDGET, f"corpus WER {wer:.4f}"


def strip_trailing_padding(tokens: tuple[int, ...], durations: tuple[int, ...]):
    token_list, duration_list = list(tokens), list(durations)
    while token_list and token_list[-1] == PAD_TOKEN_ID and duration_list[-1] == 0:
        token_list.pop()
        duration_list.pop()
    return token_list, duration_list


def test_batching_does_not_change_any_token(transcriber, speech_clips) -> None:
    """Each clip alone and all clips as one mixed batch give identical tokens and durations."""

    batched = transcriber.transcribe([clip.path for clip in speech_clips])
    batched_by_name = {file_result.path.name: file_result for file_result in batched.files}

    for clip in speech_clips:
        alone = transcriber.transcribe([clip.path]).files[0]
        together = batched_by_name[clip.path.name]

        assert alone.transcript == together.transcript, clip.clip_id
        for alone_chunk, together_chunk in zip(alone.chunks, together.chunks, strict=True):
            assert strip_trailing_padding(alone_chunk.token_ids, alone_chunk.durations) == strip_trailing_padding(
                together_chunk.token_ids,
                together_chunk.durations,
            )


def test_repeated_runs_are_deterministic(transcriber, speech_clips) -> None:
    paths = [clip.path for clip in speech_clips]

    first = transcriber.transcribe(paths).files
    second = transcriber.transcribe(paths).files

    assert [file_result.transcript for file_result in first] == [file_result.transcript for file_result in second]
    assert all(file_result.status is FileStatus.OK for file_result in first)
