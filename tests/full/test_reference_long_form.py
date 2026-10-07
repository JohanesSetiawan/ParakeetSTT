"""
Word error rate on your own labeled recordings, long-form included (opt-in).

The committed LibriSpeech clips are short, so chunking, seam merging, and
collapse recovery barely show in their WER. This test scores a local folder
of labeled recordings, which may be tens of minutes long, so any change that
touches accuracy can be measured where it matters. The audio stays outside
the repository.

    PARAKEET_TEST_REFERENCE_DIR      folder with the audio files and transcript.txt
    PARAKEET_TEST_REFERENCE_MAX_WER  corpus WER budget in percent, for example 7.0

transcript.txt has one ``<file name><tab or spaces><reference text>`` line per
file; with tabs, columns after the second are ignored. Every listed file must
exist in the folder.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.inference.offline import FileStatus, OfflineTranscriber
from support import load_reference_transcripts, normalize_words, word_errors

REFERENCE_DIR_VARIABLE = "PARAKEET_TEST_REFERENCE_DIR"
MAX_WER_VARIABLE = "PARAKEET_TEST_REFERENCE_MAX_WER"
REFERENCE_FILENAME = "transcript.txt"

# A long transcript whose length is far from the reference's points to text
# duplicated at chunk seams (the Hugging Face pipeline's chunked TDT path
# measured 1.46 times the reference words on two podcasts) or to dropped
# stretches. This runtime measured 0.99 to 1.00 on the same recordings.
MIN_WORD_RATIO = 0.85
MAX_WORD_RATIO = 1.15
# The ratio is only meaningful on long transcripts; short clips vary by a few
# words either way.
RATIO_MIN_REFERENCE_WORDS = 200

# Share of chunks the collapse recovery may re-decode. Measured 3.8 percent
# (12 of 316) on a 74-minute podcast; a numerical fault that empties chunks
# wholesale (the float16 depthwise convolution bug) pushed it to 97 percent.
MAX_RECOVERED_FRACTION = 0.10

FAILURE_STATUSES = {
    FileStatus.NUMERICAL_FAILURE,
    FileStatus.DECODER_FORCED_ADVANCE,
    FileStatus.EMPTY_TRANSCRIPT,
}


def reference_set() -> tuple[Path, dict[str, str], float] | None:
    """The configured folder, its references, and the WER budget, or None."""

    directory = os.environ.get(REFERENCE_DIR_VARIABLE)
    if not directory:
        return None
    folder = Path(directory).expanduser().resolve()
    transcript = folder / REFERENCE_FILENAME
    if not transcript.is_file():
        pytest.fail(f"{REFERENCE_DIR_VARIABLE} is set but {transcript} does not exist")

    raw_budget = os.environ.get(MAX_WER_VARIABLE)
    try:
        budget = float(raw_budget) if raw_budget else None
    except ValueError:
        budget = None
    if budget is None or budget <= 0:
        pytest.fail(f"{REFERENCE_DIR_VARIABLE} is set, so {MAX_WER_VARIABLE} must be a positive percentage")

    references = load_reference_transcripts(transcript)
    missing = [name for name in references if not (folder / name).is_file()]
    if missing:
        pytest.fail(f"Audio listed in {transcript} is missing: {missing[:5]}")
    return folder, references, budget


@pytest.mark.skipif(
    not os.environ.get(REFERENCE_DIR_VARIABLE),
    reason=f"set {REFERENCE_DIR_VARIABLE} and {MAX_WER_VARIABLE} to score your own labeled recordings",
)
def test_labeled_recordings_stay_within_the_wer_budget(real_settings, real_model) -> None:
    folder, references, budget = reference_set()
    model, configuration = real_model
    names = sorted(references)
    transcriber = OfflineTranscriber(model, configuration, real_settings.inference)

    result = transcriber.transcribe([folder / name for name in names])

    total_errors = 0
    total_words = 0
    total_chunks = 0
    recovered_chunks = 0
    lines = []
    problems = []
    for name, file_result in zip(names, result.files, strict=True):
        reference_words = normalize_words(references[name])
        hypothesis_words = normalize_words(file_result.transcript)
        errors = word_errors(reference_words, hypothesis_words)
        total_errors += errors
        total_words += len(reference_words)
        total_chunks += len(file_result.chunks)
        recovered_chunks += sum(chunk.recovered for chunk in file_result.chunks)
        ratio = len(hypothesis_words) / max(1, len(reference_words))
        lines.append(
            f"{name}: WER {100 * errors / max(1, len(reference_words)):.2f}% "
            f"({len(reference_words)} words, ratio {ratio:.3f}, {file_result.status.value})"
        )
        if file_result.status in FAILURE_STATUSES:
            problems.append(f"{name} has status {file_result.status.value}")
        if len(reference_words) >= RATIO_MIN_REFERENCE_WORDS and not MIN_WORD_RATIO <= ratio <= MAX_WORD_RATIO:
            problems.append(f"{name} has {ratio:.2f} times the reference word count")

    corpus_wer = 100 * total_errors / total_words
    if recovered_chunks > MAX_RECOVERED_FRACTION * total_chunks:
        problems.append(f"{recovered_chunks} of {total_chunks} chunks needed collapse recovery")
    summary = [
        f"recovered chunks {recovered_chunks} of {total_chunks}",
        f"corpus WER {corpus_wer:.2f}% (budget {budget:.2f}%)",
    ]
    report = "\n".join(lines + summary)
    print(report)

    assert not problems, "\n".join(problems) + "\n" + report
    assert corpus_wer <= budget, report
