"""
Word error rate for English transcripts against references.

Used by the full test tier and by the ``doctor`` command, which checks a
device's transcripts against the committed LibriSpeech references. Scoring
is lowercase and punctuation-free, so the model's cased, punctuated output
and LibriSpeech's uppercase references meet in the same form.
"""

from __future__ import annotations

import re

import numpy as np


# LibriSpeech spells titles out ("MISTER"); the model writes either "mister"
# or "Mr". Mapping both to one form keeps a spelling choice from counting as
# a recognition error. This mirrors the title handling of common English ASR
# normalizers and is limited to titles LibriSpeech actually spells out.
SPELLED_TITLES = {"mr": "mister", "mrs": "missus", "dr": "doctor"}


def normalize_words(text: str) -> list[str]:
    """
    Lowercase, drop punctuation except apostrophes, spell out titles, split.

    LibriSpeech references are uppercase without punctuation, while the model
    writes cased, punctuated text; both must meet in the same form.
    """

    lowered = text.lower().replace("-", " ")
    letters_only = re.sub(r"[^a-z0-9' ]+", " ", lowered)
    return [SPELLED_TITLES.get(word, word) for word in letters_only.split()]


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """
    Return substitutions + deletions + insertions (Levenshtein on words).

    One numpy row per reference word, so hour-long transcripts (tens of
    thousands of words) score in seconds instead of minutes.
    """

    vocabulary = {word: index for index, word in enumerate(set(reference) | set(hypothesis))}
    hypothesis_ids = np.array([vocabulary[word] for word in hypothesis], dtype=np.int64)
    columns = np.arange(len(hypothesis) + 1, dtype=np.int64)
    previous_row = columns.copy()
    for reference_index, reference_word in enumerate(reference, start=1):
        substitution = previous_row[:-1] + (hypothesis_ids != vocabulary[reference_word])
        deletion = previous_row[1:] + 1
        current_row = np.empty_like(previous_row)
        current_row[0] = reference_index
        current_row[1:] = np.minimum(substitution, deletion)
        # Insertions chain left to right: current[j] = min(current[j],
        # current[j - 1] + 1), which is a running minimum of current[j] - j.
        current_row = np.minimum.accumulate(current_row - columns) + columns
        previous_row = current_row
    return int(previous_row[-1])


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Corpus WER: total word errors divided by total reference words."""

    total_errors = 0
    total_words = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        reference_words = normalize_words(reference)
        total_errors += word_errors(reference_words, normalize_words(hypothesis))
        total_words += len(reference_words)
    return total_errors / total_words
