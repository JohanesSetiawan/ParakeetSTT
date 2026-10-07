"""The test-suite scoring helpers: word errors and reference transcript files."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from support import load_reference_transcripts, word_error_rate, word_errors


def plain_levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_word in enumerate(reference, start=1):
        current = [row] + [0] * len(hypothesis)
        for column, hypothesis_word in enumerate(hypothesis, start=1):
            current[column] = min(
                previous[column - 1] + (reference_word != hypothesis_word),
                previous[column] + 1,
                current[column - 1] + 1,
            )
        previous = current
    return previous[-1]


@pytest.mark.parametrize("seed", range(4))
def test_word_errors_match_the_textbook_recurrence(seed: int) -> None:
    generator = random.Random(seed)
    for _ in range(200):
        reference = [generator.choice("abcd") for _ in range(generator.randint(0, 12))]
        hypothesis = [generator.choice("abcde") for _ in range(generator.randint(0, 12))]
        assert word_errors(reference, hypothesis) == plain_levenshtein(reference, hypothesis)


def test_word_error_rate_counts_every_error_kind() -> None:
    # One substitution (sat -> sit), one deletion (the), one insertion (down).
    assert word_error_rate(["the cat sat on the mat"], ["cat sit on the mat down"]) == pytest.approx(3 / 6)


def test_reference_file_accepts_tab_and_space_separated_lines(tmp_path: Path) -> None:
    path = tmp_path / "transcript.txt"
    path.write_text(
        "clip.wav\tHello there.\thello there\t11\t16000\tMALE\n"
        "\n"
        "long.mp3   Hey, what's going on? I did.\n",
        encoding="utf-8",
    )

    assert load_reference_transcripts(path) == {
        "clip.wav": "Hello there.",
        "long.mp3": "Hey, what's going on? I did.",
    }
