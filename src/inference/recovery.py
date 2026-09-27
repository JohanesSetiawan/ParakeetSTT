"""
Detection of decoder collapse and the alternative windows used to recover.

Parakeet TDT occasionally "collapses" on a particular audio window: the greedy
decoder emits blank with the longest duration frame after frame and skips
seconds of clear speech. The Hugging Face reference does exactly the same on
the same samples, so it is a property of the model, not of this runtime. It
depends on where the window starts: moving the start by a few hundred
milliseconds usually makes the same speech transcribe normally.

This module holds the two pure pieces of the recovery:

- ``longest_untranscribed_gap`` finds the longest stretch of a chunk's core
  with no word, which is how a collapse shows up in the output;
- ``recovery_windows`` lists alternative windows for re-decoding a chunk that
  still cover its whole core and stay within the chunk budget.

Deciding whether a gap is suspicious (long, and not silence) and running the
re-decode lives in ``offline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Window:
    """A source interval in target-rate samples: ``[start, end)``."""

    start: int
    end: int


def longest_untranscribed_gap(
    word_start_samples: Iterable[int],
    core_start: int,
    core_end: int,
) -> Window:
    """
    Return the longest interval of the core that contains no word start.

    Args:
        word_start_samples: Absolute start of every word the chunk produced.
        core_start: First sample the chunk owns.
        core_end: One past the last sample the chunk owns.

    Returns:
        The gap, clipped to the core. A chunk without words returns the whole
        core.
    """

    inside = sorted(sample for sample in word_start_samples if core_start <= sample < core_end)
    marks = [core_start, *inside, core_end]
    best = Window(core_start, core_start)
    for left, right in zip(marks, marks[1:]):
        if right - left > best.end - best.start:
            best = Window(left, right)
    return best


def recovery_windows(
    source: Window,
    core: Window,
    start_offsets: Sequence[int],
    file_end: int,
    max_samples: int,
) -> list[Window]:
    """
    Alternative windows for re-decoding a chunk, in the given offset order.

    Each candidate moves the window start by one offset and keeps the original
    length where the file allows. A candidate is used only if it still starts
    at or before the core, ends at or after the core, stays inside the file,
    and fits the chunk budget, so every owned sample is still decoded.

    Args:
        source: The chunk's original window.
        core: The interval the chunk owns.
        start_offsets: Start shifts in samples; negative moves earlier.
        file_end: Total target-rate samples of the file.
        max_samples: Largest window the chunk budget allows.

    Returns:
        Valid distinct windows, excluding the original one.
    """

    length = source.end - source.start
    windows: list[Window] = []
    for offset in start_offsets:
        start = source.start + offset
        end = min(start + length, file_end)
        candidate = Window(start, end)
        valid = (
            0 <= start <= core.start
            and end >= core.end
            and end <= file_end
            and end - start <= max_samples
            and candidate != source
            and candidate not in windows
        )
        if valid:
            windows.append(candidate)
    return windows
