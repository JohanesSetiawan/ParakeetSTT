"""
Word-level merging of overlapping chunk transcripts.

Neighbouring chunks both transcribe their shared overlap audio. The timestamps
the two chunks assign to the same word can disagree by several hundred
milliseconds, because each chunk sees different context. Deciding ownership
token by token on those timestamps can therefore split one word across two
chunks and repeat part of it ("smile atile at one").

This module merges at word level instead:

1. Tokens are grouped into words. A word starts at a token whose piece begins
   with the SentencePiece word marker; punctuation and continuation pieces
   stay with the word before them. A word is never split between chunks.
2. Inside the overlap, the tail words of the earlier chunk and the head words
   of the later chunk are aligned. The longest run of words that both chunks
   agree on (compared letters and digits only, case-insensitively, within a
   time tolerance) marks the seam. Each agreed word is taken once, from the
   chunk whose core contains it, which is the chunk with more context there.
3. Without an agreed run (no overlap, or the chunks disagree), whole words are
   assigned by start time to the chunk whose core contains them.

Only words that both chunks produced for the same stretch of audio are ever
merged into one. Repetitions inside one chunk are left alone, because
repeated words are part of natural speech.

The module has no torch dependency: it works on plain integers and strings,
so its behavior is fully testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

WORD_MARKER = "▁"


@dataclass(frozen=True)
class TimedToken:
    """One content token with its source-sample span."""

    token_id: int
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class Word:
    """
    Consecutive tokens forming one written word, with a comparison key.

    Attributes:
        tokens: The tokens in emission order.
        start_sample: Source sample where the first token starts.
        key: Lowercased letters and digits of the word, for comparing words
            across chunks; empty for pure punctuation.
    """

    tokens: tuple[TimedToken, ...]
    start_sample: int
    key: str


@dataclass(frozen=True)
class ChunkWords:
    """The words of one chunk plus the interval bounds needed to merge it."""

    words: tuple[Word, ...]
    source_start: int
    source_end: int
    core_start: int
    core_end: int


def group_words(
    tokens: Sequence[TimedToken],
    piece_for: Callable[[int], str | None],
) -> tuple[Word, ...]:
    """
    Group content tokens into words.

    Args:
        tokens: Content tokens (no blank or padding) in emission order.
        piece_for: Returns the tokenizer piece for a token ID, or None for an
            unknown ID. Unknown IDs are skipped, matching the decoder.

    Returns:
        Words in order. A token whose piece starts with the word marker opens
        a new word; any other token extends the current word.
    """

    words: list[Word] = []
    current: list[TimedToken] = []
    current_text: list[str] = []

    def flush() -> None:
        if current:
            words.append(
                Word(
                    tokens=tuple(current),
                    start_sample=current[0].start_sample,
                    key=_comparison_key("".join(current_text)),
                )
            )

    for token in tokens:
        piece = piece_for(token.token_id)
        if piece is None:
            continue
        if piece.startswith(WORD_MARKER) and current:
            flush()
            current = []
            current_text = []
        current.append(token)
        current_text.append(piece)
    flush()
    return tuple(words)


def _comparison_key(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def _longest_agreed_run(
    earlier: Sequence[Word],
    later: Sequence[Word],
    tolerance_samples: int,
) -> tuple[int, int, int] | None:
    """
    Return ``(earlier_index, later_index, length)`` of the longest contiguous
    run of equal, non-empty keys whose paired start times differ by at most
    ``tolerance_samples``; None when no pair agrees. Ties prefer the run that
    ends latest in the earlier chunk, which is closest to the seam.
    """

    best: tuple[int, int, int] | None = None
    run_lengths = [[0] * (len(later) + 1) for _ in range(len(earlier) + 1)]
    for earlier_index in range(1, len(earlier) + 1):
        earlier_word = earlier[earlier_index - 1]
        for later_index in range(1, len(later) + 1):
            later_word = later[later_index - 1]
            agrees = (
                earlier_word.key
                and earlier_word.key == later_word.key
                and abs(earlier_word.start_sample - later_word.start_sample) <= tolerance_samples
            )
            if not agrees:
                continue
            length = run_lengths[earlier_index - 1][later_index - 1] + 1
            run_lengths[earlier_index][later_index] = length
            if best is None or length >= best[2]:
                best = (earlier_index - length, later_index - length, length)
    return best


def merge_chunks(chunks: Sequence[ChunkWords], tolerance_samples: int) -> list[int]:
    """
    Merge chunk transcripts of one file into a single token sequence.

    Args:
        chunks: Numerically valid chunks in time order.
        tolerance_samples: How far apart two chunks' timestamps for the same
            word may be. Also widens the overlap windows searched for the
            seam. Zero disables alignment, leaving start-time ownership only.

    Returns:
        Content token IDs of the merged transcript, in order.
    """

    if not chunks:
        return []

    merged: list[Word] = list(chunks[0].words)
    previous = chunks[0]
    previous_start_index = 0  # index in `merged` of the first word from `previous`

    for chunk in chunks[1:]:
        seam = previous.core_end
        earlier_region_start = chunk.source_start - tolerance_samples
        later_region_end = previous.source_end + tolerance_samples

        tail_indices = [
            index
            for index in range(previous_start_index, len(merged))
            if merged[index].start_sample >= earlier_region_start
        ]
        head = [word for word in chunk.words if word.start_sample < later_region_end]
        run = None
        if tolerance_samples > 0 and tail_indices and head:
            run = _longest_agreed_run(
                [merged[index] for index in tail_indices],
                head,
                tolerance_samples,
            )

        if run is not None:
            tail_offset, head_offset, length = run
            first_merged_index = tail_indices[tail_offset]
            agreed: list[Word] = []
            for step in range(length):
                earlier_word = merged[tail_indices[tail_offset + step]]
                later_word = head[head_offset + step]
                agreed.append(earlier_word if earlier_word.start_sample < seam else later_word)
            remainder = list(chunk.words[head_offset + length :])
            merged = merged[:first_merged_index] + agreed + remainder
            next_start_index = first_merged_index
        else:
            kept = [
                word
                for index, word in enumerate(merged)
                if index < previous_start_index or word.start_sample < previous.core_end
            ]
            incoming = [word for word in chunk.words if word.start_sample >= chunk.core_start]
            merged = kept + incoming
            next_start_index = len(kept)

        previous = chunk
        previous_start_index = next_start_index

    return [token.token_id for word in merged for token in word.tokens]
