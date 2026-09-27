"""Word-level merging of overlapping chunk transcripts (src/inference/merging.py)."""

from __future__ import annotations

import pytest

from src.inference.merging import ChunkWords, TimedToken, group_words, merge_chunks

RATE = 16000
MARK = "▁"


class Vocabulary:
    """Assigns token IDs to pieces on first use so tests can write text."""

    def __init__(self) -> None:
        self.piece_by_id: dict[int, str] = {}
        self.id_by_piece: dict[str, int] = {}

    def token_id(self, piece: str) -> int:
        if piece not in self.id_by_piece:
            new_id = len(self.id_by_piece) + 10
            self.id_by_piece[piece] = new_id
            self.piece_by_id[new_id] = piece
        return self.id_by_piece[piece]

    def text(self, token_ids: list[int]) -> str:
        return "".join(self.piece_by_id[token_id] for token_id in token_ids).replace(MARK, " ").strip()


def seconds(value: float) -> int:
    return round(value * RATE)


def chunk(
    vocabulary: Vocabulary,
    pieces: list[tuple[str, float]],
    source: tuple[float, float],
    core: tuple[float, float],
) -> ChunkWords:
    """Build a chunk from (piece, start seconds) pairs; each token lasts 80 ms."""

    tokens = [
        TimedToken(vocabulary.token_id(piece), seconds(start), seconds(start + 0.08))
        for piece, start in pieces
    ]
    return ChunkWords(
        words=group_words(tokens, vocabulary.piece_by_id.get),
        source_start=seconds(source[0]),
        source_end=seconds(source[1]),
        core_start=seconds(core[0]),
        core_end=seconds(core[1]),
    )


TOLERANCE = seconds(1.0)


# =============================================================================
# Word grouping
# =============================================================================


def test_words_start_at_the_marker_and_keep_punctuation_and_continuations() -> None:
    vocabulary = Vocabulary()
    pieces = [f"{MARK}L", "inn", "ell", "'", "s", f"{MARK}bath", ",", f"{MARK}ne", "xt", "."]
    tokens = [TimedToken(vocabulary.token_id(piece), index, index + 1) for index, piece in enumerate(pieces)]

    words = group_words(tokens, vocabulary.piece_by_id.get)

    assert [word.key for word in words] == ["linnells", "bath", "next"]
    assert [len(word.tokens) for word in words] == [5, 2, 3]
    assert [word.start_sample for word in words] == [0, 5, 7]


def test_unknown_token_ids_are_skipped() -> None:
    vocabulary = Vocabulary()
    known = vocabulary.token_id(f"{MARK}hello")
    tokens = [TimedToken(known, 0, 1), TimedToken(999, 1, 2)]

    words = group_words(tokens, vocabulary.piece_by_id.get)

    assert [word.key for word in words] == ["hello"]
    assert len(words[0].tokens) == 1


# =============================================================================
# Seams observed on real speech (tests/data/librispeech/1272-128104-0004.flac)
# =============================================================================


def test_drifted_timestamps_do_not_split_or_repeat_a_word() -> None:
    """
    Chunk 0 placed "smile at one" at 13.52-14.00 s, chunk 1 at 13.74-14.46 s.
    Token-midpoint ownership produced "smile atile at one".
    """

    vocabulary = Vocabulary()
    earlier = chunk(
        vocabulary,
        [(f"{MARK}land", 12.80), ("s", 12.96), ("c", 13.04), ("ap", 13.20), ("es", 13.36),
         (f"{MARK}sm", 13.52), ("ile", 13.68), (f"{MARK}at", 13.84), (f"{MARK}one", 14.00)],
        source=(0.0, 14.5),
        core=(0.0, 14.0),
    )
    later = chunk(
        vocabulary,
        [(f"{MARK}sm", 13.74), ("ile", 13.98), (f"{MARK}at", 14.22), (f"{MARK}one", 14.46),
         (f"{MARK}much", 14.70), (f"{MARK}in", 14.94)],
        source=(13.5, 28.5),
        core=(14.0, 28.0),
    )

    merged = merge_chunks([earlier, later], TOLERANCE)

    assert vocabulary.text(merged) == "landscapes smile at one much in"


def test_word_cut_at_the_chunk_edge_is_taken_once() -> None:
    """Chunk 1 ended inside "next" ("ne" + "xt"); chunk 2 heard "Next man.". Old output: "ne Next man"."""

    vocabulary = Vocabulary()
    earlier = chunk(
        vocabulary,
        [(f"{MARK}b", 27.34), ("ath", 27.50), (",", 27.66), (f"{MARK}ne", 27.82), ("xt", 27.98)],
        source=(13.5, 28.5),
        core=(14.0, 28.0),
    )
    later = chunk(
        vocabulary,
        [(f"{MARK}Ne", 28.14), ("xt", 28.38), (f"{MARK}man", 28.70), (".", 28.86)],
        source=(27.5, 29.4),
        core=(28.0, 29.4),
    )

    merged = merge_chunks([earlier, later], TOLERANCE)

    # "next" starts at 27.82 s, inside the earlier core, so the earlier spelling is kept.
    assert vocabulary.text(merged) == "bath, next man."


# =============================================================================
# General behavior
# =============================================================================


def test_single_chunk_keeps_every_word_including_the_last() -> None:
    vocabulary = Vocabulary()
    only = chunk(vocabulary, [(f"{MARK}hello", 0.1), (f"{MARK}world", 0.9)], source=(0.0, 1.0), core=(0.0, 1.0))

    assert vocabulary.text(merge_chunks([only], TOLERANCE)) == "hello world"


def test_repetition_inside_a_chunk_is_preserved() -> None:
    vocabulary = Vocabulary()
    earlier = chunk(
        vocabulary,
        [(f"{MARK}no", 5.0), (f"{MARK}no", 5.3), (f"{MARK}no", 5.6), (f"{MARK}stop", 13.8)],
        source=(0.0, 14.5),
        core=(0.0, 14.0),
    )
    later = chunk(vocabulary, [(f"{MARK}stop", 13.9), (f"{MARK}now", 14.3)], source=(13.5, 20.0), core=(14.0, 20.0))

    assert vocabulary.text(merge_chunks([earlier, later], TOLERANCE)) == "no no no stop now"


def test_disagreeing_chunks_fall_back_to_whole_word_ownership() -> None:
    """No agreed word in the overlap: words go to the core holding their start, unsplit."""

    vocabulary = Vocabulary()
    earlier = chunk(
        vocabulary,
        [(f"{MARK}alpha", 13.0), (f"{MARK}bra", 13.9), ("vo", 14.05)],
        source=(0.0, 14.5),
        core=(0.0, 14.0),
    )
    later = chunk(
        vocabulary,
        [(f"{MARK}bravado", 13.95), (f"{MARK}charlie", 14.3)],
        source=(13.5, 20.0),
        core=(14.0, 20.0),
    )

    merged = merge_chunks([earlier, later], TOLERANCE)

    # "bravo" (13.9) belongs to the earlier core, "bravado" (13.95) is outside
    # the later core, "charlie" belongs to the later core.
    assert vocabulary.text(merged) == "alpha bravo charlie"


@pytest.mark.parametrize("tolerance", [0, TOLERANCE])
def test_zero_tolerance_disables_alignment(tolerance: int) -> None:
    vocabulary = Vocabulary()
    earlier = chunk(vocabulary, [(f"{MARK}one", 13.9)], source=(0.0, 14.5), core=(0.0, 14.0))
    later = chunk(vocabulary, [(f"{MARK}one", 14.2)], source=(13.5, 20.0), core=(14.0, 20.0))

    text = vocabulary.text(merge_chunks([earlier, later], tolerance))

    # Without alignment both copies are in their own cores; with it they merge.
    assert text == ("one one" if tolerance == 0 else "one")


def test_agreement_requires_close_timestamps() -> None:
    """The same word far apart in time is two words, not one duplicate."""

    vocabulary = Vocabulary()
    earlier = chunk(vocabulary, [(f"{MARK}yes", 12.4), (f"{MARK}and", 13.9)], source=(0.0, 14.5), core=(0.0, 14.0))
    later = chunk(vocabulary, [(f"{MARK}and", 13.95), (f"{MARK}yes", 15.3)], source=(13.5, 20.0), core=(14.0, 20.0))

    assert vocabulary.text(merge_chunks([earlier, later], TOLERANCE)) == "yes and yes"


def test_three_chunks_chain_and_silent_chunks_are_harmless() -> None:
    vocabulary = Vocabulary()
    first = chunk(vocabulary, [(f"{MARK}a", 1.0), (f"{MARK}b", 9.9)], source=(0.0, 10.5), core=(0.0, 10.0))
    silent = chunk(vocabulary, [], source=(9.5, 20.5), core=(10.0, 20.0))
    last = chunk(vocabulary, [(f"{MARK}c", 25.0)], source=(19.5, 30.0), core=(20.0, 30.0))

    assert vocabulary.text(merge_chunks([first, silent, last], TOLERANCE)) == "a b c"
    assert merge_chunks([], TOLERANCE) == []
