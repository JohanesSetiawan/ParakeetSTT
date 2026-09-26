"""
Tokenizer artifact handling for the standalone Parakeet runtime.

Parakeet stores a JSON BPE tokenizer with a Metaspace pre-tokenizer and decoder.
The model's inference path only needs decoding generated IDs, but the module
also preserves the BPE merge table so future training-data preparation can add
an explicitly tested encoder without coupling the model to Tokenizers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


# =============================================================================
# JSON tokenizer loading
# =============================================================================
# The tokenizer JSON contains two ID spaces: the base BPE vocabulary and added
# tokens such as <blank>. Effective vocabulary means the union of both spaces;
# comparing only the base BPE count would incorrectly report a mismatch for this
# checkpoint (8192 base entries plus the ID 8192 blank token).
# =============================================================================


class BpeTokenizer:
    """Decode Parakeet token IDs using tokenizer.json without third-party code."""

    def __init__(
        self,
        tokenizer_path: Path,
        pad_token_id: int,
        blank_token_id: int,
    ) -> None:
        """Load vocabulary, added tokens, and special-token IDs from JSON."""

        tokenizer_json = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        model = tokenizer_json.get("model")
        if not isinstance(model, dict) or model.get("type") != "BPE":
            raise ValueError(f"Tokenizer model must be a BPE object: {tokenizer_path}")

        vocabulary = model.get("vocab")
        if not isinstance(vocabulary, dict):
            raise ValueError(f"Tokenizer vocabulary is missing: {tokenizer_path}")

        self.id_to_token = {
            int(token_id): token
            for token, token_id in vocabulary.items()
            if isinstance(token_id, int) and isinstance(token, str)
        }
        self.merge_rules = [tuple(rule) for rule in model.get("merges", [])]
        self.special_ids = {pad_token_id, blank_token_id}
        self.pad_token_id = pad_token_id
        self.blank_token_id = blank_token_id

        added_tokens = tokenizer_json.get("added_tokens", [])
        if not isinstance(added_tokens, list):
            raise ValueError(f"Tokenizer added_tokens must be a list: {tokenizer_path}")

        for token in added_tokens:
            if not isinstance(token, dict):
                continue
            token_id = token.get("id")
            content = token.get("content")
            if isinstance(token_id, int) and isinstance(content, str):
                self.id_to_token[token_id] = content
                if token.get("special") is True:
                    self.special_ids.add(token_id)

        self.base_vocab_size = len(vocabulary)
        self.effective_vocab_size = len(self.id_to_token)
        self.max_token_id = max(self.id_to_token, default=-1)

    def decode(self, token_ids: Iterable[int]) -> str:
        """
        Decode generated IDs using the JSON Metaspace replacement marker.

        Special tokens, padding, blanks, and unknown IDs are omitted. Generated
        Parakeet IDs are already BPE pieces, so inference does not need to rerun
        merge operations during decoding; concatenation followed by Metaspace
        replacement reconstructs the tokenizer's text surface.
        """

        pieces: list[str] = []
        for token_id in token_ids:
            normalized_id = int(token_id)
            if normalized_id in self.special_ids:
                continue
            token = self.id_to_token.get(normalized_id)
            if token is not None:
                pieces.append(token)

        return "".join(pieces).replace("\u2581", " ").strip()

    def batch_decode(self, token_sequences: torch.Tensor) -> list[str]:
        """Decode a rank-two batch of generated token IDs."""

        if token_sequences.ndim != 2:
            raise ValueError(
                f"Expected token sequences with shape (B, U), got {tuple(token_sequences.shape)}"
            )
        return [self.decode(sequence.tolist()) for sequence in token_sequences]
