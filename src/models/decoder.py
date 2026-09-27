"""
LSTM prediction network and cache for Parakeet TDT decoding.

The transducer decoder predicts from previously emitted token IDs. It is
independent from acoustic feature extraction and the Fast Conformer encoder.
This module owns the blank-aware cache semantics that let batched decoding skip
unnecessary LSTM updates when one or more samples emit the blank token.

Tensor notation
---------------
B: batch size
L: decoder LSTM layer count
U: decoder token steps (one during greedy inference)
H: decoder hidden size
"""

from __future__ import annotations

import torch
from torch import nn

from ..configuration.config import ParakeetConfig


# =============================================================================
# Decoder cache
# =============================================================================
# TDT decoding calls the prediction network once per emitted token-duration
# decision. Recomputing the complete token history would be quadratic in output
# length, so the LSTM hidden/cell states and last projected output are retained.
# Blank predictions do not change decoder history and therefore preserve the
# corresponding cache row.
# =============================================================================


class DecoderCache:
    """Mutable per-batch cache for LSTM hidden, cell, and projected states."""

    def __init__(self, configuration: ParakeetConfig) -> None:
        """Store dimensions; allocate tensors lazily on the runtime device."""

        self.configuration = configuration
        self.cache: torch.Tensor | None = None
        self.hidden_state: torch.Tensor | None = None
        self.cell_state: torch.Tensor | None = None
        self.is_initialized = False

    def initialize(self, reference: torch.Tensor) -> None:
        """
        Allocate zero state using the reference tensor's device and dtype.

        Args:
            reference: Decoder embedding or output with shape ``(B, U, H)``.
                Its batch size, device, and dtype define cache allocation.
        """

        batch_size = reference.shape[0]
        hidden_size = self.configuration.model["decoder_hidden_size"]
        layer_count = self.configuration.model["num_decoder_layers"]

        self.cache = torch.zeros(
            batch_size,
            1,
            hidden_size,
            device=reference.device,
            dtype=reference.dtype,
        )
        self.hidden_state = torch.zeros(
            layer_count,
            batch_size,
            hidden_size,
            device=reference.device,
            dtype=reference.dtype,
        )
        self.cell_state = torch.zeros_like(self.hidden_state)
        self.is_initialized = True

    def update(
        self,
        decoder_output: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor,
        update_mask: torch.Tensor | None,
    ) -> None:
        """
        Commit new decoder state for all rows or selected non-blank rows.

        Args:
            decoder_output: Projected LSTM output ``(B, 1, H)``.
            hidden_state: LSTM hidden state ``(L, B, H)``.
            cell_state: LSTM cell state ``(L, B, H)``.
            update_mask: Boolean ``(B,)`` mask. True commits the new state;
                false retains the preceding blank-token state. ``None`` commits
                every row during initial cache population.
        """

        if not self.is_initialized:
            self.initialize(decoder_output)

        # These attributes are guaranteed by initialize(), but keeping the
        # explicit runtime check produces an actionable failure if cache state
        # is ever mutated incorrectly by future code.
        if self.cache is None or self.hidden_state is None or self.cell_state is None:
            raise RuntimeError("Decoder cache initialization did not create all state tensors")

        if update_mask is None:
            self.cache.copy_(decoder_output)
            self.hidden_state.copy_(hidden_state)
            self.cell_state.copy_(cell_state)
            return

        update_mask = update_mask.to(device=decoder_output.device)
        self.cache = torch.where(
            update_mask[:, None, None],
            decoder_output,
            self.cache,
        )
        self.hidden_state = torch.where(
            update_mask[None, :, None],
            hidden_state,
            self.hidden_state,
        )
        self.cell_state = torch.where(
            update_mask[None, :, None],
            cell_state,
            self.cell_state,
        )


# =============================================================================
# LSTM prediction network
# =============================================================================
# Attribute names embedding, lstm, and decoder_projector are fixed by the
# checkpoint. Class names and file boundaries may change, but these attributes
# must remain stable for strict state-dict loading.
# =============================================================================


class Decoder(nn.Module):
    """Blank-aware LSTM prediction network for RNN-T/TDT token history."""

    def __init__(self, configuration: ParakeetConfig) -> None:
        """Construct embedding, stacked LSTM, and output projection from JSON."""

        super().__init__()
        model = configuration.model
        hidden_size = model["decoder_hidden_size"]
        self.blank_token_id = model["blank_token_id"]

        self.embedding = nn.Embedding(model["vocab_size"], hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=model["num_decoder_layers"],
            batch_first=True,
        )
        self.decoder_projector = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor,
        cache: DecoderCache | None,
    ) -> torch.Tensor:
        """
        Predict the decoder representation for the latest token IDs.

        Args:
            input_ids: Token IDs with shape ``(B, 1)`` during greedy decoding.
            cache: Optional mutable LSTM cache. Rows whose input is blank keep
                their cached state and output; other rows advance.

        Returns:
            Projected decoder states with shape ``(B, 1, H)``.
        """

        # There is deliberately no "every row is blank, skip the LSTM" fast
        # path: deciding it needs bool() on a GPU tensor, which stalls the host
        # once per decoding step. Measured on real speech the fast path fired
        # on about 8 percent of steps, so the sync cost far more than the
        # LSTM calls it saved. The masked cache update below leaves blank rows
        # unchanged, so the output is identical either way.
        blank_mask: torch.Tensor | None = None
        if cache is not None:
            blank_mask = input_ids[:, -1] == self.blank_token_id

        embeddings = self.embedding(input_ids)
        hidden_cell_states = None
        was_initialized = False

        if cache is not None:
            was_initialized = cache.is_initialized
            if not was_initialized:
                cache.initialize(embeddings)
            if cache.hidden_state is None or cache.cell_state is None:
                raise RuntimeError("Decoder cache is missing LSTM state")
            hidden_cell_states = (cache.hidden_state, cache.cell_state)

        lstm_output, (hidden_state, cell_state) = self.lstm(
            embeddings,
            hidden_cell_states,
        )
        decoder_output = self.decoder_projector(lstm_output)

        if cache is not None:
            if blank_mask is None:
                raise RuntimeError("Blank mask was not calculated for cached decoding")
            update_mask = ~blank_mask if was_initialized else None
            cache.update(
                decoder_output,
                hidden_state,
                cell_state,
                update_mask,
            )
            if cache.cache is None:
                raise RuntimeError("Decoder cache update did not produce an output")
            return cache.cache

        return decoder_output