"""
Fast Conformer encoder-stack orchestration.

Attention math, block internals, and convolutional subsampling are implemented
in focused sibling modules. This module owns only stack composition, positional
encoding application, dropout, valid-length conversion, pair-mask construction,
and sequential block execution.
"""

from __future__ import annotations

import torch
from torch import nn

from ..configuration.config import ParakeetConfig
from .attention import RelativePositionalEncoding
from .conformer import EncoderBlock
from .subsampling import Subsampling


# =============================================================================
# Acoustic encoder stack
# =============================================================================
# Input mask shape:  (B, T_input)
# Output mask shape: (B, T_encoded)
# Pair mask shape:   (B, 1, T_encoded, T_encoded)
#
# The pair mask blocks invalid keys and invalid queries. Attention handles fully
# invalid query rows explicitly, preventing NaN in mixed-duration batches.
# =============================================================================


class Encoder(nn.Module):
    """Compose subsampling, relative positions, and Fast Conformer blocks."""

    def __init__(self, configuration: ParakeetConfig) -> None:
        """Build the encoder stack from the validated JSON configuration."""

        super().__init__()
        encoder = configuration.encoder
        self.config = encoder
        self.dropout = encoder["dropout"]
        self.dropout_positions = encoder["dropout_positions"]
        self.subsampling = Subsampling(configuration)
        self.encode_positions = RelativePositionalEncoding(configuration)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(configuration, layer_index)
                for layer_index in range(encoder["num_hidden_layers"])
            ]
        )

    def output_length(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Return valid encoder lengths after configured subsampling."""

        return self.subsampling.output_length(input_lengths)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Encode padded log-mel features.

        Args:
            input_features: Normalized features with shape ``(B, T_input, M)``.
            attention_mask: Optional valid-frame mask ``(B, T_input)``.

        Returns:
            Hidden states ``(B, T_encoded, H)`` and optional valid-frame mask
            ``(B, T_encoded)``.
        """

        hidden_states = self.subsampling(input_features, attention_mask)
        position_embeddings = self.encode_positions(hidden_states)
        hidden_states = torch.nn.functional.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        position_embeddings = torch.nn.functional.dropout(
            position_embeddings,
            p=self.dropout_positions,
            training=self.training,
        )

        output_mask = None
        pair_attention_mask = None
        if attention_mask is not None:
            output_lengths = self.output_length(attention_mask.sum(dim=-1))
            output_mask = (
                torch.arange(hidden_states.shape[1], device=hidden_states.device)[None, :]
                < output_lengths[:, None]
            )

            pair_attention_mask = output_mask.unsqueeze(1).expand(
                -1,
                hidden_states.shape[1],
                -1,
            )
            pair_attention_mask = pair_attention_mask & pair_attention_mask.transpose(1, 2)
            pair_attention_mask = pair_attention_mask.unsqueeze(1)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                pair_attention_mask,
                position_embeddings,
            )

        return hidden_states, output_mask
