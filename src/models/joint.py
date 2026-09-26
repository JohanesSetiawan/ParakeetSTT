"""
Token-and-duration joint network for the standalone Parakeet runtime.

The joint network is the boundary where acoustic encoder state and token-history
decoder state meet. It owns no recurrent cache, feature processing, or greedy
control flow. Its only responsibility is producing token and duration logits.
"""

from __future__ import annotations

import torch
from torch import nn

from ..configuration.config import ParakeetConfig


# =============================================================================
# TDT joint projection
# =============================================================================
# Encoder and decoder are projected to the same hidden dimension before they
# reach this module. Their sum passes through the JSON-configured ReLU activation
# and one linear head. The first vocab_size rows are token logits; the remaining
# rows correspond, in order, to configuration.durations.
# =============================================================================


class JointNetwork(nn.Module):
    """Combine encoder and decoder states into token and duration logits."""

    def __init__(self, configuration: ParakeetConfig) -> None:
        """Create the checkpoint-compatible joint output head."""

        super().__init__()
        hidden_size = configuration.model["decoder_hidden_size"]
        output_size = configuration.vocab_size + len(configuration.durations)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce TDT logits for one or more aligned encoder/decoder positions.

        Args:
            decoder_hidden_states: Token-history states ending in dimension H.
            encoder_hidden_states: Acoustic states broadcast-compatible with the
                decoder tensor and ending in the same H dimension.

        Returns:
            Logits ending in ``vocab_size + duration_class_count``.
        """

        joint_hidden_states = torch.relu(
            encoder_hidden_states + decoder_hidden_states
        )
        return self.head(joint_hidden_states)