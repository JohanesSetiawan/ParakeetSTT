"""
Fast Conformer block components for the Parakeet acoustic encoder.

This module owns the Macaron feed-forward branches, depthwise convolution
branch, layer normalization, and residual ordering. Relative attention and
subsampling live in separate modules so each numerical subsystem can be tested
without constructing the complete encoder stack.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ParakeetConfig
from .attention import Attention


# =============================================================================
# Feed-forward branch
# =============================================================================
# Fast Conformer uses two feed-forward branches around attention/convolution.
# Each residual contribution is multiplied by 0.5, the Macaron scaling that
# prevents two full feed-forward updates from doubling residual magnitude.
# =============================================================================


class FeedForward(nn.Module):
    """One SiLU feed-forward branch of a Fast Conformer block."""

    def __init__(self, configuration: ParakeetConfig) -> None:
        super().__init__()
        encoder = configuration.encoder
        hidden_size = encoder["hidden_size"]
        intermediate_size = encoder["intermediate_size"]
        attention_bias = encoder["attention_bias"]

        self.linear1 = nn.Linear(hidden_size, intermediate_size, bias=attention_bias)
        self.linear2 = nn.Linear(intermediate_size, hidden_size, bias=attention_bias)
        self.activation_dropout = encoder["activation_dropout"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project ``H -> intermediate -> H`` with SiLU and training dropout."""

        hidden_states = torch.nn.functional.silu(self.linear1(hidden_states))
        hidden_states = torch.nn.functional.dropout(
            hidden_states,
            p=self.activation_dropout,
            training=self.training,
        )
        return self.linear2(hidden_states)


# =============================================================================
# Convolution branch
# =============================================================================
# The pointwise projection doubles channels so GLU can split values and gates.
# A same-padded depthwise convolution then models local time context per channel,
# followed by BatchNorm, SiLU, and a pointwise projection back to H channels.
# =============================================================================


class ConvolutionModule(nn.Module):
    """GLU-gated depthwise convolution branch with checkpoint BatchNorm state."""

    def __init__(self, configuration: ParakeetConfig) -> None:
        super().__init__()
        encoder = configuration.encoder
        channels = encoder["hidden_size"]
        kernel_size = encoder["conv_kernel_size"]
        convolution_bias = encoder["convolution_bias"]
        padding = (kernel_size - 1) // 2

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            bias=convolution_bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=convolution_bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=convolution_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Apply local convolution to an encoder tensor ``(B, T, H)``.

        Args:
            hidden_states: Normalized block state ``(B, T, H)``.
            attention_mask: Optional pair mask ``(B, 1, T, T)``.

        Returns:
            Convolution output with shape ``(B, T, H)``.
        """

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = torch.nn.functional.glu(hidden_states, dim=1)

        if attention_mask is not None:
            # Reduce the pair mask over keys to identify padded query rows. The
            # resulting (B, 1, T) mask matches the Conv1d channel-first layout.
            all_masked_queries = torch.all(~attention_mask, dim=2)
            hidden_states = hidden_states.masked_fill(all_masked_queries, 0.0)

        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = torch.nn.functional.silu(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return hidden_states.transpose(1, 2)


# =============================================================================
# Complete Fast Conformer block
# =============================================================================
# Operation order:
# 1. pre-norm feed-forward with 0.5 residual scale;
# 2. pre-norm relative self-attention;
# 3. pre-norm depthwise convolution;
# 4. pre-norm feed-forward with 0.5 residual scale;
# 5. final layer normalization.
# =============================================================================


class EncoderBlock(nn.Module):
    """One checkpoint-compatible Fast Conformer encoder block."""

    def __init__(self, configuration: ParakeetConfig, layer_index: int) -> None:
        super().__init__()
        hidden_size = configuration.encoder["hidden_size"]

        self.feed_forward1 = FeedForward(configuration)
        self.self_attn = Attention(configuration, layer_index)
        self.conv = ConvolutionModule(configuration)
        self.feed_forward2 = FeedForward(configuration)
        self.norm_feed_forward1 = nn.LayerNorm(hidden_size)
        self.norm_self_att = nn.LayerNorm(hidden_size)
        self.norm_conv = nn.LayerNorm(hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(hidden_size)
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Run all block stages while preserving shape ``(B, T, H)``."""

        residual = hidden_states
        feed_forward_output = self.feed_forward1(
            self.norm_feed_forward1(hidden_states)
        )
        hidden_states = residual + 0.5 * feed_forward_output

        attention_output = self.self_attn(
            self.norm_self_att(hidden_states),
            position_embeddings,
            attention_mask,
        )
        hidden_states = hidden_states + attention_output

        convolution_output = self.conv(
            self.norm_conv(hidden_states),
            attention_mask,
        )
        hidden_states = hidden_states + convolution_output

        feed_forward_output = self.feed_forward2(
            self.norm_feed_forward2(hidden_states)
        )
        hidden_states = hidden_states + 0.5 * feed_forward_output

        return self.norm_out(hidden_states)
