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

from ..configuration.config import ParakeetConfig
from .attention import Attention, EncoderMasks


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
#
# The two pointwise (kernel 1) convolutions are matrix products over the
# channels, so they run as linear layers on the (B, T, H) layout: the weights
# stay Conv1d parameters (checkpoint layout unchanged), but cuBLAS runs them
# instead of cuDNN, which honours float16 accumulation, and only the depthwise
# convolution needs the channel-first layout.
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
        self.norm: nn.Module = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=convolution_bias,
        )

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        """
        Fold the inference BatchNorm into the depthwise convolution.

        In eval mode BatchNorm is a fixed per-channel affine map,
        ``y = (x - mean) * gamma / sqrt(var + eps) + beta``, so it can be
        merged into the convolution that precedes it: the weights are scaled
        by ``gamma / sqrt(var + eps)`` and the bias absorbs the rest. One
        full-size kernel per block disappears, and the affine map runs in the
        convolution's precision. The fold itself is computed in float64.
        Afterwards ``norm`` is the identity and the state dict no longer
        matches the checkpoint, so this is for inference only.
        """

        if isinstance(self.norm, nn.Identity):
            return
        norm = self.norm
        convolution = self.depthwise_conv
        scale = norm.weight.double() / torch.sqrt(norm.running_var.double() + norm.eps)

        weight = convolution.weight.double() * scale[:, None, None]
        if convolution.bias is not None:
            bias = convolution.bias.double()
        else:
            bias = torch.zeros_like(scale)
        bias = (bias - norm.running_mean.double()) * scale + norm.bias.double()

        dtype = convolution.weight.dtype
        convolution.weight = nn.Parameter(weight.to(dtype), requires_grad=False)
        convolution.bias = nn.Parameter(bias.to(dtype), requires_grad=False)
        self.norm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        masks: EncoderMasks | None,
    ) -> torch.Tensor:
        """
        Apply local convolution to an encoder tensor ``(B, T, H)``.

        Args:
            hidden_states: Normalized block state ``(B, T, H)``.
            masks: Padding masks of the batch, or None without padding.

        Returns:
            Convolution output with shape ``(B, T, H)``.
        """

        expansion_weight = self.pointwise_conv1.weight.squeeze(-1)  # (2H, H)
        hidden_states = torch.nn.functional.linear(
            hidden_states,
            expansion_weight,
            self.pointwise_conv1.bias,
        )  # (B, T, 2H)
        hidden_states = torch.nn.functional.glu(hidden_states, dim=-1)  # (B, T, H)

        if masks is not None:
            # Padding must not leak into real frames through the kernel.
            hidden_states = hidden_states.masked_fill(masks.padded_frames, 0.0)

        # With a float16 encoder the depthwise convolution keeps float32
        # weights (see ParakeetTDT.set_encoder_dtype): cuDNN 9's float16
        # depthwise kernel returned wrong values for batches of 7 or more rows
        # in a long-running process, which emptied whole chunks. Each layout
        # change is a single copy that also casts; in float32 mode only the
        # transpose remains.
        convolution_dtype = self.depthwise_conv.weight.dtype
        block_dtype = hidden_states.dtype
        hidden_states = hidden_states.transpose(1, 2).to(
            convolution_dtype,
            memory_format=torch.contiguous_format,
        )  # (B, H, T)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).to(
            block_dtype,
            memory_format=torch.contiguous_format,
        )  # (B, T, H)

        hidden_states = torch.nn.functional.silu(hidden_states)
        projection_weight = self.pointwise_conv2.weight.squeeze(-1)  # (H, H)
        return torch.nn.functional.linear(
            hidden_states,
            projection_weight,
            self.pointwise_conv2.bias,
        )


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
        masks: EncoderMasks | None,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Run all block stages while preserving shape ``(B, T, H)``."""

        # torch.add(..., alpha=0.5) scales and adds in one kernel; 0.5 is a
        # power of two, so the result matches ``residual + 0.5 * output``.
        feed_forward_output = self.feed_forward1(
            self.norm_feed_forward1(hidden_states)
        )
        hidden_states = torch.add(hidden_states, feed_forward_output, alpha=0.5)

        attention_output = self.self_attn(
            self.norm_self_att(hidden_states),
            position_embeddings,
            masks,
        )
        hidden_states = hidden_states + attention_output

        convolution_output = self.conv(
            self.norm_conv(hidden_states),
            masks,
        )
        hidden_states = hidden_states + convolution_output

        feed_forward_output = self.feed_forward2(
            self.norm_feed_forward2(hidden_states)
        )
        hidden_states = torch.add(hidden_states, feed_forward_output, alpha=0.5)

        return self.norm_out(hidden_states)
