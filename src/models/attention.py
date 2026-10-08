"""
Relative-position attention for the Parakeet Fast Conformer encoder.

This module owns positional encoding, key/value head expansion, relative score
alignment, padding-safe additive masks, and the PyTorch SDPA call. Feed-forward,
convolution, subsampling, and encoder stack orchestration live in separate
modules.

Tensor notation
---------------
B: batch size
T: encoded time steps
H: encoder hidden size
A: attention head count
KV: key/value head count
D: per-head dimension, H / A
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..configuration.config import ParakeetConfig


# =============================================================================
# Padding masks
# =============================================================================
# Every block needs the same two facts about padding: which keys attention
# must ignore, and which frames are padding. Both are built once per batch;
# deriving them from a (B, 1, T, T) boolean pair mask in each of the 24 blocks
# cost three extra full-size kernels per block.
# =============================================================================


@dataclass(frozen=True)
class EncoderMasks:
    """
    Padding facts of one encoder batch, shared by every block.

    Attributes:
        key_bias: Additive attention bias ``(B, 1, 1, T)`` in the encoder
            dtype: 0 for real keys, a large finite negative value for padded
            keys. Half of the dtype's lowest value, so adding a score can
            never overflow to -inf; a row whose keys are all padding then
            stays finite instead of turning into NaN in the softmax.
        padded_frames: ``(B, T, 1)``, True for padding frames.
    """

    key_bias: torch.Tensor
    padded_frames: torch.Tensor

    @classmethod
    def from_valid_frames(cls, valid_frames: torch.Tensor, dtype: torch.dtype) -> "EncoderMasks":
        """
        Build the masks from a valid-frame mask.

        Args:
            valid_frames: ``(B, T)``, True for real frames.
            dtype: Floating-point type of the attention scores.
        """

        padded_frames = ~valid_frames
        blocked_value = torch.finfo(dtype).min / 2
        key_bias = torch.zeros(valid_frames.shape, dtype=dtype, device=valid_frames.device)
        key_bias = key_bias.masked_fill(padded_frames, blocked_value)
        return cls(
            key_bias=key_bias[:, None, None, :],
            padded_frames=padded_frames[:, :, None],
        )


# =============================================================================
# Relative positional encoding
# =============================================================================
# Parakeet uses Transformer-XL-style relative positions. For T encoded frames,
# positions span +(T-1) to -(T-1), producing 2T-1 vectors. Sine and cosine are
# interleaved into H dimensions and projected per attention layer.
# =============================================================================


class RelativePositionalEncoding(nn.Module):
    """Generate sinusoidal relative positions with shape ``(1, 2T-1, H)``."""

    def __init__(self, configuration: ParakeetConfig) -> None:
        """Precompute inverse frequencies from the configured hidden size."""

        super().__init__()
        encoder = configuration.encoder
        hidden_size = encoder["hidden_size"]
        self.max_position_embeddings = encoder["max_position_embeddings"]

        # Explicit CPU placement keeps this non-checkpoint buffer real when the
        # model skeleton is built on the meta device by load_model().
        even_dimensions = torch.arange(0, hidden_size, 2, dtype=torch.float32, device="cpu")
        inverse_frequency = 1.0 / (10000.0 ** (even_dimensions / hidden_size))
        self.register_buffer("inv_freq", inverse_frequency, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Construct relative position vectors for the current encoded length.

        Args:
            hidden_states: Encoder tensor ``(B, T, H)``. Its shape, device, and
                dtype define the returned positional tensor.

        Returns:
            Relative embeddings with shape ``(1, 2T-1, H)``. They are the same
            for every row of a batch, so one copy is returned and broadcast;
            each attention layer then projects them once per batch instead of
            once per row.
        """

        sequence_length = hidden_states.shape[1]
        position_ids = torch.arange(
            sequence_length - 1,
            -sequence_length,
            -1,
            device=hidden_states.device,
        )
        inverse_frequency = self.inv_freq[None, :, None].float()
        position_ids = position_ids[None, None, :].float()

        frequencies = (inverse_frequency @ position_ids).transpose(1, 2)
        sine = frequencies.sin()
        cosine = frequencies.cos()
        position_embeddings = torch.stack([sine, cosine], dim=-1)
        position_embeddings = position_embeddings.reshape(
            *position_embeddings.shape[:-2],
            -1,
        )

        return position_embeddings.to(dtype=hidden_states.dtype)


# =============================================================================
# Head expansion
# =============================================================================


def repeat_key_value(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    """
    Expand key/value heads from ``(B, KV, T, D)`` to ``(B, A, T, D)``.

    Expansion uses a view-compatible broadcast before reshaping, avoiding a
    physical repeat when grouped-query attention uses shared key/value heads.
    """

    if repetitions == 1:
        return hidden_states

    batch_size, key_value_heads, sequence_length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        key_value_heads,
        repetitions,
        sequence_length,
        head_dim,
    )
    return hidden_states.reshape(
        batch_size,
        key_value_heads * repetitions,
        sequence_length,
        head_dim,
    )


# =============================================================================
# Relative-position multi-head attention
# =============================================================================
# The attention score combines two terms:
# 1. content score from (query + bias_u) dot key;
# 2. relative score from (query + bias_v) dot projected relative position.
#
# The relative score is supplied to SDPA as an additive position-bias mask,
# with the padding key bias added in the same kernel. Padded query rows are
# zeroed after SDPA.
# =============================================================================


class Attention(nn.Module):
    """Relative-position multi-head self-attention for one encoder block."""

    def __init__(self, configuration: ParakeetConfig, layer_index: int) -> None:
        """Create checkpoint-compatible projections and learned global biases."""

        super().__init__()
        encoder = configuration.encoder
        hidden_size = encoder["hidden_size"]
        self.layer_index = layer_index
        self.num_heads = encoder["num_attention_heads"]
        self.num_key_value_heads = encoder["num_key_value_heads"]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        attention_bias = encoder["attention_bias"]

        self.q_proj = nn.Linear(
            hidden_size,
            self.num_heads * self.head_dim,
            bias=attention_bias,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
        )
        self.relative_k_proj = nn.Linear(
            hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

    @staticmethod
    def _relative_shift(attention_scores: torch.Tensor) -> torch.Tensor:
        """Align ``(B, A, T, 2T-1)`` relative scores to query-key pairs."""

        batch_size, num_heads, query_length, position_length = attention_scores.shape
        attention_scores = nn.functional.pad(attention_scores, pad=(1, 0))
        attention_scores = attention_scores.view(
            batch_size,
            num_heads,
            -1,
            query_length,
        )
        return attention_scores[:, :, 1:].view(
            batch_size,
            num_heads,
            query_length,
            position_length,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        masks: EncoderMasks | None,
    ) -> torch.Tensor:
        """
        Apply relative attention to ``(B, T, H)`` hidden states.

        Args:
            hidden_states: Encoder hidden states ``(B, T, H)``.
            position_embeddings: Relative position tensor ``(1, 2T-1, H)``,
                shared by every row.
            masks: Padding masks of the batch, or None without padding.

        Returns:
            Projected attention output ``(B, T, H)``.
        """

        batch_size, sequence_length, _ = hidden_states.shape
        projected_shape = (batch_size, sequence_length, -1, self.head_dim)

        # Project and expose head dimension: (B, T, H) -> (B, A, T, D).
        query = self.q_proj(hidden_states).view(projected_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(projected_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(projected_shape).transpose(1, 2)

        query_with_content_bias = query + self.bias_u.view(
            1,
            self.num_heads,
            1,
            self.head_dim,
        )
        query_with_position_bias = query + self.bias_v.view(
            1,
            self.num_heads,
            1,
            self.head_dim,
        )

        # One projection for the whole batch: (1, 2T-1, H) -> (1, 2T-1, A, D).
        # The score scaling is applied here, on 2T-1 rows, instead of on the
        # (B, A, T, T) scores. With D = 128 the scale 1/sqrt(128) is not a
        # power of two, so this moves one rounding step: the scores can differ
        # in the last bit, which the labeled long-form WER did not register.
        relative_key = self.relative_k_proj(position_embeddings) * self.scaling
        relative_key = relative_key.view(
            position_embeddings.shape[0],
            -1,
            self.num_heads,
            self.head_dim,
        )
        # (B, A, T, D) @ (1, A, D, 2T-1) broadcasts to (B, A, T, 2T-1).
        relative_scores = query_with_position_bias @ relative_key.permute(0, 2, 3, 1)
        relative_scores = self._relative_shift(relative_scores)
        relative_scores = relative_scores[..., :sequence_length]  # (B, A, T, T) view

        key = repeat_key_value(key, self.num_key_value_groups)
        value = repeat_key_value(value, self.num_key_value_groups)

        # The shifted scores are a strided view; SDPA needs a dense mask, and
        # the padding bias is added in the same pass that materializes it.
        if masks is not None:
            additive_mask = relative_scores + masks.key_bias
        else:
            additive_mask = relative_scores.contiguous()

        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query_with_content_bias,
            key,
            value,
            attn_mask=additive_mask,
            dropout_p=0.0,
            scale=self.scaling,
            is_causal=False,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, sequence_length, -1)

        if masks is not None:
            attention_output = attention_output.masked_fill(masks.padded_frames, 0.0)

        return self.o_proj(attention_output)
