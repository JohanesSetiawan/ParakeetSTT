"""
Convolutional time-frequency subsampling for Parakeet log-mel features.

The subsampler reduces acoustic sequence length and mel-frequency resolution by
the JSON-configured factor before the Fast Conformer stack. It owns convolution
geometry and valid-length propagation, but no attention or encoder layers.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..configuration.config import ParakeetConfig


# =============================================================================
# Depthwise-separable Conv2d subsampling
# =============================================================================
# For subsampling_factor=8, log2(8)=3 stride-two stages reduce time and mel
# frequency to one eighth. The first convolution creates channels; later stages
# use depthwise 3x3 plus pointwise 1x1 convolutions. ModuleList indices mirror
# checkpoint keys such as encoder.subsampling.layers.2.weight.
# =============================================================================


class Subsampling(nn.Module):
    """Reduce ``(B, T, M)`` log-mel features into ``(B, T/8, H)`` states."""

    def __init__(self, configuration: ParakeetConfig) -> None:
        """Build the configured Conv2d stack and final hidden projection."""

        super().__init__()
        encoder = configuration.encoder
        channels = encoder["subsampling_conv_channels"]
        kernel_size = encoder["subsampling_conv_kernel_size"]
        stride = encoder["subsampling_conv_stride"]
        self.padding = (kernel_size - 1) // 2
        layer_count = int(math.log2(encoder["subsampling_factor"]))

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv2d(1, channels, kernel_size, stride, self.padding)
        )
        self.layers.append(nn.ReLU())

        for _ in range(layer_count - 1):
            self.layers.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size,
                    stride,
                    self.padding,
                    groups=channels,
                )
            )
            self.layers.append(nn.Conv2d(channels, channels, kernel_size=1))
            self.layers.append(nn.ReLU())

        output_frequency = encoder["num_mel_bins"] // (stride**layer_count)
        self.linear = nn.Linear(
            channels * output_frequency,
            encoder["hidden_size"],
        )

    def leaves_padding(self, shortest_length: int, padded_length: int) -> bool:
        """
        Whether the time masks change anything for this batch.

        The feature extractor always pads one frame (``samples // hop + 1``
        frames, ``samples // hop`` valid), but the first strided convolution
        maps both lengths to the same count. Masks matter only if, after some
        convolution, the shortest row is still shorter than the padded one.

        Args:
            shortest_length: Valid input frames of the shortest row.
            padded_length: Input frames of the padded batch.
        """

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                shortest_length = self._convolved_length(layer, shortest_length)
                padded_length = self._convolved_length(layer, padded_length)
                if shortest_length < padded_length:
                    return True
        return False

    @staticmethod
    def _convolved_length(layer: nn.Conv2d, length: int) -> int:
        """Time length after one Conv2d, by the standard convolution formula."""

        return (length + layer.padding[0] + layer.padding[1] - layer.kernel_size[0]) // layer.stride[0] + 1

    def output_length(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Propagate valid frame lengths through every strided convolution.

        The standard discrete convolution length formula is applied only to
        Conv2d layers whose stride differs from one. ReLU and pointwise stride-one
        convolutions preserve sequence length.
        """

        lengths = input_lengths
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
                lengths = (
                    lengths
                    + layer.padding[0]
                    + layer.padding[1]
                    - layer.kernel_size[0]
                ) // layer.stride[0] + 1
        return lengths

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Subsample padded features while clearing invalid time rows per stage.

        Args:
            input_features: Log-mel features ``(B, T, M)``.
            attention_mask: Optional valid feature frames ``(B, T)``.

        Returns:
            Hidden states ``(B, T_subsampled, H)``.
        """

        hidden_states = input_features.unsqueeze(1)
        valid_lengths = attention_mask.sum(dim=-1) if attention_mask is not None else None

        for layer in self.layers:
            hidden_states = layer(hidden_states)

            if isinstance(layer, nn.Conv2d) and valid_lengths is not None:
                valid_lengths = (
                    valid_lengths
                    + layer.padding[0]
                    + layer.padding[1]
                    - layer.kernel_size[0]
                ) // layer.stride[0] + 1
                current_length = hidden_states.shape[2]
                time_mask = (
                    torch.arange(current_length, device=hidden_states.device)[None, :]
                    < valid_lengths[:, None]
                )
                hidden_states = hidden_states * time_mask[:, None, :, None]

        # (B, C, T, F) -> (B, T, C*F) -> (B, T, H)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            -1,
        )
        return self.linear(hidden_states)
