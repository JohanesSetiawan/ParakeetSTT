"""
Anti-aliased sample-rate conversion on a global output grid.

Linear interpolation lets everything above the target Nyquist frequency
(8 kHz at 16 kHz) fold back into the speech band. This module uses windowed
sinc (Hann window) polyphase resampling instead, with the same kernel design
as torchaudio's ``sinc_interp_hann`` (6 zero crossings, 0.99 roll-off),
implemented with plain PyTorch because torchaudio is not a dependency.

Chunked decoding needs one more property: resampling any target interval of a
file must give the same samples as resampling the whole file and slicing it.
Otherwise overlap reuse and fresh reads would disagree, and chunk seams would
shift for ratios such as 44.1 kHz -> 16 kHz. ``SourceBlock`` therefore
aligns every block to the polyphase period, so output index k of a block is
always global target frame ``(block_start / reduced_orig) * reduced_new + k``,
and it includes enough real neighbouring samples on both sides that the
filter never sees the block edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.nn import functional as torch_functional

# Kernel design constants of the windowed-sinc filter. They fix the filter
# quality (stop-band rejection, transition width); they are not user settings.
LOWPASS_FILTER_WIDTH = 6
ROLLOFF = 0.99


@dataclass(frozen=True)
class ResamplingPlan:
    """
    Everything needed to resample ``[target_start, target_end)`` of one file.

    Attributes:
        block_start: First source sample to read; may be negative, in which
            case the samples before the file start are zeros.
        block_end: One past the last source sample to read; may exceed the
            file, in which case the samples after its end are zeros.
        output_offset: Index in the resampled block of ``target_start``.
        output_length: Number of target samples wanted.
    """

    block_start: int
    block_end: int
    output_offset: int
    output_length: int


@lru_cache(maxsize=16)
def _kernel(reduced_orig: int, reduced_new: int) -> tuple[torch.Tensor, int]:
    """
    Build the polyphase windowed-sinc kernel for a reduced rate pair.

    Returns:
        ``(kernel, width)``. The kernel has shape
        ``(reduced_new, 1, 2 * width + reduced_orig)``: one filter per output
        phase. ``width`` is the one-sided filter reach in source samples.
    """

    base_frequency = min(reduced_orig, reduced_new) * ROLLOFF
    width = math.ceil(LOWPASS_FILTER_WIDTH * reduced_orig / base_frequency)

    source_positions = torch.arange(-width, width + reduced_orig, dtype=torch.float64)[None, None] / reduced_orig
    phase_offsets = torch.arange(0, -reduced_new, -1, dtype=torch.float64)[:, None, None] / reduced_new
    time = (phase_offsets + source_positions) * base_frequency  # (new, 1, 2w + orig)
    time = time.clamp(-LOWPASS_FILTER_WIDTH, LOWPASS_FILTER_WIDTH)

    window = torch.cos(time * math.pi / LOWPASS_FILTER_WIDTH / 2) ** 2
    time = time * math.pi
    sinc = torch.where(time == 0, torch.ones_like(time), torch.sin(time) / time)
    kernel = sinc * window * (base_frequency / reduced_orig)
    return kernel.to(torch.float32), width


def reduced_rates(source_rate: int, target_rate: int) -> tuple[int, int]:
    """Return the rate pair divided by its greatest common divisor."""

    divisor = math.gcd(source_rate, target_rate)
    return source_rate // divisor, target_rate // divisor


def plan_block(target_start: int, target_end: int, source_rate: int, target_rate: int) -> ResamplingPlan:
    """
    Compute the source block that yields ``[target_start, target_end)`` exactly.

    The block start is a multiple of the reduced source period, so its first
    output lands on the global grid. One extra period plus the filter width
    of real samples on each side keeps the block edges out of every wanted
    output.
    """

    reduced_orig, reduced_new = reduced_rates(source_rate, target_rate)
    _, width = _kernel(reduced_orig, reduced_new)
    margin = width + reduced_orig

    first_needed = math.floor(target_start * reduced_orig / reduced_new) - margin
    block_period_index = math.floor(first_needed / reduced_orig)
    block_start = block_period_index * reduced_orig
    last_needed = math.ceil(target_end * reduced_orig / reduced_new) + margin
    return ResamplingPlan(
        block_start=block_start,
        block_end=last_needed,
        output_offset=target_start - block_period_index * reduced_new,
        output_length=target_end - target_start,
    )


def resample_block(block: torch.Tensor, plan: ResamplingPlan, source_rate: int, target_rate: int) -> torch.Tensor:
    """
    Resample a mono source block and return the planned target samples.

    Args:
        block: Float32 mono samples ``(plan.block_end - plan.block_start,)``,
            zeros wherever the block lies outside the file.
        plan: The plan the block was read for.
        source_rate: Sample rate of ``block``.
        target_rate: Output sample rate.

    Returns:
        Float32 tensor ``(plan.output_length,)``.
    """

    if block.numel() != plan.block_end - plan.block_start:
        raise ValueError(
            f"Block has {block.numel()} samples, plan expects {plan.block_end - plan.block_start}"
        )
    if plan.output_length <= 0:
        return torch.empty(0, dtype=torch.float32)

    reduced_orig, reduced_new = reduced_rates(source_rate, target_rate)
    kernel, width = _kernel(reduced_orig, reduced_new)

    # Same padding convention as the kernel design: output k of the padded
    # block is centred on input position k * orig / new.
    padded = torch_functional.pad(block.to(torch.float32)[None, None], (width, width + reduced_orig))
    phases = torch_functional.conv1d(padded, kernel, stride=reduced_orig)  # (1, new, periods)
    interleaved = phases.transpose(1, 2).reshape(-1)  # (periods * new,)

    output = interleaved[plan.output_offset : plan.output_offset + plan.output_length]
    if output.numel() != plan.output_length:
        raise RuntimeError(
            f"Resampled block is too short: wanted {plan.output_length} samples from "
            f"offset {plan.output_offset}, block produced {interleaved.numel()}"
        )
    return output.contiguous()
