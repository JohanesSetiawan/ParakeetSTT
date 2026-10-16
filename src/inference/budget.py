"""
Automatic batch budget from the accelerator memory that is actually free.

How much fits depends on the GPU and on what else is using it. After the
model loads, the memory of one and of three full-length chunks is measured
with a real forward pass, which gives how many full chunks fit under the
allocator ceiling from ``runtime.memory``. A configured
``inference.max_batch_feature_frames`` is lowered (never raised) to that;
with ``"auto"`` the budget is what fits. ``batch_rows_for`` then picks the
rows per batch. A setting that would overflow therefore shrinks before the
first batch instead of failing, or on Windows silently spilling into system
memory, halfway through a run.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Callable

import torch

from ..audio.features import ParakeetFeatureExtractor
from ..configuration.config import ParakeetConfig
from ..configuration.settings import InferenceSettings, MemorySettings
from ..models.parakeet import ParakeetTDT
from ..runtime.device import is_nvidia_gpu
from ..runtime.memory import AcceleratorMemory, cap_allocator, measure_accelerator_memory

logger = logging.getLogger(__name__)

MIB = 2**20

# Rows of the calibration batches: one row, then the first of these larger
# batches that fits. The growth per row is the difference divided by the rows
# between them, so far-apart points keep noise in the one-row peak small. With
# 1 and 3 rows, 20 MiB that earlier work had left in the allocator's cache
# raised the one-row peak and turned 53 MiB per row into 43, and the batch
# chosen from it ran out of memory. Measured growth per row on the reference
# GPU (float16 encoder): 53 MiB from 1 to 3 rows, 55 from 1 to 8, 59 from 8 to
# 16, 63 from 16 to 24. It still grows a little with the batch (larger
# allocator blocks), which memory.auto_batch_memory_share leaves room for.
CALIBRATION_ONE_ROW = 1
CALIBRATION_MORE_ROWS = (8, 3)

# Amplitude of the deterministic calibration noise: speech-like level, so the
# decoding loop runs a realistic number of steps.
CALIBRATION_AMPLITUDE = 0.05



@dataclass(frozen=True)
class MemoryBudget:
    """
    Outcome of the automatic budget, for the terminal and the run log.

    Attributes:
        applied: Whether a ceiling and/or calibration ran (NVIDIA GPUs only).
        reason: Why nothing was applied, when ``applied`` is False.
        ceiling_bytes: Allocator ceiling after calibration, if one was set.
        bytes_per_chunk: Measured growth of reserved memory per extra row of
            a full-length chunk.
        chunks_that_fit: Full-length rows that fit under the ceiling.
        configured_batch_feature_frames: The value from config.toml, or None
            for "auto".
        batch_feature_frames: The value the run uses, or None while it is
            still "auto" (nothing was measured).
        headroom_reduced: Less than ``memory.reserve_mib`` could be kept free;
            the ceiling is the physically free memory instead.
    """

    applied: bool
    reason: str
    ceiling_bytes: int | None
    bytes_per_chunk: int | None
    chunks_that_fit: int | None
    configured_batch_feature_frames: int | None
    batch_feature_frames: int | None
    headroom_reduced: bool = False

    def lines(self) -> tuple[str, ...]:
        """Plain-text report lines."""

        if not self.applied:
            return (f"Memory budget: not applied ({self.reason})",)
        lines = []
        if self.ceiling_bytes is not None:
            lines.append(f"Accelerator memory ceiling: {self.ceiling_bytes / MIB:.0f} MiB")
        if self.bytes_per_chunk is not None and self.chunks_that_fit is not None:
            lines.append(
                f"Memory per full chunk: {self.bytes_per_chunk / MIB:.0f} MiB, "
                f"full chunks that fit: {self.chunks_that_fit}"
            )
        configured = self.configured_batch_feature_frames
        lines.append(
            f"Batch feature frames: {self.batch_feature_frames} "
            f"(configured {configured if configured is not None else 'auto'})"
        )
        return tuple(lines)

    def warnings(self) -> tuple[str, ...]:
        """Lines the user must see even when details stay in the log."""

        if not self.headroom_reduced:
            return ()
        return (
            "Warning: less than memory.reserve_mib of GPU memory was free; running "
            "without that headroom. Other GPU programs may now cause an out-of-memory stop.",
        )


def chunks_that_fit(
    ceiling_bytes: int,
    baseline_bytes: int,
    peak_one_row_bytes: int,
    peak_rows_bytes: int,
    rows: int,
) -> tuple[int, int]:
    """
    Linear memory model from two calibration peaks.

    Args:
        ceiling_bytes: Memory PyTorch may reserve.
        baseline_bytes: Reserved memory before calibration (the weights).
        peak_one_row_bytes: Peak reserved memory with one full chunk.
        peak_rows_bytes: Peak reserved memory with ``rows`` full chunks.
        rows: Rows of the second calibration batch, more than one.

    Returns:
        ``(rows that fit, bytes per extra row)``. Zero rows means not even
        one chunk fits.
    """

    per_row = (peak_rows_bytes - peak_one_row_bytes) // (rows - 1)
    if per_row <= 0:
        # Cached blocks hid the growth; charge the whole one-row cost per row,
        # which overestimates and therefore stays safe.
        per_row = max(1, peak_one_row_bytes - baseline_bytes)
    if ceiling_bytes < peak_one_row_bytes:
        return 0, per_row
    return 1 + (ceiling_bytes - peak_one_row_bytes) // per_row, per_row


def _peak_reserved_for_rows(
    model: ParakeetTDT,
    extractor: ParakeetFeatureExtractor,
    waveform: torch.Tensor,
    rows: int,
    sample_rate: int,
    device: torch.device,
) -> int:
    """Run features and generation on ``rows`` copies; return peak reserved bytes."""

    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats(device)
    features, mask = extractor([waveform] * rows, [sample_rate] * rows, device)
    model.generate(features, mask)
    torch.accelerator.synchronize(device)
    del features, mask
    return torch.accelerator.max_memory_reserved(device)


def resolve_memory_budget(
    model: ParakeetTDT,
    configuration: ParakeetConfig,
    inference: InferenceSettings,
    memory: MemorySettings,
) -> tuple[InferenceSettings, MemoryBudget]:
    """
    Apply the allocator ceiling and derive the batch budget for this device.

    Args:
        model: The loaded model; its parameters' device is used.
        configuration: Checkpoint configuration (feature extractor input).
        inference: Settings from config.toml.
        memory: The ``[memory]`` policy.

    Returns:
        Settings to run with (``max_batch_feature_frames`` lowered to what
        fits, or set from what fits when it was "auto") and a report of what
        was decided. Without a measurement an "auto" budget stays None.

    Raises:
        RuntimeError: If not even one full-length chunk fits in the free
            accelerator memory.
    """

    device = next(model.parameters()).device
    configured_frames = inference.max_batch_feature_frames

    def unchanged(reason: str) -> tuple[InferenceSettings, MemoryBudget]:
        budget = MemoryBudget(False, reason, None, None, None, configured_frames, configured_frames)
        return inference, budget

    if not is_nvidia_gpu(device):
        place = "ROCm" if device.type == "cuda" else device.type
        return unchanged(f"{place} device, measured on NVIDIA GPUs only")
    if not memory.cap_to_free_memory and not memory.auto_batch_budget:
        return unchanged("disabled in [memory]")

    reserve_bytes = memory.reserve_mib * MIB
    # The reserve is preferred headroom, not a hard requirement: when keeping
    # it would leave too little room, the physically free memory is the cap,
    # because running with less headroom beats refusing work that fits. Either
    # way the cap stays within physical memory, so nothing spills.
    before = measure_accelerator_memory(device)
    if not memory.auto_batch_budget:
        headroom_reduced = before.free <= reserve_bytes
        ceiling = before.ceiling(0 if headroom_reduced else reserve_bytes)
        cap_allocator(device, ceiling, before.total)
        budget = MemoryBudget(
            True,
            "",
            ceiling,
            None,
            None,
            configured_frames,
            configured_frames,
            headroom_reduced=headroom_reduced,
        )
        return inference, budget

    # Calibrate under the physical limit so the measurement cannot spill.
    if memory.cap_to_free_memory:
        cap_allocator(device, before.ceiling(0), before.total)
    # With the cap, a calibration batch that does not fit fails with an
    # out-of-memory error. Without it, Windows would spill it into system
    # memory and it would measure as if it fit, so the free memory is checked
    # before each larger batch instead.
    physical_limit = None if memory.cap_to_free_memory else (lambda: measure_accelerator_memory(device).ceiling(0))
    peak_one, peak_more, calibration_rows, baseline = _calibrate(
        model,
        configuration,
        inference,
        device,
        physical_limit=physical_limit,
    )

    # Loading kernels and library workspaces during calibration grew the CUDA
    # context outside the allocator, so measure the free memory again.
    after = measure_accelerator_memory(device)
    rows, per_row, ceiling, headroom_reduced = _rows_under_ceiling(
        after,
        reserve_bytes,
        baseline,
        peak_one,
        peak_more,
        calibration_rows,
    )
    if rows < 1:
        raise RuntimeError(
            f"One full chunk needs about {(peak_one - baseline) / MIB:.0f} MiB but only "
            f"{(after.free) / MIB:.0f} MiB of GPU memory is free. Close other programs that "
            "use the GPU, or lower inference.max_chunk_feature_frames."
        )
    if memory.cap_to_free_memory:
        cap_allocator(device, ceiling, after.total)

    fitting_frames = rows * inference.max_chunk_feature_frames
    if configured_frames is None:
        batch_frames = fitting_frames
    else:
        batch_frames = min(configured_frames, fitting_frames)
    logger.info(
        "memory budget ceiling_mib=%.0f baseline_mib=%.0f per_chunk_mib=%.0f "
        "chunks_that_fit=%d batch_feature_frames=%d configured=%s",
        ceiling / MIB,
        baseline / MIB,
        per_row / MIB,
        rows,
        batch_frames,
        configured_frames,
    )
    budget = MemoryBudget(
        True,
        "",
        ceiling,
        per_row,
        rows,
        configured_frames,
        batch_frames,
        headroom_reduced=headroom_reduced,
    )
    return dataclasses.replace(inference, max_batch_feature_frames=batch_frames), budget


def _calibrate(
    model: ParakeetTDT,
    configuration: ParakeetConfig,
    inference: InferenceSettings,
    device: torch.device,
    *,
    physical_limit: Callable[[], int] | None,
) -> tuple[int, int, int, int]:
    """
    Peak reserved memory for one and for several full-length chunks.

    Args:
        physical_limit: None when the allocator is capped (an oversized
            batch then fails with an out-of-memory error, which is caught).
            Otherwise it returns, at the moment it is called, the most memory
            PyTorch can reserve without exceeding the free VRAM; a larger
            batch is tried only when the one-row cost, charged per row
            (overhead included, so the estimate is on the safe side),
            predicts it stays within that. Without the cap, an oversized
            batch would not fail on Windows but spill into system memory and
            measure as if it fit.

    Returns:
        ``(peak with one row, peak with more rows, rows of the second run,
        reserved memory before calibration)``. When no larger batch fits,
        the second run is the one-row run again.
    """

    extractor = ParakeetFeatureExtractor(configuration)
    sample_rate = int(configuration.feature_extractor["sampling_rate"])
    hop_length = int(configuration.feature_extractor["hop_length"])
    # The longest waveform whose STFT stays within one chunk's frame budget.
    samples = (inference.max_chunk_feature_frames - 1) * hop_length
    generator = torch.Generator().manual_seed(0)
    waveform = torch.randn(samples, generator=generator) * CALIBRATION_AMPLITUDE

    baseline = torch.accelerator.memory_reserved(device)
    with torch.inference_mode():
        try:
            peak_one = _peak_reserved_for_rows(model, extractor, waveform, CALIBRATION_ONE_ROW, sample_rate, device)
        except torch.OutOfMemoryError as error:
            raise RuntimeError(
                "Not even one full chunk fits in the free GPU memory. Close other programs "
                "that use the GPU, or lower inference.max_chunk_feature_frames."
            ) from error
        # Too tight for every larger batch: one row is the budget.
        peak_more, calibration_rows = peak_one, CALIBRATION_ONE_ROW
        one_row_cost = max(1, peak_one - baseline)
        for rows in CALIBRATION_MORE_ROWS:
            if physical_limit is not None and peak_one + (rows - 1) * one_row_cost > physical_limit():
                continue
            try:
                peak_more = _peak_reserved_for_rows(model, extractor, waveform, rows, sample_rate, device)
            except torch.OutOfMemoryError:
                torch.accelerator.empty_cache()
                continue
            calibration_rows = rows
            break
    torch.accelerator.empty_cache()
    return peak_one, peak_more, calibration_rows, baseline


def _rows_under_ceiling(
    memory: AcceleratorMemory,
    reserve_bytes: int,
    baseline: int,
    peak_one: int,
    peak_more: int,
    calibration_rows: int,
) -> tuple[int, int, int, bool]:
    """
    Rows that fit, keeping the reserve free when possible.

    Returns:
        ``(rows, bytes per row, ceiling, headroom reduced)``. The reserve is
        dropped only when keeping it would leave no room for a single chunk.
    """

    def fit(ceiling: int) -> tuple[int, int]:
        if calibration_rows == 1:
            return (1 if ceiling >= peak_one else 0), max(1, peak_one - baseline)
        return chunks_that_fit(ceiling, baseline, peak_one, peak_more, calibration_rows)

    preferred_ceiling = memory.ceiling(reserve_bytes)
    rows, per_row = fit(preferred_ceiling)
    if rows >= 1:
        return rows, per_row, preferred_ceiling, False
    physical_ceiling = memory.ceiling(0)
    rows, per_row = fit(physical_ceiling)
    return rows, per_row, physical_ceiling, True


# =============================================================================
# Rows per batch
# =============================================================================


def batch_rows_for(inference: InferenceSettings, budget: MemoryBudget, memory: MemorySettings) -> int:
    """
    Rows per batch for this run.

    A configured ``batch_size`` is kept, lowered to the full chunks that fit
    when that was measured. ``"auto"`` takes ``memory.auto_batch_memory_share``
    of the chunks that fit, at most ``memory.auto_batch_max_rows``; without a
    measurement it is ``memory.unmeasured_batch_rows``. Larger batches were
    never slower on the reference GPU (8, 16, and 24 rows: 8.9, 8.0, and
    7.8 s for a 74-minute file), because the decoding loop's per-batch cost
    is shared by more chunks.

    Returns:
        At least one row.
    """

    fit = budget.chunks_that_fit
    if inference.batch_size is not None:
        if fit is None:
            return inference.batch_size
        return max(1, min(inference.batch_size, fit))
    if fit is None:
        return memory.unmeasured_batch_rows
    share_of_fit = math.floor(fit * memory.auto_batch_memory_share)
    return max(1, min(memory.auto_batch_max_rows, share_of_fit))


def rows_after_reserving(budget: MemoryBudget, rows: int, reserved_bytes: int) -> int:
    """
    Rows that still fit once ``reserved_bytes`` more stay allocated.

    The CUDA Graph decoder is built after the budget is measured, sized for
    the chosen rows, and keeps its static buffers and capture pool for the
    whole run. Each full chunk needs ``bytes_per_chunk``, so that much less
    room costs ``ceil(reserved / bytes_per_chunk)`` chunks of the measured
    capacity.

    Returns:
        ``rows``, lowered when the capacity left is smaller; at least zero.
    """

    if budget.chunks_that_fit is None or not budget.bytes_per_chunk or reserved_bytes <= 0:
        return rows
    lost_chunks = math.ceil(reserved_bytes / budget.bytes_per_chunk)
    return max(0, min(rows, budget.chunks_that_fit - lost_chunks))
