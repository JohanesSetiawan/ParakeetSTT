"""
Automatic batch budget from the accelerator memory that is actually free.

``inference.max_batch_feature_frames`` is a fixed number, but how much fits
depends on the GPU and on what else is using it. After the model loads, the
memory of one and of three full-length chunks is measured with a real
forward pass, and the batch budget is lowered (never raised) to what fits
under the allocator ceiling from ``runtime.memory``. A setting that would
overflow therefore shrinks before the first batch instead of failing, or on
Windows silently spilling into system memory, halfway through a run.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import torch

from ..audio.features import ParakeetFeatureExtractor
from ..configuration.config import ParakeetConfig
from ..configuration.settings import InferenceSettings, MemorySettings
from ..models.parakeet import ParakeetTDT
from ..runtime.memory import AcceleratorMemory, cap_allocator, measure_accelerator_memory

logger = logging.getLogger(__name__)

MIB = 2**20

# Rows of the two calibration batches. Three instead of two: from one to two
# rows the allocator still reuses cached blocks, so the per-row growth it
# shows is lower than at larger batches (measured 84 MiB from 1 to 2 rows,
# about 100 MiB per row from 1 to 3 and 103 MiB per row up to 8 on the
# reference GPU).
CALIBRATION_ROWS = (1, 3)

# Amplitude of the deterministic calibration noise: speech-like level, so the
# decoding loop runs a realistic number of steps.
CALIBRATION_AMPLITUDE = 0.05


@dataclass(frozen=True)
class MemoryBudget:
    """
    Outcome of the automatic budget, for the terminal and the run log.

    Attributes:
        applied: Whether a ceiling and/or calibration ran (CUDA only).
        reason: Why nothing was applied, when ``applied`` is False.
        ceiling_bytes: Allocator ceiling after calibration, if one was set.
        bytes_per_chunk: Measured growth of reserved memory per extra row of
            a full-length chunk.
        chunks_that_fit: Full-length rows that fit under the ceiling.
        configured_batch_feature_frames: The value from config.toml.
        batch_feature_frames: The value the run uses.
        headroom_reduced: Less than ``memory.reserve_mib`` could be kept free;
            the ceiling is the physically free memory instead.
    """

    applied: bool
    reason: str
    ceiling_bytes: int | None
    bytes_per_chunk: int | None
    chunks_that_fit: int | None
    configured_batch_feature_frames: int
    batch_feature_frames: int
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
        lines.append(
            f"Batch feature frames: {self.batch_feature_frames} "
            f"(configured {self.configured_batch_feature_frames})"
        )
        if self.headroom_reduced:
            lines.append(
                "Warning: less than memory.reserve_mib of GPU memory was free; running "
                "without that headroom. Other GPU programs may now cause an out-of-memory stop."
            )
        return tuple(lines)


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

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    features, mask = extractor([waveform] * rows, [sample_rate] * rows, device)
    model.generate(features, mask)
    torch.cuda.synchronize(device)
    del features, mask
    return torch.cuda.max_memory_reserved(device)


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
        Settings to run with (``max_batch_feature_frames`` possibly lowered)
        and a report of what was decided.

    Raises:
        RuntimeError: If not even one full-length chunk fits in the free
            accelerator memory.
    """

    device = next(model.parameters()).device
    configured_frames = inference.max_batch_feature_frames

    def unchanged(reason: str) -> tuple[InferenceSettings, MemoryBudget]:
        budget = MemoryBudget(False, reason, None, None, None, configured_frames, configured_frames)
        return inference, budget

    if device.type != "cuda":
        return unchanged(f"{device.type} device")
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
    peak_one, peak_more, calibration_rows, baseline = _calibrate(model, configuration, inference, device)

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
    batch_frames = min(configured_frames, fitting_frames)
    logger.info(
        "memory budget ceiling_mib=%.0f baseline_mib=%.0f per_chunk_mib=%.0f "
        "chunks_that_fit=%d batch_feature_frames=%d configured=%d",
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
) -> tuple[int, int, int, int]:
    """
    Peak reserved memory for one and for several full-length chunks.

    Returns:
        ``(peak with one row, peak with more rows, rows of the second run,
        reserved memory before calibration)``. When the larger batch does
        not fit, the second run is the one-row run again.
    """

    extractor = ParakeetFeatureExtractor(configuration)
    sample_rate = int(configuration.feature_extractor["sampling_rate"])
    hop_length = int(configuration.feature_extractor["hop_length"])
    # The longest waveform whose STFT stays within one chunk's frame budget.
    samples = (inference.max_chunk_feature_frames - 1) * hop_length
    generator = torch.Generator().manual_seed(0)
    waveform = torch.randn(samples, generator=generator) * CALIBRATION_AMPLITUDE

    baseline = torch.cuda.memory_reserved(device)
    one_row, more_rows = CALIBRATION_ROWS
    with torch.inference_mode():
        try:
            peak_one = _peak_reserved_for_rows(model, extractor, waveform, one_row, sample_rate, device)
        except torch.OutOfMemoryError as error:
            raise RuntimeError(
                "Not even one full chunk fits in the free GPU memory. Close other programs "
                "that use the GPU, or lower inference.max_chunk_feature_frames."
            ) from error
        try:
            peak_more = _peak_reserved_for_rows(model, extractor, waveform, more_rows, sample_rate, device)
            calibration_rows = more_rows
        except torch.OutOfMemoryError:
            # Too tight for the larger calibration batch: one row is the budget.
            peak_more, calibration_rows = peak_one, one_row
    torch.cuda.empty_cache()
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
