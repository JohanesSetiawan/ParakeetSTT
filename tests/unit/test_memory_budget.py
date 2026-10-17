"""Accelerator memory ceiling and the automatic batch budget."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.configuration.config import load_config
from src.inference.budget import MemoryBudget, _rows_under_ceiling, chunks_that_fit, resolve_memory_budget
from src.models.parakeet import ParakeetTDT
from src.runtime import memory as memory_module
from support import inference_settings, memory_settings, write_tiny_checkpoint

MIB = 2**20


# =============================================================================
# Linear memory model
# =============================================================================


def test_rows_follow_the_measured_growth_per_row() -> None:
    # Weights 2400 MiB, one row peaks at 2500, three rows at 2700: 100 MiB per
    # extra row. A 3300 MiB ceiling leaves room for 1 + 800 // 100 = 9 rows.
    rows, per_row = chunks_that_fit(3300 * MIB, 2400 * MIB, 2500 * MIB, 2700 * MIB, rows=3)

    assert per_row == 100 * MIB
    assert rows == 9


def test_ceiling_below_one_row_fits_nothing() -> None:
    rows, _per_row = chunks_that_fit(2450 * MIB, 2400 * MIB, 2500 * MIB, 2700 * MIB, rows=3)

    assert rows == 0


def test_hidden_growth_falls_back_to_the_whole_one_row_cost() -> None:
    # Cached blocks made three rows peak no higher than one: charge the full
    # one-row cost (100 MiB over the weights) per row, which is conservative.
    rows, per_row = chunks_that_fit(2800 * MIB, 2400 * MIB, 2500 * MIB, 2500 * MIB, rows=3)

    assert per_row == 100 * MIB
    assert rows == 4


@pytest.mark.parametrize("ceiling_mib", [2500, 2550, 2999, 3000, 5000])
def test_predicted_batches_never_exceed_the_ceiling(ceiling_mib: int) -> None:
    rows, per_row = chunks_that_fit(ceiling_mib * MIB, 2400 * MIB, 2500 * MIB, 2700 * MIB, rows=3)

    predicted_peak = 2500 * MIB + (rows - 1) * per_row
    assert predicted_peak <= ceiling_mib * MIB
    assert predicted_peak + per_row > ceiling_mib * MIB


# =============================================================================
# Allocator ceiling
# =============================================================================


def test_ceiling_keeps_the_requested_headroom_of_free_memory() -> None:
    memory = memory_module.AcceleratorMemory(reserved=2400 * MIB, free=1000 * MIB, total=4096 * MIB)

    assert memory.ceiling(256 * MIB) == (2400 + 1000 - 256) * MIB
    assert memory.ceiling(0) == 3400 * MIB
    # Headroom larger than the free memory leaves PyTorch what it holds.
    assert memory.ceiling(2000 * MIB) == 2400 * MIB


def test_measurement_releases_the_cache_first() -> None:
    device = torch.device("cuda", 0)
    calls: list[str] = []
    with (
        patch.object(memory_module.torch.accelerator, "empty_cache", side_effect=lambda: calls.append("empty_cache")),
        patch.object(
            memory_module.torch.accelerator,
            "get_memory_info",
            side_effect=lambda _device: calls.append("get_memory_info") or (1000 * MIB, 4096 * MIB),
        ),
        patch.object(memory_module.torch.accelerator, "memory_reserved", return_value=2400 * MIB),
    ):
        memory = memory_module.measure_accelerator_memory(device)

    assert calls == ["empty_cache", "get_memory_info"]
    assert memory == memory_module.AcceleratorMemory(2400 * MIB, 1000 * MIB, 4096 * MIB)


def test_cap_sets_the_fraction_of_total_memory() -> None:
    fractions: list[float] = []
    with patch.object(
        memory_module.torch.cuda,
        "set_per_process_memory_fraction",
        side_effect=lambda fraction, _device: fractions.append(fraction),
    ):
        memory_module.cap_allocator(torch.device("cuda", 0), 3072 * MIB, 4096 * MIB)

    assert fractions == [0.75]


def test_reserve_is_dropped_only_when_no_chunk_would_fit_with_it() -> None:
    # 2400 MiB weights, 100 MiB per chunk. 350 MiB free: with a 256 MiB
    # reserve not even one chunk fits, without it three do.
    tight = memory_module.AcceleratorMemory(reserved=2400 * MIB, free=350 * MIB, total=4096 * MIB)
    rows, _per_row, ceiling, reduced = _rows_under_ceiling(tight, 256 * MIB, 2400 * MIB, 2500 * MIB, 2700 * MIB, 3)

    assert reduced is True
    assert ceiling == 2750 * MIB
    assert rows == 3

    roomy = memory_module.AcceleratorMemory(reserved=2400 * MIB, free=1000 * MIB, total=4096 * MIB)
    rows, _per_row, ceiling, reduced = _rows_under_ceiling(roomy, 256 * MIB, 2400 * MIB, 2500 * MIB, 2700 * MIB, 3)

    assert reduced is False
    assert ceiling == (3400 - 256) * MIB
    assert rows == 1 + (ceiling - 2500 * MIB) // (100 * MIB)


def test_nothing_fits_when_even_physical_memory_is_too_small() -> None:
    full = memory_module.AcceleratorMemory(reserved=2400 * MIB, free=50 * MIB, total=4096 * MIB)

    rows, _per_row, _ceiling, _reduced = _rows_under_ceiling(full, 256 * MIB, 2400 * MIB, 2500 * MIB, 2700 * MIB, 3)

    assert rows == 0


# =============================================================================
# Budget resolution
# =============================================================================


def test_budget_is_not_applied_off_cuda(tmp_path: Path) -> None:
    configuration = load_config(write_tiny_checkpoint(tmp_path))
    model = ParakeetTDT(configuration).eval()
    settings = inference_settings(max_batch_feature_frames=300)
    memory = memory_settings(reserve_mib=512)

    resolved, budget = resolve_memory_budget(model, configuration, settings, memory)

    assert resolved == settings
    assert budget.applied is False
    assert budget.batch_feature_frames == 300
    assert "cpu" in budget.lines()[0]


def test_headroom_warning_is_kept_apart_from_the_details() -> None:
    """The warning must reach the terminal even when details stay in the log."""

    reduced = MemoryBudget(True, "", 3000 * MIB, 100 * MIB, 6, 12000, 9000, headroom_reduced=True)
    normal = MemoryBudget(True, "", 3000 * MIB, 100 * MIB, 6, 12000, 9000)

    assert len(reduced.warnings()) == 1 and "reserve_mib" in reduced.warnings()[0]
    assert not any("Warning" in line for line in reduced.lines())
    assert normal.warnings() == ()


def test_report_lines_name_the_used_and_configured_budget() -> None:
    budget = MemoryBudget(True, "", 3000 * MIB, 100 * MIB, 6, 12000, 9000)

    text = "\n".join(budget.lines())

    assert "3000 MiB" in text
    assert "full chunks that fit: 6" in text
    assert "Batch feature frames: 9000 (configured 12000)" in text


# =============================================================================
# Rows per batch
# =============================================================================


def budget_with_fit(chunks: int | None, per_chunk_mib: int = 56) -> MemoryBudget:
    return MemoryBudget(
        chunks is not None,
        "",
        3000 * MIB if chunks is not None else None,
        per_chunk_mib * MIB if chunks is not None else None,
        chunks,
        None,
        None,
    )


@pytest.mark.parametrize(
    ("batch_size", "fit", "expected"),
    [
        (None, 34, 23),  # auto: 70 percent of what fits (the reference GPU)
        (None, 200, 64),  # auto: never more than 64
        (None, 1, 1),  # auto: at least one row
        (None, None, 16),  # auto without a measurement
        (16, 29, 16),  # configured and fits
        (16, 9, 9),  # configured but lowered to what fits
        (16, None, 16),  # configured, nothing measured
    ],
)
def test_batch_rows_follow_the_setting_and_the_measurement(batch_size, fit, expected) -> None:
    from src.inference.budget import batch_rows_for

    settings = inference_settings(batch_size=batch_size, max_batch_feature_frames=None)

    assert batch_rows_for(settings, budget_with_fit(fit), memory_settings()) == expected


def test_batch_rows_use_the_configured_knobs() -> None:
    from src.inference.budget import batch_rows_for

    settings = inference_settings(batch_size=None, max_batch_feature_frames=None)
    knobs = memory_settings(auto_batch_memory_share=0.5, auto_batch_max_rows=10, unmeasured_batch_rows=4)

    assert batch_rows_for(settings, budget_with_fit(14), knobs) == 7
    assert batch_rows_for(settings, budget_with_fit(100), knobs) == 10
    assert batch_rows_for(settings, budget_with_fit(None), knobs) == 4


@pytest.mark.parametrize(
    ("rows", "reserved_mib", "expected"),
    [
        (23, 30, 23),  # 30 MiB costs one of 29 chunks: 28 still covers 23
        (29, 30, 28),
        (29, 120, 26),  # ceil(120 / 56) = 3 chunks
        (1, 100, 0),  # nothing left for a single chunk
        (16, 0, 16),
    ],
)
def test_graph_memory_is_taken_off_the_rows(rows, reserved_mib, expected) -> None:
    from src.inference.budget import rows_after_reserving

    assert rows_after_reserving(budget_with_fit(29 if rows > 1 else 1), rows, reserved_mib * MIB) == expected


def test_nothing_measured_leaves_the_rows_alone() -> None:
    from src.inference.budget import rows_after_reserving

    assert rows_after_reserving(budget_with_fit(None), 16, 500 * MIB) == 16


def test_snapshot_reports_unavailable_statistics_instead_of_failing() -> None:
    assert memory_module.accelerator_memory_snapshot(torch.device("cpu")) == {
        "available": False,
        "device": "cpu",
    }
    with patch.object(memory_module.torch.accelerator, "memory_allocated", side_effect=RuntimeError("unsupported")):
        snapshot = memory_module.accelerator_memory_snapshot(torch.device("mps"))

    assert snapshot == {"available": False, "device": "mps"}


def test_calibration_skips_larger_batches_it_is_not_allowed(tmp_path: Path) -> None:
    """Each larger batch is checked first, with its predicted peak, and skipped when refused."""

    from src.inference import budget as budget_module

    configuration = load_config(write_tiny_checkpoint(tmp_path))
    peaks = {1: 1100 * MIB, 3: 1300 * MIB, 8: 1800 * MIB}
    tried: list[int] = []

    def fake_peak(_model, _extractor, _waveform, rows, _rate, _device):
        tried.append(rows)
        return peaks[rows]

    with (
        patch.object(budget_module, "_peak_reserved_for_rows", side_effect=fake_peak),
        patch.object(budget_module.torch.accelerator, "memory_reserved", return_value=1000 * MIB),
        patch.object(budget_module.torch.accelerator, "empty_cache"),
    ):
        # One row costs 100 MiB: 8 rows are predicted at 1800 MiB, 3 at 1300.
        tight = budget_module._calibrate(
            None,
            configuration,
            inference_settings(),
            torch.device("cuda"),
            larger_batch_allowed=lambda predicted: predicted <= 1500 * MIB,
        )
        roomy = budget_module._calibrate(
            None,
            configuration,
            inference_settings(),
            torch.device("cuda"),
            larger_batch_allowed=lambda predicted: predicted <= 4000 * MIB,
        )

    assert tried == [1, 3, 1, 8]
    assert tight == (1100 * MIB, 1300 * MIB, 3, 1000 * MIB)
    assert roomy == (1100 * MIB, 1800 * MIB, 8, 1000 * MIB)


def test_capped_calibration_falls_back_to_three_rows_when_eight_run_out_of_memory(tmp_path: Path) -> None:
    """With the cap (the default) every larger batch is tried; the one that does not fit fails and is caught."""

    from src.inference import budget as budget_module

    configuration = load_config(write_tiny_checkpoint(tmp_path))
    tried: list[int] = []

    def fake_peak(_model, _extractor, _waveform, rows, _rate, _device):
        tried.append(rows)
        if rows == 8:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return {1: 1100 * MIB, 3: 1300 * MIB}[rows]

    with (
        patch.object(budget_module, "_peak_reserved_for_rows", side_effect=fake_peak),
        patch.object(budget_module.torch.accelerator, "memory_reserved", return_value=1000 * MIB),
        patch.object(budget_module.torch.accelerator, "empty_cache") as empty_cache,
    ):
        calibrated = budget_module._calibrate(
            None,
            configuration,
            inference_settings(),
            torch.device("cuda"),
            larger_batch_allowed=lambda _predicted: True,
        )

    assert tried == [1, 8, 3]
    assert calibrated == (1100 * MIB, 1300 * MIB, 3, 1000 * MIB)
    # The failed batch's blocks are released before the next one is measured.
    assert empty_cache.call_count >= 2


@pytest.mark.parametrize(
    ("capped", "predicted_mib", "allowed"),
    [
        (True, 5000, True),  # capped: always tried, an oversized batch fails and is caught
        (False, 1900, True),
        (False, 2100, False),  # would spill past the 2000 MiB free now
    ],
)
def test_a_larger_calibration_batch_is_checked_against_the_memory_free_now(capped, predicted_mib, allowed) -> None:
    """Review finding: the cap set before calibration could exceed what the grown CUDA context left free."""

    from src.inference import budget as budget_module
    from src.runtime.memory import AcceleratorMemory

    device = torch.device("cuda")
    now = AcceleratorMemory(reserved=1200 * MIB, free=800 * MIB, total=4096 * MIB)
    with (
        patch.object(budget_module, "measure_accelerator_memory", return_value=now),
        patch.object(budget_module, "cap_allocator") as cap,
    ):
        result = budget_module._allow_larger_calibration_batch(device, capped, predicted_mib * MIB)

    assert result is allowed
    if capped:
        cap.assert_called_once_with(device, 2000 * MIB, 4096 * MIB)
    else:
        cap.assert_not_called()


def test_budget_is_not_measured_on_rocm(tmp_path: Path) -> None:
    """Review finding: AMD GPUs are "cuda" devices too; the budget is verified on NVIDIA only."""

    from types import SimpleNamespace

    from src.inference import budget as budget_module

    configuration = load_config(write_tiny_checkpoint(tmp_path))
    rocm_model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda"))]))
    settings = inference_settings()
    with (
        patch.object(budget_module.torch.version, "hip", "6.2"),
        patch.object(budget_module, "cap_allocator") as cap,
    ):
        resolved, budget = budget_module.resolve_memory_budget(rocm_model, configuration, settings, memory_settings())

    assert resolved == settings
    assert budget.applied is False
    assert budget.reason == "ROCm device, measured on NVIDIA GPUs only"
    cap.assert_not_called()
