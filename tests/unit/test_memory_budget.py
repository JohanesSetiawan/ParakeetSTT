"""Accelerator memory ceiling and the automatic batch budget."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.configuration.config import load_config
from src.configuration.settings import MemorySettings
from src.inference.budget import MemoryBudget, _rows_under_ceiling, chunks_that_fit, resolve_memory_budget
from src.models.parakeet import ParakeetTDT
from src.runtime import memory as memory_module
from support import inference_settings, write_tiny_checkpoint

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
        patch.object(memory_module.torch.cuda, "empty_cache", side_effect=lambda: calls.append("empty_cache")),
        patch.object(
            memory_module.torch.cuda,
            "mem_get_info",
            side_effect=lambda _device: calls.append("mem_get_info") or (1000 * MIB, 4096 * MIB),
        ),
        patch.object(memory_module.torch.cuda, "memory_reserved", return_value=2400 * MIB),
    ):
        memory = memory_module.measure_accelerator_memory(device)

    assert calls == ["empty_cache", "mem_get_info"]
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
    memory = MemorySettings(cap_to_free_memory=True, auto_batch_budget=True, reserve_mib=512)

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
