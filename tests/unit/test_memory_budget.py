"""Accelerator memory ceiling and the automatic batch budget."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.configuration.config import load_config
from src.configuration.settings import MemorySettings
from src.inference.budget import MemoryBudget, chunks_that_fit, resolve_memory_budget
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


def test_ceiling_is_reserved_plus_free_minus_reserve() -> None:
    fractions: list[float] = []
    device = torch.device("cuda", 0)
    with (
        patch.object(memory_module.torch.cuda, "mem_get_info", return_value=(1000 * MIB, 4096 * MIB)),
        patch.object(memory_module.torch.cuda, "empty_cache"),
        patch.object(memory_module.torch.cuda, "memory_reserved", return_value=2400 * MIB),
        patch.object(
            memory_module.torch.cuda,
            "set_per_process_memory_fraction",
            side_effect=lambda fraction, _device: fractions.append(fraction),
        ),
    ):
        ceiling = memory_module.cap_allocator_to_free_memory(device, 512 * MIB)

    assert ceiling == (2400 + 1000 - 512) * MIB
    assert fractions == [pytest.approx(ceiling / (4096 * MIB))]


def test_free_memory_below_the_reserve_is_reported() -> None:
    device = torch.device("cuda", 0)
    with (
        patch.object(memory_module.torch.cuda, "mem_get_info", return_value=(300 * MIB, 4096 * MIB)),
        patch.object(memory_module.torch.cuda, "empty_cache"),
        patch.object(memory_module.torch.cuda, "memory_reserved", return_value=2400 * MIB),
        patch.object(memory_module.torch.cuda, "set_per_process_memory_fraction") as setter,
    ):
        with pytest.raises(RuntimeError, match="memory.reserve_mib"):
            memory_module.cap_allocator_to_free_memory(device, 512 * MIB)

    setter.assert_not_called()


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


def test_report_lines_name_the_used_and_configured_budget() -> None:
    budget = MemoryBudget(True, "", 3000 * MIB, 100 * MIB, 6, 12000, 9000)

    text = "\n".join(budget.lines())

    assert "3000 MiB" in text
    assert "full chunks that fit: 6" in text
    assert "Batch feature frames: 9000 (configured 12000)" in text
