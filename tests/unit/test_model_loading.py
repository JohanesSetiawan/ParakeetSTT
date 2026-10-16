"""Startup decisions in commands.model_loading, without a GPU."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.commands import model_loading
from src.inference.budget import MemoryBudget
from support import inference_settings, memory_settings

KNOBS = memory_settings()

MIB = 2**20
CHUNK = 100  # max_chunk_feature_frames in the tiny settings


def measured_budget(fit: int, per_chunk_mib: int = 56) -> MemoryBudget:
    return MemoryBudget(True, "", 3000 * MIB, per_chunk_mib * MIB, fit, None, fit * CHUNK)


def cuda_model() -> SimpleNamespace:
    """Just enough of a model for the graph budget: a CUDA parameter device."""

    return SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda"))]),
        graph_decoder="previous decoder",
    )


def run_graph_budget(budget: MemoryBudget, rows: int, reserved_mib_per_build: list[int | None]):
    """
    Run the graph budget with a CUDA-looking model.

    Each decoder build reserves the next amount; None makes that build run
    out of memory instead.
    """

    model = cuda_model()
    settings = inference_settings(batch_size=rows, max_batch_feature_frames=rows * CHUNK, cuda_graphs=True)
    enabled_rows: list[int] = []

    def fake_enable(target, resolved):
        build = len(enabled_rows)
        enabled_rows.append(resolved.batch_size)
        target.graph_decoder = f"decoder for {resolved.batch_size}"
        if reserved_mib_per_build[build] is None:
            raise torch.OutOfMemoryError("CUDA out of memory while capturing")
        return True

    # Reserved memory read before each build, and after the ones that finish.
    readings = []
    for reserved_mib in reserved_mib_per_build:
        readings.append(1000 * MIB)
        if reserved_mib is not None:
            readings.append((1000 + reserved_mib) * MIB)
    with (
        patch.object(model_loading, "enable_graph_decoding", side_effect=fake_enable),
        patch.object(model_loading.torch.accelerator, "memory_reserved", side_effect=readings),
        patch.object(model_loading.torch.accelerator, "empty_cache"),
    ):
        on, resolved, warning = model_loading._enable_graph_decoding_within_budget(model, settings, budget)
    return on, resolved, warning, enabled_rows, model


def test_graph_memory_that_fits_keeps_the_rows() -> None:
    on, resolved, warning, enabled_rows, _model = run_graph_budget(measured_budget(29), 23, [60])

    assert (on, resolved.batch_size, warning) == (True, 23, "")
    assert enabled_rows == [23]


def test_graph_memory_lowers_the_rows_and_rebuilds_the_decoder() -> None:
    on, resolved, warning, enabled_rows, model = run_graph_budget(measured_budget(10), 10, [120, 100])

    # ceil(120 / 56) = 3 of the 10 chunks go to the 10-row decoder; the 7-row
    # decoder, measured again, takes 2, which still leaves 8 >= 7.
    assert (on, resolved.batch_size, warning) == (True, 7, "")
    assert resolved.max_batch_feature_frames == 7 * CHUNK
    assert enabled_rows == [10, 7]
    assert model.graph_decoder == "decoder for 7"


def test_graph_decoding_is_turned_off_when_no_chunk_fits_next_to_it() -> None:
    """A tight GPU (another process held most of it in the full tier): fall back, do not fail."""

    on, resolved, warning, enabled_rows, model = run_graph_budget(measured_budget(1, per_chunk_mib=43), 1, [58])

    assert on is False
    assert resolved.batch_size == 1
    assert warning.startswith("Warning: CUDA Graph decoding is off")
    assert enabled_rows == [1]
    assert model.graph_decoder is None


def test_a_one_row_decoder_is_tried_before_turning_graph_decoding_off() -> None:
    """Review finding: graph decoding was turned off although a smaller decoder fit."""

    # Two chunks fit; the two-row decoder takes 100 MiB (3 chunks), the
    # one-row decoder 20 MiB (1 chunk), which leaves room for one row.
    on, resolved, warning, enabled_rows, model = run_graph_budget(measured_budget(2, per_chunk_mib=43), 2, [100, 20])

    assert (on, resolved.batch_size, warning) == (True, 1, "")
    assert resolved.max_batch_feature_frames == CHUNK
    assert enabled_rows == [2, 1]
    assert model.graph_decoder == "decoder for 1"


def test_rows_lowered_for_the_decoder_are_measured_again() -> None:
    """Review finding: rows were lowered once from the larger decoder's cost and never checked."""

    # 10 chunks fit. The 10-row decoder takes 3 (rows 7); the 7-row decoder,
    # measured at 200 MiB, takes 4 (rows 6); the 6-row decoder takes 2, and
    # the 8 chunks left hold 6 rows.
    on, resolved, _warning, enabled_rows, _model = run_graph_budget(measured_budget(10), 10, [120, 200, 60])

    assert (on, resolved.batch_size) == (True, 6)
    assert enabled_rows == [10, 7, 6]


def test_a_decoder_build_that_runs_out_of_memory_is_retried_with_half_the_rows() -> None:
    """Review findings: an out-of-memory capture crashed startup, then dropped straight to one row."""

    # 23 rows run out of memory; 11 rows build with 60 MiB (2 chunks of 29), leaving 27 >= 11.
    on, resolved, warning, enabled_rows, model = run_graph_budget(measured_budget(29), 23, [None, 60])

    assert (on, resolved.batch_size, warning) == (True, 11, "")
    assert resolved.max_batch_feature_frames == 11 * CHUNK
    assert enabled_rows == [23, 11]
    assert model.graph_decoder == "decoder for 11"


def test_graph_decoding_is_turned_off_when_even_the_one_row_build_runs_out_of_memory() -> None:
    on, resolved, warning, enabled_rows, model = run_graph_budget(measured_budget(1), 1, [None])

    assert (on, resolved.batch_size) == (False, 1)
    assert warning.startswith("Warning: CUDA Graph decoding is off")
    assert enabled_rows == [1]
    assert model.graph_decoder is None


def test_unmeasured_budget_keeps_the_decoder_as_built() -> None:
    model = cuda_model()
    budget = MemoryBudget(False, "cpu device", None, None, None, None, None)
    settings = inference_settings(batch_size=16, max_batch_feature_frames=16 * CHUNK, cuda_graphs=True)

    with patch.object(model_loading, "enable_graph_decoding", return_value=True) as enable:
        on, resolved, warning = model_loading._enable_graph_decoding_within_budget(model, settings, budget)

    assert (on, resolved, warning) == (True, settings, "")
    enable.assert_called_once()


@pytest.mark.parametrize(
    ("setting", "dtype", "warned"),
    [
        ("float16", torch.float32, True),  # ROCm, MPS, CPU: the setting is overridden
        ("float16", torch.float16, False),
        ("auto", torch.float32, False),  # "auto" choosing float32 is a decision, not an override
        ("float32", torch.float32, False),
    ],
)
def test_an_overridden_float16_setting_is_warned_about(setting, dtype, warned) -> None:
    """Review finding: an explicit float16 on ROCm turned into float32 with no visible warning."""

    description = "float32 (float16 is used on NVIDIA GPUs only, this is ROCm)"

    warning = model_loading._ignored_precision_warning(setting, dtype, description)

    assert bool(warning) is warned
    if warned:
        assert warning.startswith("Warning:") and "this is ROCm" in warning


@pytest.mark.parametrize(
    ("configured_frames", "rows", "expected_frames"),
    [(None, 23, 23 * CHUNK), (2000, 4, 4 * CHUNK), (300, 16, 300)],
)
def test_resolved_settings_replace_every_auto(configured_frames, rows, expected_frames) -> None:
    budgeted = inference_settings(
        batch_size=None,
        max_batch_feature_frames=configured_frames,
        encoder_precision="auto",
        float16_accumulation=None,
        decode_workers=None,
    )

    resolved = model_loading._resolved_settings(
        budgeted,
        rows=rows,
        encoder_dtype=torch.float16,
        float16_accumulation=True,
        decode_workers=4,
    )

    resolved.require_resolved()
    assert resolved.batch_size == rows
    assert resolved.max_batch_feature_frames == expected_frames
    assert (resolved.encoder_precision, resolved.float16_accumulation, resolved.decode_workers) == (
        "float16",
        True,
        4,
    )


@pytest.mark.parametrize(
    ("configured_rows", "fit", "budget_rows", "rows", "expected"),
    [
        (None, 34, 23, 23, "Batch rows: 23 (auto: 70% of the 34 full chunks that fit, at most 64)"),
        (None, None, 16, 16, "Batch rows: 16 (auto: memory not measured on this device)"),
        (16, 29, 16, 16, "Batch rows: 16 (configured 16)"),
        (16, 9, 9, 9, "Batch rows: 9 (configured 16, 9 full chunks fit)"),
        (
            None,
            34,
            23,
            18,
            "Batch rows: 18 (auto: 70% of the 34 full chunks that fit, at most 64, "
            "lowered from 23 for the CUDA Graph decoder)",
        ),
    ],
)
def test_batch_rows_line_names_the_reason(configured_rows, fit, budget_rows, rows, expected) -> None:
    configured = inference_settings(batch_size=configured_rows)
    resolved = inference_settings(batch_size=rows)
    budget = MemoryBudget(fit is not None, "", None, None, fit, None, None)

    assert model_loading._batch_rows_line(configured, resolved, budget, KNOBS, budget_rows=budget_rows) == expected
