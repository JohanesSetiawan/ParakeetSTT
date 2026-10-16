"""
Model startup shared by the commands: device, precision, load, and budget.

A float16 encoder is read from the derived ``model.encoder-float16.pth``
(built once from ``model.pth``), which halves the bytes read and the peak
host memory of the load; float32 reads ``model.pth`` directly.

Every ``"auto"`` setting is decided here for the device in use, from what
the device is (compute capability, CPU count) and what it measures (a float16
accumulation microbenchmark, the memory of real chunks). The settings that
leave this module are fully resolved, and each decision is reported with its
reason.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from ..checkpoint.bootstrap import ensure_first_run_ready
from ..checkpoint.derived import HALF_ENCODER_FILENAME, ensure_half_encoder_checkpoint
from ..configuration.config import ParakeetConfig
from ..configuration.settings import InferenceSettings, Settings
from ..inference.budget import (
    AUTO_BATCH_MEMORY_SHARE,
    AUTO_BATCH_ROWS_LIMIT,
    MemoryBudget,
    batch_rows_for,
    resolve_memory_budget,
    rows_after_reserving,
)
from ..inference.offline import enable_graph_decoding
from ..models.parakeet import ParakeetTDT, load_model
from ..runtime.device import (
    FLOAT16_ACCUMULATION_MIN_SPEEDUP,
    RuntimeReport,
    apply_float16_accumulation,
    apply_float32_matmul_precision,
    build_supports_device,
    cuda_capability,
    default_decode_workers,
    describe_runtime,
    encoder_dtype_for,
    measure_float16_accumulation_speedup,
    select_device,
)

# Literal name: under `python -m` __name__ would be "__main__" for the caller.
logger = logging.getLogger("src.commands.model_loading")

LoadedModel = tuple[ParakeetTDT, ParakeetConfig, dict[str, Any]]


def inference_model_loader(
    device: torch.device,
    encoder_dtype: torch.dtype,
    progress_callback: Callable[[str], None] | None = None,
) -> Callable[[Path], LoadedModel]:
    """
    Build the strict loader passed to ``ensure_first_run_ready``.

    Args:
        device: Device the model runs on.
        encoder_dtype: Resolved encoder precision for that device.
        progress_callback: Optional plain-text progress output (the one-time
            float16 file preparation reports through it).

    Returns:
        A function from the prepared weights directory to the loaded model.
    """

    def load(weights_dir: Path) -> LoadedModel:
        checkpoint_file = None
        if encoder_dtype == torch.float16:

            def half_encoder_checkpoint() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
                # Unfolded: the derived file must keep the checkpoint layout.
                model, _configuration, metadata = load_model(
                    weights_dir,
                    torch.device("cpu"),
                    encoder_dtype=torch.float16,
                    fold_batch_norm=False,
                )
                return model.state_dict(), metadata

            try:
                checkpoint_file = ensure_half_encoder_checkpoint(
                    weights_dir,
                    half_encoder_checkpoint,
                    progress_callback,
                )
            except OSError as error:
                # A read-only weights directory (shared install, container
                # volume) cannot hold the derived file; casting model.pth in
                # memory gives the same model, only with a higher peak of
                # host memory while loading.
                message = (
                    f"Warning: cannot write {HALF_ENCODER_FILENAME} in {weights_dir} ({error}); "
                    "casting model.pth to float16 in memory instead"
                )
                logger.warning(message)
                if progress_callback is not None:
                    progress_callback(message)
        return load_model(
            weights_dir,
            device,
            encoder_dtype=encoder_dtype,
            checkpoint_file=checkpoint_file,
        )

    return load


@dataclass(frozen=True)
class PreparedModel:
    """
    A loaded model ready to transcribe, plus what was decided while preparing it.

    Attributes:
        model: The model on its device.
        configuration: Checkpoint configuration.
        inference: Inference settings to run with (batch budget applied).
        weights_action: What the checkpoint bootstrap did ("ready", "prepared").
        load_seconds: Bootstrap and strict load time.
        runtime: Device and library versions.
        precision: Human-readable precision of encoder and decoder.
        float16_accumulation: State of float16 accumulation ("on", "off ...").
        graph_decoding: Whether CUDA Graph decoding is on.
        memory_budget: The memory ceiling and batch budget decision.
        decisions: One line per setting decided at startup, with its reason.
    """

    model: ParakeetTDT
    configuration: ParakeetConfig
    inference: InferenceSettings
    weights_action: str
    load_seconds: float
    runtime: RuntimeReport
    precision: str
    float16_accumulation: str
    graph_decoding: bool
    memory_budget: MemoryBudget
    decisions: tuple[str, ...] = ()


def prepare_inference_model(
    settings: Settings,
    report: Callable[[str], None],
    notice: Callable[[str], None] | None = None,
) -> PreparedModel:
    """
    Load the model for inference, decide every "auto" setting, and report each decision.

    Order matters: the memory budget is measured on the loaded model before
    the CUDA Graph decoder exists, the rows per batch follow from it, and the
    graph decoder is then built for those rows; its memory is measured and,
    if needed, taken off the rows.

    Args:
        settings: Validated application settings.
        report: Receives one plain-text line per decision (the caller decides
            whether that is the terminal, standard error, or only the log).
        notice: Receives what the user must see even when ``report`` only
            logs: first-run download and conversion progress, and warnings.
            Defaults to ``report``.

    Returns:
        The prepared model, fully resolved settings, and the decisions.

    Raises:
        RuntimeError: If the installed PyTorch build has no kernels for the
            GPU, or not even one full chunk fits in its free memory.
    """

    notice = notice or report
    configured = settings.inference
    device = select_device()
    _require_supported_gpu(device)
    apply_float32_matmul_precision(configured.float32_matmul_precision)
    encoder_dtype, precision = encoder_dtype_for(configured.encoder_precision, device)

    load_started = time.perf_counter()
    bootstrap, (model, configuration, _metadata) = ensure_first_run_ready(
        checkpoint_dir=settings.paths.weights_dir,
        checkpoint_settings=settings.checkpoint,
        loader=inference_model_loader(device, encoder_dtype, progress_callback=notice),
        progress_callback=notice,
    )
    load_seconds = time.perf_counter() - load_started
    report(f"Weights: {bootstrap.action}")

    parameter = next(model.parameters())
    model_device = parameter.device
    runtime = describe_runtime(model_device, parameter.dtype)
    for line in runtime.lines():
        report(line)
    report(f"Encoder precision: {precision}")

    accumulation_enabled, accumulation_reason = _decide_float16_accumulation(
        configured.float16_accumulation,
        model_device,
        encoder_dtype,
        configuration,
    )
    float16_accumulation = apply_float16_accumulation(accumulation_enabled, encoder_dtype)
    if accumulation_reason:
        float16_accumulation = f"{float16_accumulation} ({accumulation_reason})"
    report(f"Float16 accumulation: {float16_accumulation}")

    if configured.decode_workers is None:
        decode_workers = default_decode_workers()
        workers_line = f"Decode workers: {decode_workers} (auto: one per four logical CPUs)"
    else:
        decode_workers = configured.decode_workers
        workers_line = f"Decode workers: {decode_workers}"
    report(f"Model load seconds: {load_seconds:.3f}")

    budgeted, memory_budget = resolve_memory_budget(model, configuration, configured, settings.memory)
    rows = batch_rows_for(configured, memory_budget)
    resolved = _resolved_settings(
        budgeted,
        rows=rows,
        encoder_dtype=encoder_dtype,
        float16_accumulation=accumulation_enabled,
        decode_workers=decode_workers,
    )

    graph_decoding, resolved, graph_warning = _enable_graph_decoding_within_budget(model, resolved, memory_budget)
    report(f"CUDA Graph decoding: {'on' if graph_decoding else 'off'}")
    if graph_warning:
        notice(graph_warning)
    # The report shows the budget the run uses, not the most that would fit.
    memory_budget = dataclasses.replace(memory_budget, batch_feature_frames=resolved.max_batch_feature_frames)

    for line in memory_budget.lines():
        report(line)
    for line in memory_budget.warnings():
        notice(line)
    rows_line = _batch_rows_line(configured, resolved, memory_budget)
    report(rows_line)
    report(workers_line)

    decisions = (
        f"Encoder precision: {precision}",
        f"Float16 accumulation: {float16_accumulation}",
        rows_line,
        workers_line,
        f"CUDA Graph decoding: {'on' if graph_decoding else 'off'}",
    )
    return PreparedModel(
        model=model,
        configuration=configuration,
        inference=resolved,
        weights_action=bootstrap.action,
        load_seconds=load_seconds,
        runtime=runtime,
        precision=precision,
        float16_accumulation=float16_accumulation,
        graph_decoding=graph_decoding,
        memory_budget=memory_budget,
        decisions=decisions,
    )


# =============================================================================
# Decisions
# =============================================================================


def _require_supported_gpu(device: torch.device) -> None:
    """
    Stop before loading when the PyTorch build cannot run on this GPU.

    Otherwise the first kernel launch fails deep inside the model with "no
    kernel image is available", after seconds of loading.
    """

    if build_supports_device(device) is not False:
        return
    capability = cuda_capability(device)
    raise RuntimeError(
        f"The installed PyTorch build has no CUDA kernels for {torch.cuda.get_device_name(device)} "
        f"(compute capability {capability[0]}.{capability[1]}); it was built for "
        f"{', '.join(torch.cuda.get_arch_list())}. Install a PyTorch build that supports this GPU "
        "(see README, Installation), or a CPU build."
    )


def _decide_float16_accumulation(
    setting: bool | None,
    device: torch.device,
    encoder_dtype: torch.dtype,
    configuration: ParakeetConfig,
) -> tuple[bool, str]:
    """
    Whether to accumulate float16 products in float16, and why ("" when configured).

    "auto" measures the encoder's feed-forward products both ways on this GPU
    and turns the switch on only when it is clearly faster.
    """

    if setting is not None:
        return setting, ""
    if encoder_dtype != torch.float16:
        return False, "auto: the encoder runs in float32"
    speedup = measure_float16_accumulation_speedup(
        device,
        configuration.encoder["hidden_size"],
        configuration.encoder["intermediate_size"],
    )
    if speedup is None:
        return False, "auto: not measurable on this device or PyTorch build"
    if speedup >= FLOAT16_ACCUMULATION_MIN_SPEEDUP:
        return True, f"auto: matrix products {speedup:.2f}x faster with it on this GPU"
    return False, f"auto: {speedup:.2f}x, no clear gain on this GPU"


def _resolved_settings(
    budgeted: InferenceSettings,
    *,
    rows: int,
    encoder_dtype: torch.dtype,
    float16_accumulation: bool,
    decode_workers: int,
) -> InferenceSettings:
    """
    The settings with every "auto" replaced by its decision.

    The batch budget never needs to exceed ``rows`` full chunks, because no
    batch has more rows; an "auto" budget that nothing measured becomes
    exactly that.
    """

    full_rows_frames = rows * budgeted.max_chunk_feature_frames
    if budgeted.max_batch_feature_frames is None:
        batch_frames = full_rows_frames
    else:
        batch_frames = min(budgeted.max_batch_feature_frames, full_rows_frames)
    return dataclasses.replace(
        budgeted,
        batch_size=rows,
        max_batch_feature_frames=batch_frames,
        encoder_precision="float16" if encoder_dtype == torch.float16 else "float32",
        float16_accumulation=float16_accumulation,
        decode_workers=decode_workers,
    )


def _enable_graph_decoding_within_budget(
    model: ParakeetTDT,
    resolved: InferenceSettings,
    budget: MemoryBudget,
) -> tuple[bool, InferenceSettings, str]:
    """
    Build the CUDA Graph decoder for the chosen rows, then pay for its memory.

    The decoder's static buffers and capture pool (58 MiB at one row in a
    measured run on the reference GPU) stay allocated for the whole run,
    outside what the budget measured. They are measured here.
    When they leave too little room for the chosen rows, the rows are
    lowered and the decoder rebuilt for them. When not even one full chunk
    would fit next to them, graph decoding is turned off instead: the
    step-by-step loop gives the same output, only slower, and the budget
    was measured with it.

    Returns:
        Whether graph decoding is on, the settings to run with, and a
        warning for the user ("" when there is none).
    """

    device = next(model.parameters()).device
    measured = budget.applied and budget.chunks_that_fit is not None and device.type == "cuda"
    before = torch.accelerator.memory_reserved(device) if measured else 0
    graph_decoding = enable_graph_decoding(model, resolved)
    if not (graph_decoding and measured):
        return graph_decoding, resolved, ""

    reserved = torch.accelerator.memory_reserved(device) - before
    rows = rows_after_reserving(budget, resolved.batch_size, reserved)
    logger.info("graph decoder reserved_mib=%.1f rows=%d of %d", reserved / 2**20, rows, resolved.batch_size)
    if rows == resolved.batch_size:
        return graph_decoding, resolved, ""

    model.graph_decoder = None
    torch.accelerator.empty_cache()
    if rows < 1:
        warning = (
            "Warning: CUDA Graph decoding is off: the free GPU memory holds one chunk but not "
            "the graph decoder next to it. Transcripts are the same, decoding is slower; "
            "close other programs that use the GPU to get it back."
        )
        logger.warning(warning)
        return False, resolved, warning
    lowered = dataclasses.replace(
        resolved,
        batch_size=rows,
        max_batch_feature_frames=min(resolved.max_batch_feature_frames, rows * resolved.max_chunk_feature_frames),
    )
    return enable_graph_decoding(model, lowered), lowered, ""


def _batch_rows_line(configured: InferenceSettings, resolved: InferenceSettings, budget: MemoryBudget) -> str:
    """The rows decision and its reason, for the report."""

    rows = resolved.batch_size
    if configured.batch_size is not None:
        return f"Batch rows: {rows} (configured {configured.batch_size})"
    if budget.chunks_that_fit is None:
        return f"Batch rows: {rows} (auto: memory not measured on this device)"
    return (
        f"Batch rows: {rows} (auto: {AUTO_BATCH_MEMORY_SHARE:.0%} of the {budget.chunks_that_fit} "
        f"full chunks that fit, at most {AUTO_BATCH_ROWS_LIMIT})"
    )
