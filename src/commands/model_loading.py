"""
Model startup shared by the commands: device, precision, load, and budget.

A float16 encoder is read from the derived ``model.encoder-float16.pth``
(built once from ``model.pth``), which halves the bytes read and the peak
host memory of the load; float32 reads ``model.pth`` directly.
"""

from __future__ import annotations

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
from ..inference.budget import MemoryBudget, resolve_memory_budget
from ..inference.offline import enable_graph_decoding
from ..models.parakeet import ParakeetTDT, load_model
from ..runtime.device import (
    RuntimeReport,
    apply_float16_accumulation,
    apply_float32_matmul_precision,
    describe_runtime,
    encoder_dtype_for,
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


def prepare_inference_model(
    settings: Settings,
    report: Callable[[str], None],
    notice: Callable[[str], None] | None = None,
) -> PreparedModel:
    """
    Load the model for inference and report each decision as it is made.

    Order matters: graph decoding allocates its static buffers before the
    memory budget is measured, so the budget accounts for them.

    Args:
        settings: Validated application settings.
        report: Receives one plain-text line per decision (the caller decides
            whether that is the terminal, standard error, or only the log).
        notice: Receives what the user must see even when ``report`` only
            logs: first-run download and conversion progress, and warnings.
            Defaults to ``report``.

    Returns:
        The prepared model and the decisions behind it.
    """

    notice = notice or report
    device = select_device()
    apply_float32_matmul_precision(settings.inference.float32_matmul_precision)
    encoder_dtype, precision = encoder_dtype_for(settings.inference.encoder_precision, device)

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
    runtime = describe_runtime(parameter.device, parameter.dtype)
    for line in runtime.lines():
        report(line)
    report(f"Encoder precision: {precision}")
    float16_accumulation = apply_float16_accumulation(
        settings.inference.float16_accumulation,
        encoder_dtype,
    )
    report(f"Float16 accumulation: {float16_accumulation}")
    report(f"Model load seconds: {load_seconds:.3f}")

    graph_decoding = enable_graph_decoding(model, settings.inference)
    report(f"CUDA Graph decoding: {'on' if graph_decoding else 'off'}")

    inference, memory_budget = resolve_memory_budget(
        model,
        configuration,
        settings.inference,
        settings.memory,
    )
    for line in memory_budget.lines():
        report(line)
    for line in memory_budget.warnings():
        notice(line)

    return PreparedModel(
        model=model,
        configuration=configuration,
        inference=inference,
        weights_action=bootstrap.action,
        load_seconds=load_seconds,
        runtime=runtime,
        precision=precision,
        float16_accumulation=float16_accumulation,
        graph_decoding=graph_decoding,
        memory_budget=memory_budget,
    )
