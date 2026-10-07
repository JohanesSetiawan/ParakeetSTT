"""
Centralized device resolution and runtime environment reporting.

Every component receives the device chosen here instead of probing hardware on
its own, so the model, feature extractor, and reporting always agree.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeReport:
    """Dynamically detected software and hardware facts for one process."""

    device: str
    accelerator: str
    device_count: int
    device_name: str
    precision: str
    torch_version: str
    cuda_version: str
    python_version: str

    def lines(self) -> tuple[str, ...]:
        """Return plain-text ``Label: value`` lines for terminal and log."""

        return (
            f"Device: {self.device}",
            f"Accelerator: {self.accelerator}",
            f"Device count: {self.device_count}",
            f"Device name: {self.device_name}",
            f"Precision: {self.precision}",
            f"PyTorch: {self.torch_version}",
            f"CUDA runtime: {self.cuda_version}",
            f"Python: {self.python_version}",
        )


def select_device() -> torch.device:
    """Select CUDA, then MPS, then CPU from actual PyTorch availability."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encoder_dtype_for(precision: str, device: torch.device) -> tuple[torch.dtype, str]:
    """
    Resolve the configured encoder precision for the selected device.

    Float16 is used on CUDA only: CPUs lack fast float16 kernels and MPS has
    not been verified, so those run float32 and the reason is reported.

    Returns:
        The dtype and a one-line explanation for the terminal and log.
    """

    if precision == "float16" and device.type == "cuda":
        return torch.float16, "float16 encoder, float32 decoder and joint"
    if precision == "float16":
        return torch.float32, f"float32 (float16 is used on CUDA only, this is {device.type})"
    return torch.float32, "float32"


def apply_float32_matmul_precision(precision: str) -> None:
    """
    Set how float32 matrix products run, for the whole process.

    "high" lets GPUs with TF32 tensor cores (NVIDIA Ampere and newer) use
    them: a float32 encoder ran 28 percent faster with the same tokens on the
    sampled batches. With a float16 encoder only the decoder and joint use
    float32 products, and those are bound by reading weights, not arithmetic.
    """

    torch.set_float32_matmul_precision(precision)


def apply_float16_accumulation(enabled: bool, encoder_dtype: torch.dtype) -> str:
    """
    Let cuBLAS accumulate float16 matrix products in float16, for the whole process.

    Tensor cores of consumer GPUs run float16 products about twice as fast
    when they also add in float16. Only the float16 encoder is affected (the
    decoder and joint are float32). On the RTX 3050 Ti the encoder ran 1.17
    times faster, and the word error rate on the docs corpus was 6.66 percent
    against 6.67 with float32 accumulation. The flag is set explicitly either
    way, so a process never inherits it from earlier code.

    Returns:
        A one-line state for the terminal and log.
    """

    matmul_backend = torch.backends.cuda.matmul
    supported = hasattr(matmul_backend, "allow_fp16_accumulation")
    active = enabled and encoder_dtype == torch.float16 and supported
    if supported:
        matmul_backend.allow_fp16_accumulation = active

    if active:
        return "on"
    if not enabled:
        return "off"
    if encoder_dtype != torch.float16:
        return "off (the encoder runs in float32)"
    logger.warning("float16 accumulation requested but this PyTorch build does not support it")
    return "off (not supported by this PyTorch build)"


def describe_runtime(device: torch.device, dtype: torch.dtype) -> RuntimeReport:
    """
    Describe the resolved device and library versions.

    Args:
        device: Device the model parameters live on.
        dtype: Parameter dtype actually used for inference.

    Returns:
        Report whose values are all read from the running process.
    """

    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        accelerator = "CUDA"
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(device_index)
    elif device.type == "mps":
        accelerator = "MPS"
        device_count = 1
        device_name = "Apple Metal"
    else:
        accelerator = "none"
        device_count = 1
        device_name = platform.processor() or platform.machine()

    return RuntimeReport(
        device=str(device),
        accelerator=accelerator,
        device_count=device_count,
        device_name=device_name,
        precision=str(dtype).removeprefix("torch."),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "not available",
        python_version=platform.python_version(),
    )
