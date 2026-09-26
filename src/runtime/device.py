"""
Centralized device resolution and runtime environment reporting.

Every component receives the device chosen here instead of probing hardware on
its own, so the model, feature extractor, and reporting always agree.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

import torch


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
