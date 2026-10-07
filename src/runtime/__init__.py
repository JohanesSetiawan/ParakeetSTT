"""Process-level runtime services: device, logging, and crash-safe files."""

from .device import (
    RuntimeReport,
    apply_float32_matmul_precision,
    describe_runtime,
    encoder_dtype_for,
    select_device,
)
from .filesystem import write_json_atomic, write_text_atomic
from .logging_setup import configure_run_logging, log_file_path
from .memory import (
    AcceleratorMemory,
    cap_allocator,
    measure_accelerator_memory,
    peak_process_memory_bytes,
)

__all__ = [
    "AcceleratorMemory",
    "RuntimeReport",
    "apply_float32_matmul_precision",
    "cap_allocator",
    "configure_run_logging",
    "describe_runtime",
    "encoder_dtype_for",
    "log_file_path",
    "measure_accelerator_memory",
    "peak_process_memory_bytes",
    "select_device",
    "write_json_atomic",
    "write_text_atomic",
]
