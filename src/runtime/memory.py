"""
Process and accelerator memory: peak host memory and the GPU allocator cap.

Peak process memory is reported next to accelerator memory so a run's host
footprint (decoded audio, model staging, Python objects) is visible too.
psutil is not a dependency, so each platform's own counter is read directly.

Accelerator statistics go through ``torch.accelerator``, so they work for
whatever accelerator the run uses. A backend that does not implement a
statistic reports it as unavailable instead of failing the run. Capping the
allocator is CUDA-only (``torch.cuda.set_per_process_memory_fraction``).
"""

from __future__ import annotations

import ctypes
import logging
import sys
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


def peak_process_memory_bytes() -> int | None:
    """
    Return the process's peak resident memory in bytes, or None if unknown.

    Windows reports the peak working set; Linux reports ru_maxrss in KiB and
    macOS in bytes.
    """

    if sys.platform == "win32":
        return _windows_peak_working_set()

    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_peak_working_set() -> int | None:
    from ctypes import wintypes

    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    if not get_info(get_process(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)


# =============================================================================
# Accelerator statistics
# =============================================================================


def reset_peak_accelerator_memory(device: torch.device) -> None:
    """Start a new peak measurement; a no-op on the CPU or an unsupported backend."""

    if device.type == "cpu":
        return
    try:
        torch.accelerator.reset_peak_memory_stats(device)
    except RuntimeError as error:
        logger.debug("peak memory statistics unavailable on %s: %s", device, error)


def accelerator_memory_snapshot(device: torch.device) -> dict[str, Any]:
    """
    Current and peak allocator statistics for a run report.

    Returns:
        ``{"available": True, "allocated_bytes", "reserved_bytes",
        "peak_allocated_bytes", "peak_reserved_bytes"}``, or
        ``{"available": False, "device": ...}`` on the CPU or when the
        backend does not provide the statistics.
    """

    unavailable = {"available": False, "device": str(device)}
    if device.type == "cpu":
        return unavailable
    try:
        return {
            "available": True,
            "allocated_bytes": torch.accelerator.memory_allocated(device),
            "reserved_bytes": torch.accelerator.memory_reserved(device),
            "peak_allocated_bytes": torch.accelerator.max_memory_allocated(device),
            "peak_reserved_bytes": torch.accelerator.max_memory_reserved(device),
        }
    except RuntimeError as error:
        logger.debug("memory statistics unavailable on %s: %s", device, error)
        return unavailable


# =============================================================================
# Accelerator memory ceiling
# =============================================================================


@dataclass(frozen=True)
class AcceleratorMemory:
    """CUDA memory as PyTorch and the driver see it, in bytes."""

    reserved: int
    free: int
    total: int

    def ceiling(self, headroom: int) -> int:
        """What PyTorch may hold while leaving ``headroom`` of today's free memory unused."""

        return self.reserved + max(0, self.free - headroom)


def measure_accelerator_memory(device: torch.device) -> AcceleratorMemory:
    """
    Reserved and free memory on a CUDA device, after releasing PyTorch's cache.

    Cached but unused blocks count as used in ``mem_get_info``, so they are
    returned first and "free" is what the driver can really hand out.
    """

    torch.accelerator.empty_cache()
    free_bytes, total_bytes = torch.accelerator.get_memory_info(device)
    return AcceleratorMemory(
        reserved=torch.accelerator.memory_reserved(device),
        free=free_bytes,
        total=total_bytes,
    )


def cap_allocator(device: torch.device, ceiling_bytes: int, total_bytes: int) -> None:
    """
    Limit PyTorch's CUDA allocator to ``ceiling_bytes``.

    On Windows (WDDM) the driver backs allocations that do not fit in VRAM
    with shared system memory instead of failing, and the run silently
    becomes several times slower. With the allocator capped at memory that is
    physically free, an over-allocation becomes an out-of-memory error the
    runtime reports instead.
    """

    torch.cuda.set_per_process_memory_fraction(ceiling_bytes / total_bytes, device)
