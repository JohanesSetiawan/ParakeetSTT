"""
Process and accelerator memory: peak host memory and the GPU allocator cap.

Peak process memory is reported next to accelerator memory so a run's host
footprint (decoded audio, model staging, Python objects) is visible too.
psutil is not a dependency, so each platform's own counter is read directly.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

import torch


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

    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return AcceleratorMemory(
        reserved=torch.cuda.memory_reserved(device),
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
