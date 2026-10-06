"""
Process and accelerator memory: peak host memory and the GPU allocator cap.

Peak process memory is reported next to accelerator memory so a run's host
footprint (decoded audio, model staging, Python objects) is visible too.
psutil is not a dependency, so each platform's own counter is read directly.
"""

from __future__ import annotations

import ctypes
import sys

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


def cap_allocator_to_free_memory(device: torch.device, reserve_bytes: int) -> int:
    """
    Limit PyTorch's CUDA allocator to the memory that is free right now.

    On Windows (WDDM) the driver backs allocations that do not fit in VRAM
    with shared system memory instead of failing, and the run silently
    becomes several times slower. Capping the allocator at the reserved
    memory plus the currently free VRAM, minus ``reserve_bytes`` for the CUDA
    context, library workspaces, and other programs, turns that into an
    out-of-memory error the runtime reports.

    Args:
        device: A CUDA device.
        reserve_bytes: VRAM to leave outside the cap.

    Returns:
        The ceiling in bytes that PyTorch may now reserve on ``device``.

    Raises:
        RuntimeError: If the free memory does not exceed the reserve.
    """

    # Cached but unused blocks count as used in mem_get_info; return them first
    # so "free" is what the driver can really hand out.
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    ceiling_bytes = reserved_bytes + free_bytes - reserve_bytes
    if ceiling_bytes <= reserved_bytes:
        raise RuntimeError(
            f"Only {free_bytes / 2**20:.0f} MiB of accelerator memory is free, which does "
            f"not exceed memory.reserve_mib ({reserve_bytes / 2**20:.0f} MiB). Close other "
            "programs that use the GPU or lower memory.reserve_mib."
        )
    torch.cuda.set_per_process_memory_fraction(ceiling_bytes / total_bytes, device)
    return ceiling_bytes
