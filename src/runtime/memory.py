"""
Peak process memory, measured with the standard library only.

Reported next to accelerator memory so a run's host footprint (decoded audio,
model staging, Python objects) is visible too. psutil is not a dependency, so
each platform's own counter is read directly.
"""

from __future__ import annotations

import ctypes
import sys


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
