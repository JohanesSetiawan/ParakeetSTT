"""
Runtime diagnostics and source-boundary checks.

The native runtime permits PyTorch and the standard library. Static direct-import
inspection is used instead of inspecting ``sys.modules`` because PyTorch or the
host editor may load transitive modules such as NumPy internally even when the
application source never imports them.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import torch


FORBIDDEN_ECOSYSTEM_IMPORT_PREFIXES = (
    "transformers",
    "huggingface_hub",
    "safetensors",
    "tokenizers",
    "librosa",
    "numpy",
    "soundfile",
)


def find_forbidden_direct_imports(paths: Iterable[Path]) -> list[str]:
    """
    Return forbidden module names imported directly by the supplied sources.

    Args:
        paths: Python source files that define the standalone runtime boundary.

    Returns:
        Sorted unique forbidden direct imports. An empty list means the source
        boundary does not depend directly on the external model ecosystem.
    """

    source_files: list[Path] = []
    for path in paths:
        if path.is_dir():
            source_files.extend(sorted(path.rglob("*.py")))
        else:
            source_files.append(path)

    imported_modules: set[str] = set()
    for source_file in source_files:
        tree = ast.parse(
            source_file.read_text(encoding="utf-8"),
            filename=str(source_file),
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)

    return sorted(
        module_name
        for module_name in imported_modules
        if module_name.startswith(FORBIDDEN_ECOSYSTEM_IMPORT_PREFIXES)
    )


def reset_device_memory_statistics(device: torch.device) -> None:
    """Synchronize and reset CUDA peak counters when the selected device is CUDA."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def collect_device_memory_statistics(device: torch.device) -> dict[str, object]:
    """Return CUDA allocated/reserved memory counters or an unavailable marker."""

    if device.type != "cuda":
        return {
            "available": False,
            "reason": f"CUDA memory counters do not apply to device type {device.type}",
        }

    torch.cuda.synchronize(device)
    return {
        "available": True,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
