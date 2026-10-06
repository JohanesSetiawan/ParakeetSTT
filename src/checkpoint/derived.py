"""
Precision-specific checkpoint files derived from the verified ``model.pth``.

With a float16 encoder, loading the float32 ``model.pth`` and casting on the
CPU touched 2.4 GB of mapped weights plus 1.2 GB of new copies: a 4.2 GB
peak working set and 6.1 GB peak commit charge for a 1.2 GB result. Loading a
file that is already float16 measured 2.0 GB and 4.2 GB, and 0.24 s faster.
The file is built once from ``model.pth`` (itself hash-verified at download)
and rebuilt automatically whenever ``model.pth`` changes.

This module stays independent of the model architecture: the caller passes a
function that produces the state dict to store.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable

import torch

from ..runtime.filesystem import write_json_atomic

logger = logging.getLogger(__name__)

HALF_ENCODER_FILENAME = "model.encoder-float16.pth"
HALF_ENCODER_MANIFEST = "model.encoder-float16.json"
# Bump when the stored layout changes, so older derived files are rebuilt.
DERIVED_SCHEMA_VERSION = 1


def _expected_manifest(source: Path) -> dict[str, object]:
    return {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "source": source.name,
        "source_bytes": source.stat().st_size,
    }


def _is_current(path: Path, manifest_path: Path, expected: dict[str, object]) -> bool:
    """The derived file exists, is complete, and was built from this model.pth."""

    if not path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning("derived checkpoint manifest %s is unreadable (%s); rebuilding", manifest_path, error)
        return False
    if not isinstance(manifest, dict):
        return False
    recorded_size = manifest.get("bytes")
    return all(manifest.get(key) == value for key, value in expected.items()) and recorded_size == path.stat().st_size


def _save_atomic(payload: dict[str, object], path: Path) -> None:
    """torch.save to a temporary file in the same directory, then rename over ``path``."""

    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def ensure_half_encoder_checkpoint(
    weights_dir: Path,
    build_state_dict: Callable[[], dict[str, torch.Tensor]],
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """
    Return the float16-encoder checkpoint, building it when missing or stale.

    Args:
        weights_dir: Prepared checkpoint directory containing ``model.pth``.
        build_state_dict: Produces the state dict to store (the model loaded
            from ``model.pth`` with its encoder in float16).
        progress_callback: Optional plain-text progress output.

    Returns:
        Path of the derived file, current with ``model.pth``.
    """

    source = weights_dir / "model.pth"
    path = weights_dir / HALF_ENCODER_FILENAME
    manifest_path = weights_dir / HALF_ENCODER_MANIFEST
    expected = _expected_manifest(source)
    if _is_current(path, manifest_path, expected):
        return path

    if progress_callback is not None:
        progress_callback(f"Preparing {HALF_ENCODER_FILENAME} from model.pth (one time)")
    logger.info("building %s from %s", path, source)
    # The manifest is removed first and written last, so a crash in between
    # leaves no manifest and the next run rebuilds.
    manifest_path.unlink(missing_ok=True)
    _save_atomic({"state_dict": build_state_dict(), "metadata": dict(expected)}, path)
    write_json_atomic(manifest_path, {**expected, "bytes": path.stat().st_size})
    return path
