"""
Precision-specific checkpoint files derived from the verified ``model.pth``.

With a float16 encoder, loading the float32 ``model.pth`` and casting on the
CPU touched 2.4 GB of mapped weights plus 1.2 GB of new copies: a 4.2 GB
peak working set and 6.1 GB peak commit charge for a 1.2 GB result. Loading a
file that is already float16 measured 2.0 GB and 4.2 GB, and 0.24 s faster.

The file is built once from ``model.pth`` (itself hash-verified at download)
and is current only while ``model.pth`` is the same file it was built from:
same size and same modification time in nanoseconds. Re-converting
``model.pth`` also deletes it. Hashing 2.4 GB on every start would cost
seconds; any replacement of ``model.pth`` (conversion, repair, a copied-in
checkpoint) writes a new file and therefore a new modification time.

This module stays independent of the model architecture: the caller passes a
function that produces the state dict and the source metadata to store.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch

from ..configuration.config import CHECKPOINT_FILENAME
from ..runtime.filesystem import write_json_atomic

logger = logging.getLogger(__name__)

HALF_ENCODER_FILENAME = "model.encoder-float16.pth"
HALF_ENCODER_MANIFEST = "model.encoder-float16.json"
# Bump when the stored layout or the freshness rule changes, so older derived
# files are rebuilt. Version 2 added the source modification time.
DERIVED_SCHEMA_VERSION = 2

StateAndMetadata = tuple[dict[str, torch.Tensor], dict[str, Any]]


def _source_identity(source: Path) -> dict[str, object]:
    """What a derived file must have been built from to be current."""

    status = source.stat()
    return {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "source": source.name,
        "source_bytes": status.st_size,
        "source_mtime_ns": status.st_mtime_ns,
    }


def _is_current(path: Path, manifest_path: Path, expected: dict[str, object]) -> bool:
    """The derived file exists, is complete, and was built from this exact source file."""

    if not path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning("derived checkpoint manifest %s is unreadable (%s); rebuilding", manifest_path, error)
        return False
    if not isinstance(manifest, dict):
        return False
    same_source = all(manifest.get(key) == value for key, value in expected.items())
    return same_source and manifest.get("bytes") == path.stat().st_size


def remove_derived_checkpoints(weights_dir: Path) -> None:
    """Delete every file derived from ``model.pth``; called when it is replaced."""

    for name in (HALF_ENCODER_MANIFEST, HALF_ENCODER_FILENAME):
        (weights_dir / name).unlink(missing_ok=True)


def ensure_half_encoder_checkpoint(
    weights_dir: Path,
    build: Callable[[], StateAndMetadata],
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """
    Return the float16-encoder checkpoint, building it when missing or stale.

    Args:
        weights_dir: Prepared checkpoint directory containing ``model.pth``.
        build: Produces the state dict to store (the model loaded from
            ``model.pth`` with its encoder in float16) and ``model.pth``'s own
            metadata, which the derived file keeps.
        progress_callback: Optional plain-text progress output.

    Returns:
        Path of the derived file, current with ``model.pth``.

    Raises:
        OSError: If the weights directory cannot be written (callers may then
            cast in memory instead).
    """

    source = weights_dir / CHECKPOINT_FILENAME
    path = weights_dir / HALF_ENCODER_FILENAME
    manifest_path = weights_dir / HALF_ENCODER_MANIFEST
    expected = _source_identity(source)
    if _is_current(path, manifest_path, expected):
        return path

    # The temporary file comes first: a read-only directory fails here, in
    # milliseconds, instead of after loading and casting 2.4 GB of weights.
    descriptor, temporary_name = tempfile.mkstemp(dir=weights_dir, prefix=f"{path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        if progress_callback is not None:
            progress_callback(f"Preparing {HALF_ENCODER_FILENAME} from {source.name} (one time)")
        logger.info("building %s from %s", path, source)
        # The manifest is removed first and written last, so a crash in
        # between leaves no manifest and the next run rebuilds.
        manifest_path.unlink(missing_ok=True)
        state_dict, source_metadata = build()
        torch.save(
            {"state_dict": state_dict, "metadata": source_metadata, "derived_from": expected},
            temporary_path,
        )
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    write_json_atomic(manifest_path, {**expected, "bytes": path.stat().st_size})
    return path
