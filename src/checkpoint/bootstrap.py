"""
One-time checkpoint bootstrap for the easy-to-use inference command.

The first run validates/downloads source artifacts, converts ``model.pth`` when
needed, and strict-loads the native model. Only after all stages succeed is a
readiness marker written atomically. Subsequent runs use a constant-time marker
existence check and intentionally skip remote metadata, hashing, download, and
conversion validation.

If the user manually changes or deletes checkpoint files after bootstrap,
``clear_readiness_marker()`` is the internal recovery path. The public inference
command intentionally exposes no checkpoint-management flags and repeated runs
perform no weight checks.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .orchestration import CheckpointPreparationResult, prepare_checkpoint
from ..configuration.config import DEFAULT_WEIGHTS_DIR
from ..models.parakeet import load_model


READINESS_MARKER_FILENAME = ".ready"
READINESS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BootstrapResult:
    """Observable outcome of the one-time readiness gate."""

    action: str
    marker_path: str
    preparation: CheckpointPreparationResult | None


def readiness_marker_path(
    checkpoint_dir: Path = DEFAULT_WEIGHTS_DIR,
) -> Path:
    """Return the checkpoint-local readiness marker path."""

    return checkpoint_dir.resolve() / READINESS_MARKER_FILENAME


def _write_readiness_marker(checkpoint_dir: Path) -> Path:
    """Atomically write readiness metadata after strict model validation."""

    marker_path = readiness_marker_path(checkpoint_dir)
    marker_payload = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "model_pth_size_bytes": (checkpoint_dir / "model.pth").stat().st_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=checkpoint_dir,
        prefix="ready.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            json.dump(
                marker_payload,
                temporary_file,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    temporary_path.replace(marker_path)
    return marker_path


def clear_readiness_marker(
    checkpoint_dir: Path = DEFAULT_WEIGHTS_DIR,
) -> None:
    """Remove readiness state so the next call performs full preparation."""

    readiness_marker_path(checkpoint_dir).unlink(missing_ok=True)


def ensure_first_run_ready(
    checkpoint_dir: Path = DEFAULT_WEIGHTS_DIR,
    force_repair: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> BootstrapResult:
    """
    Prepare checkpoint once, then use marker-only fast paths thereafter.

    Args:
        checkpoint_dir: Checkpoint directory containing source and PTH artifacts.
        force_repair: Remove readiness and rerun preparation even when marked.
        progress_callback: Optional plain-text progress callback.

    Returns:
        ``ready`` for marker fast path or ``prepared`` after full first-run work.
    """

    resolved_dir = checkpoint_dir.resolve()
    marker_path = readiness_marker_path(resolved_dir)

    if force_repair:
        marker_path.unlink(missing_ok=True)

    # This is intentionally the only repeated-run readiness check. Do not add
    # hashing, network metadata, source-file enumeration, or model loading here.
    if marker_path.is_file():
        return BootstrapResult(
            action="ready",
            marker_path=str(marker_path),
            preparation=None,
        )

    if progress_callback is not None:
        progress_callback("First run: preparing checkpoint artifacts")
    preparation = prepare_checkpoint(
        checkpoint_dir=resolved_dir,
        progress_callback=progress_callback,
    )

    if progress_callback is not None:
        progress_callback("First run: strict-loading model.pth before marking ready")

    # Strict-load on CPU to prove the generated bundle matches the native model.
    # The temporary model is released when this function returns; inference then
    # loads once onto the selected runtime device.
    model, _configuration, _metadata = load_model(
        resolved_dir,
        device=None,
    )
    del model

    marker_path = _write_readiness_marker(resolved_dir)
    if progress_callback is not None:
        progress_callback(f"Checkpoint ready marker created: {marker_path}")

    return BootstrapResult(
        action="prepared",
        marker_path=str(marker_path),
        preparation=preparation,
    )
