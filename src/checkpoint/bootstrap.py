"""
One-time checkpoint bootstrap for the easy-to-use inference command.

The first run validates/downloads source artifacts and converts ``model.pth``
when needed. The caller's loader then strict-loads the model exactly once, and
only after that load succeeds is the readiness marker written atomically. The
loaded model is returned, so the first run does not pay for a second 2.5 GB
checkpoint load just to prove the first one worked.

Later runs take a fast path: the marker must exist and ``model.pth`` must still
have the byte size recorded in it. That check is two ``stat`` calls, so it
stays constant-time while still catching a deleted, truncated, or replaced
checkpoint. Anything deeper (hashing, network metadata) remains the job of a
full preparation, which ``clear_readiness_marker()`` or ``force_repair`` forces.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

from ..configuration.config import CHECKPOINT_FILENAME
from ..configuration.settings import CheckpointSettings
from ..runtime.filesystem import write_json_atomic
from .orchestration import CheckpointPreparationResult, prepare_checkpoint


logger = logging.getLogger(__name__)

READINESS_MARKER_FILENAME = ".ready"
READINESS_SCHEMA_VERSION = 1

LoadedModel = TypeVar("LoadedModel")


@dataclass(frozen=True)
class BootstrapResult:
    """Observable outcome of the one-time readiness gate."""

    action: str
    marker_path: str
    preparation: CheckpointPreparationResult | None


def readiness_marker_path(checkpoint_dir: Path) -> Path:
    """Return the checkpoint-local readiness marker path."""

    return checkpoint_dir.resolve() / READINESS_MARKER_FILENAME


def _write_readiness_marker(checkpoint_dir: Path) -> Path:
    """Atomically write readiness metadata after strict model validation."""

    marker_path = readiness_marker_path(checkpoint_dir)
    write_json_atomic(
        marker_path,
        {
            "schema_version": READINESS_SCHEMA_VERSION,
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "model_pth_size_bytes": (checkpoint_dir / CHECKPOINT_FILENAME).stat().st_size,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return marker_path


def _marker_is_current(marker_path: Path, checkpoint_path: Path) -> bool:
    """
    Return whether the marker still describes the checkpoint on disk.

    A malformed marker counts as stale rather than as an error, because the
    recovery (a full preparation) is the same and needs no user action.
    """

    if not marker_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("readiness marker unreadable, re-preparing: %s", error)
        return False
    if not isinstance(marker, dict) or marker.get("schema_version") != READINESS_SCHEMA_VERSION:
        return False
    return marker.get("model_pth_size_bytes") == checkpoint_path.stat().st_size


def clear_readiness_marker(checkpoint_dir: Path) -> None:
    """Remove readiness state so the next call performs full preparation."""

    readiness_marker_path(checkpoint_dir).unlink(missing_ok=True)


def ensure_first_run_ready(
    checkpoint_dir: Path,
    checkpoint_settings: CheckpointSettings,
    loader: Callable[[Path], LoadedModel],
    force_repair: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[BootstrapResult, LoadedModel]:
    """
    Prepare the checkpoint when needed, then strict-load it exactly once.

    Args:
        checkpoint_dir: Checkpoint directory containing source and PTH artifacts.
        checkpoint_settings: Download/stream policy from config.toml.
        loader: Strict loader called with the resolved directory; its return
            value is passed through. It must raise when the checkpoint does
            not match the model, because the marker is written only after it
            returns.
        force_repair: Remove readiness and rerun preparation even when marked.
        progress_callback: Optional plain-text progress callback.

    Returns:
        ``(result, loaded)`` where ``result.action`` is ``ready`` for the fast
        path or ``prepared`` after full first-run work.
    """

    resolved_dir = checkpoint_dir.resolve()
    marker_path = readiness_marker_path(resolved_dir)

    if force_repair:
        marker_path.unlink(missing_ok=True)

    if _marker_is_current(marker_path, resolved_dir / CHECKPOINT_FILENAME):
        loaded = loader(resolved_dir)
        return BootstrapResult(action="ready", marker_path=str(marker_path), preparation=None), loaded

    if marker_path.is_file():
        logger.warning("readiness marker is stale (model.pth missing or resized); re-preparing")
        marker_path.unlink()

    if progress_callback is not None:
        progress_callback("First run: preparing checkpoint artifacts")
    preparation = prepare_checkpoint(
        checkpoint_dir=resolved_dir,
        checkpoint_settings=checkpoint_settings,
        progress_callback=progress_callback,
    )

    if progress_callback is not None:
        progress_callback("First run: strict-loading model.pth before marking ready")
    loaded = loader(resolved_dir)

    marker_path = _write_readiness_marker(resolved_dir)
    if progress_callback is not None:
        progress_callback(f"Checkpoint ready marker created: {marker_path}")

    result = BootstrapResult(action="prepared", marker_path=str(marker_path), preparation=preparation)
    return result, loaded
