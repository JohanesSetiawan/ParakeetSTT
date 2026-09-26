"""Checkpoint acquisition, conversion, bootstrap, and artifact boundaries."""

from .bootstrap import (
    BootstrapResult,
    clear_readiness_marker,
    ensure_first_run_ready,
    readiness_marker_path,
)
from .orchestration import (
    CheckpointPreparationResult,
    ConversionResult,
    ensure_converted_checkpoint,
    prepare_checkpoint,
)

__all__ = [
    "BootstrapResult",
    "CheckpointPreparationResult",
    "ConversionResult",
    "clear_readiness_marker",
    "ensure_converted_checkpoint",
    "ensure_first_run_ready",
    "prepare_checkpoint",
    "readiness_marker_path",
]
