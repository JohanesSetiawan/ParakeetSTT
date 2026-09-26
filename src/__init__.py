"""
Standalone PyTorch Parakeet runtime package.

Public consumers should import configuration, audio processing, tokenization,
model, and inference services from this package. Internal module names are
organized by responsibility.
"""

from .audio import (
    DecodedSegment,
    MediaSession,
    ParakeetFeatureExtractor,
    build_mel_filter_bank,
    inspect_media,
    open_media_session,
)
from .checkpoint import (
    BootstrapResult,
    CheckpointPreparationResult,
    ConversionResult,
    clear_readiness_marker,
    ensure_first_run_ready,
    prepare_checkpoint,
    readiness_marker_path,
)
from .configuration import (
    PROJECT_ROOT,
    CheckpointSettings,
    InferenceSettings,
    ParakeetConfig,
    Settings,
    load_config,
    load_settings,
)
from .inference.offline import (
    ChunkResult,
    FileStatus,
    OfflineFileResult,
    OfflineRunResult,
    OfflineTranscriber,
)
from .inference.planning import AudioMetadata, ExecutionPlan, WorkItem, build_execution_plan
from .models import GenerationResult, ParakeetTDT, load_model
from .runtime import describe_runtime, select_device
from .text import BpeTokenizer


__all__ = [
    "AudioMetadata",
    "BootstrapResult",
    "BpeTokenizer",
    "CheckpointPreparationResult",
    "CheckpointSettings",
    "ChunkResult",
    "ConversionResult",
    "DecodedSegment",
    "ExecutionPlan",
    "FileStatus",
    "GenerationResult",
    "InferenceSettings",
    "MediaSession",
    "OfflineFileResult",
    "OfflineRunResult",
    "OfflineTranscriber",
    "PROJECT_ROOT",
    "ParakeetConfig",
    "ParakeetFeatureExtractor",
    "ParakeetTDT",
    "Settings",
    "WorkItem",
    "build_execution_plan",
    "build_mel_filter_bank",
    "clear_readiness_marker",
    "describe_runtime",
    "ensure_first_run_ready",
    "inspect_media",
    "load_config",
    "load_model",
    "load_settings",
    "open_media_session",
    "prepare_checkpoint",
    "readiness_marker_path",
    "select_device",
]
