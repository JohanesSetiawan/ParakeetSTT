"""
Standalone PyTorch Parakeet runtime package.

Public consumers should import configuration, processing, tokenization, model,
and inference services from this package. Internal module names are organized by
responsibility and do not use a redundant ``Native`` prefix because the package
itself defines the native runtime boundary.
"""

from .audio import (
    DecodedSegment,
    ParakeetFeatureExtractor,
    build_mel_filter_bank,
    inspect_media,
    read_media_segment,
    read_wav,
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
    DEFAULT_AUDIO_DIR,
    DEFAULT_WEIGHTS_DIR,
    InferenceSettings,
    PROJECT_ROOT,
    ParakeetConfig,
    load_config,
    load_inference_settings,
)
from .inference.offline import (
    AudioMetadata,
    ChunkResult,
    ExecutionPlan,
    OfflineFileResult,
    OfflineRunResult,
    OfflineTranscriber,
    QualityAssessment,
    WorkItem,
)
from .inference.planning import build_execution_plan
from .inference.service import Transcriber, TranscriptionBatch
from .models import (
    Attention,
    GenerationResult,
    ParakeetTDT,
    load_model,
    select_device,
)
from .text import BpeTokenizer


__all__ = [
    "Attention",
    "AudioMetadata",
    "BootstrapResult",
    "BpeTokenizer",
    "ChunkResult",
    "CheckpointPreparationResult",
    "ConversionResult",
    "DEFAULT_AUDIO_DIR",
    "DEFAULT_WEIGHTS_DIR",
    "DecodedSegment",
    "ExecutionPlan",
    "GenerationResult",
    "InferenceSettings",
    "OfflineFileResult",
    "OfflineRunResult",
    "OfflineTranscriber",
    "PROJECT_ROOT",
    "ParakeetConfig",
    "ParakeetFeatureExtractor",
    "ParakeetTDT",
    "QualityAssessment",
    "Transcriber",
    "TranscriptionBatch",
    "WorkItem",
    "build_execution_plan",
    "build_mel_filter_bank",
    "load_config",
    "load_inference_settings",
    "load_model",
    "prepare_checkpoint",
    "ensure_first_run_ready",
    "clear_readiness_marker",
    "readiness_marker_path",
    "read_wav",
    "inspect_media",
    "read_media_segment",
    "select_device",
]
