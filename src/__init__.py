"""
Standalone PyTorch Parakeet runtime package.

Public consumers should import configuration, processing, tokenization, model,
and inference services from this package. Internal module names are organized by
responsibility and do not use a redundant ``Native`` prefix because the package
itself defines the native runtime boundary.
"""

from .audio import read_wav
from .bootstrap import (
    BootstrapResult,
    clear_readiness_marker,
    ensure_first_run_ready,
    readiness_marker_path,
)
from .checkpoint import (
    CheckpointPreparationResult,
    ConversionResult,
    prepare_checkpoint,
)
from .config import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_WEIGHTS_DIR,
    PROJECT_ROOT,
    ParakeetConfig,
    load_config,
)
from .inference import Transcriber, TranscriptionBatch
from .model import (
    Attention,
    GenerationResult,
    ParakeetTDT,
    load_model,
    select_device,
)
from .processing import ParakeetFeatureExtractor, build_mel_filter_bank
from .settings import InferenceSettings, load_inference_settings
from .tokenization import BpeTokenizer


__all__ = [
    "Attention",
    "BootstrapResult",
    "BpeTokenizer",
    "CheckpointPreparationResult",
    "ConversionResult",
    "DEFAULT_AUDIO_DIR",
    "DEFAULT_WEIGHTS_DIR",
    "GenerationResult",
    "InferenceSettings",
    "PROJECT_ROOT",
    "ParakeetConfig",
    "ParakeetFeatureExtractor",
    "ParakeetTDT",
    "Transcriber",
    "TranscriptionBatch",
    "build_mel_filter_bank",
    "load_config",
    "load_inference_settings",
    "load_model",
    "prepare_checkpoint",
    "ensure_first_run_ready",
    "clear_readiness_marker",
    "readiness_marker_path",
    "read_wav",
    "select_device",
]
