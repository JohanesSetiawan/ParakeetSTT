"""
Configuration contracts for the standalone Parakeet TDT runtime.

This module owns all checkpoint configuration loading and validation. Model,
processing, tokenizer, and inference modules consume ``ParakeetConfig`` rather
than reading JSON independently. The checkpoint JSON files remain the source of
truth for architecture dimensions, token IDs, duration classes, and audio
processing parameters.

Dependency boundary
-------------------
This module uses only the Python standard library. It must never import PyTorch,
Transformers, Safetensors, Tokenizers, or an audio package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =============================================================================
# Repository root
# =============================================================================
# The root is derived from this file's physical location instead of the current
# working directory, so config.toml is found no matter where the process starts.
# Every other location (weights, logs) comes from config.toml [paths].
# =============================================================================

# This module lives under src/configuration, so two parents reach src and the
# third reaches the repository root that owns config.toml.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Activation functions this runtime implements. The checkpoint names them in
# config.json; any other value would load weights into the wrong math.
SUPPORTED_ENCODER_ACTIVATION = "silu"
SUPPORTED_JOINT_ACTIVATION = "relu"


@dataclass(frozen=True)
class ParakeetConfig:
    """
    Immutable view of the checkpoint's model and processing configuration.

    The raw dictionaries are preserved because they originate from external
    artifacts and contain fields that may evolve independently. Typed
    properties expose values that are shared across multiple modules and must
    remain consistent, such as vocabulary size and blank token ID.

    Attributes:
        model: Parsed ``config.json`` object.
        processor: Parsed ``processor_config.json`` object.
        generation: Parsed ``generation_config.json`` object.
        weights_dir: Directory that owns the JSON and checkpoint artifacts.
    """

    model: dict[str, Any]
    processor: dict[str, Any]
    generation: dict[str, Any]
    weights_dir: Path

    @property
    def encoder(self) -> dict[str, Any]:
        """Return the nested Fast Conformer encoder configuration."""

        return self.model["encoder_config"]

    @property
    def feature_extractor(self) -> dict[str, Any]:
        """Return the nested audio feature-extractor configuration."""

        return self.processor["feature_extractor"]

    @property
    def vocab_size(self) -> int:
        """Return the number of token classes, including the TDT blank token."""

        return self.model["vocab_size"]

    @property
    def blank_token_id(self) -> int:
        """Return the transducer blank token ID used for frame advancement."""

        return self.model["blank_token_id"]

    @property
    def pad_token_id(self) -> int:
        """Return the sequence padding token ID used after batch completion."""

        return self.model["pad_token_id"]

    @property
    def durations(self) -> tuple[int, ...]:
        """Return ordered TDT duration classes as an immutable tuple."""

        return tuple(self.model["durations"])

    @property
    def tokenizer_path(self) -> Path:
        """Return the tokenizer artifact associated with this configuration."""

        return self.weights_dir / "tokenizer.json"

    @property
    def checkpoint_path(self) -> Path:
        """Return the converted PyTorch checkpoint associated with the config."""

        return self.weights_dir / "model.pth"


# =============================================================================
# JSON loading and validation
# =============================================================================
# Configuration validation happens before model allocation. A malformed model
# relationship should fail while reading small JSON files, not after allocating
# approximately 2.5 GB of parameters on CPU or GPU.
# =============================================================================


def _load_json_object(path: Path) -> dict[str, Any]:
    """
    Load one required JSON artifact and require an object at its root.

    Args:
        path: JSON artifact path.

    Returns:
        Parsed JSON object.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        ValueError: If the JSON root is not an object.
        json.JSONDecodeError: If the artifact contains invalid JSON.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint artifact not found: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(value).__name__}")

    return value


def _require_positive_integer(mapping: dict[str, Any], key: str, context: str) -> int:
    """Read a positive integer and preserve the field name in validation errors."""

    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer, got {value!r}")

    return value


def validate_config(configuration: ParakeetConfig) -> None:
    """
    Validate architecture and processor relationships required by the runtime.

    Validation covers values that affect module construction, tensor layouts,
    and decoding semantics. It intentionally does not invent defaults when the
    checkpoint omits a required field.

    Args:
        configuration: Candidate checkpoint configuration.

    Raises:
        ValueError: If a required field or cross-artifact relationship is
            missing, malformed, or incompatible with this runtime.
    """

    model = configuration.model
    encoder = configuration.encoder
    feature = configuration.feature_extractor
    generation = configuration.generation

    if model.get("model_type") != "parakeet_tdt":
        raise ValueError("config.json model_type must be 'parakeet_tdt'")
    if model.get("architectures") != ["ParakeetForTDT"]:
        raise ValueError("config.json architectures must contain only 'ParakeetForTDT'")
    if configuration.processor.get("processor_class") != "ParakeetProcessor":
        raise ValueError("processor_config.json processor_class must be 'ParakeetProcessor'")

    vocab_size = _require_positive_integer(model, "vocab_size", "model")
    blank_token_id = model.get("blank_token_id")
    pad_token_id = model.get("pad_token_id")
    if not isinstance(blank_token_id, int) or not 0 <= blank_token_id < vocab_size:
        raise ValueError("blank_token_id must reference a valid vocabulary row")
    if not isinstance(pad_token_id, int) or not 0 <= pad_token_id < vocab_size:
        raise ValueError("pad_token_id must reference a valid vocabulary row")

    durations = model.get("durations")
    if not isinstance(durations, list) or not durations:
        raise ValueError("durations must be a non-empty list")
    if not all(isinstance(duration, int) and duration >= 0 for duration in durations):
        raise ValueError("durations must contain non-negative integers")
    if len(set(durations)) != len(durations):
        raise ValueError("durations must not contain duplicate values")
    _require_positive_integer(model, "max_symbols_per_step", "model")

    # These fields select math that the native modules hard-wire. Rejecting
    # other values is safer than loading a checkpoint that would run silently
    # with the wrong activation or input scaling.
    if model.get("hidden_act") != SUPPORTED_JOINT_ACTIVATION:
        raise ValueError(
            f"model.hidden_act must be {SUPPORTED_JOINT_ACTIVATION!r}, "
            f"got {model.get('hidden_act')!r}"
        )
    if encoder.get("hidden_act") != SUPPORTED_ENCODER_ACTIVATION:
        raise ValueError(
            f"encoder.hidden_act must be {SUPPORTED_ENCODER_ACTIVATION!r}, "
            f"got {encoder.get('hidden_act')!r}"
        )
    if encoder.get("scale_input") is not False:
        raise ValueError(
            "encoder.scale_input must be false; input scaling is not implemented, "
            f"got {encoder.get('scale_input')!r}"
        )

    hidden_size = _require_positive_integer(encoder, "hidden_size", "encoder")
    attention_heads = _require_positive_integer(
        encoder,
        "num_attention_heads",
        "encoder",
    )
    key_value_heads = _require_positive_integer(
        encoder,
        "num_key_value_heads",
        "encoder",
    )
    if hidden_size % attention_heads != 0:
        raise ValueError("encoder.hidden_size must be divisible by num_attention_heads")
    if attention_heads % key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    subsampling_factor = _require_positive_integer(
        encoder,
        "subsampling_factor",
        "encoder",
    )
    if subsampling_factor & (subsampling_factor - 1):
        raise ValueError("encoder.subsampling_factor must be a power of two")

    feature_size = _require_positive_integer(feature, "feature_size", "processor")
    mel_bins = _require_positive_integer(encoder, "num_mel_bins", "encoder")
    if feature_size != mel_bins:
        raise ValueError("processor feature_size must equal encoder num_mel_bins")

    _require_positive_integer(feature, "sampling_rate", "processor")
    _require_positive_integer(feature, "n_fft", "processor")
    _require_positive_integer(feature, "hop_length", "processor")
    _require_positive_integer(feature, "win_length", "processor")

    decoder_start_token_id = generation.get("decoder_start_token_id")
    if decoder_start_token_id != blank_token_id:
        raise ValueError("generation decoder_start_token_id must equal blank_token_id")

    expected_suppressed_ids = list(range(vocab_size, vocab_size + len(durations)))
    if generation.get("suppress_tokens") != expected_suppressed_ids:
        raise ValueError(
            "generation suppress_tokens must identify all duration-logit indices"
        )


def load_config(weights_dir: Path) -> ParakeetConfig:
    """
    Load and validate all JSON artifacts needed by native inference.

    Args:
        weights_dir: Directory containing checkpoint JSON files and model.pth.

    Returns:
        Validated immutable configuration whose paths remain tied to the
        supplied checkpoint directory.
    """

    resolved_weights_dir = weights_dir.resolve()
    configuration = ParakeetConfig(
        model=_load_json_object(resolved_weights_dir / "config.json"),
        processor=_load_json_object(resolved_weights_dir / "processor_config.json"),
        generation=_load_json_object(resolved_weights_dir / "generation_config.json"),
        weights_dir=resolved_weights_dir,
    )
    validate_config(configuration)

    return configuration
