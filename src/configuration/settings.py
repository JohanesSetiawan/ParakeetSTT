"""
Application settings loaded from the repository TOML configuration.

User-controlled behavior (paths, logging, checkpoint transfer policy, and
inference budgets) belongs in ``config.toml`` rather than in module constants.
Every field is required and validated before any model allocation, so a typo
fails in milliseconds with the offending key named instead of silently falling
back to a default.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config.toml"
ENCODER_PRECISIONS = ("float32", "float16")
# torch.set_float32_matmul_precision values: "highest" is exact float32,
# "high" allows TF32 tensor cores on GPUs that have them.
FLOAT32_MATMUL_PRECISIONS = ("highest", "high")


# =============================================================================
# Settings contracts
# =============================================================================


@dataclass(frozen=True)
class PathSettings:
    """Filesystem locations, already resolved to absolute paths."""

    weights_dir: Path
    log_dir: Path
    metrics_dir: Path


@dataclass(frozen=True)
class LoggingSettings:
    """Run-log verbosity."""

    level: str


@dataclass(frozen=True)
class CheckpointSettings:
    """Network and streaming policy for checkpoint acquisition."""

    request_timeout_seconds: float
    download_attempts: int
    retry_backoff_seconds: float
    stream_block_bytes: int


@dataclass(frozen=True)
class InferenceSettings:
    """Validated user-facing settings for file and directory transcription."""

    batch_size: int
    recursive: bool
    output_filename: str
    audio_extensions: tuple[str, ...]
    max_chunk_feature_frames: int
    overlap_feature_frames: int
    max_batch_feature_frames: int
    max_padding_fraction: float
    merge_tolerance_feature_frames: int
    untranscribed_gap_seconds: float
    gap_silence_rms: float
    recovery_start_offsets_feature_frames: tuple[int, ...]
    progress_interval_seconds: float
    max_open_files: int
    encoder_precision: str
    cuda_graphs: bool
    float32_matmul_precision: str
    float16_accumulation: bool
    decode_workers: int


@dataclass(frozen=True)
class MemorySettings:
    """Accelerator memory policy: spill protection and the automatic batch budget."""

    cap_to_free_memory: bool
    auto_batch_budget: bool
    reserve_mib: int


@dataclass(frozen=True)
class BenchmarkSettings:
    """Rounds for the development benchmark command."""

    warmup_rounds: int
    measured_rounds: int
    git_timeout_seconds: float


@dataclass(frozen=True)
class Settings:
    """Complete validated application configuration."""

    paths: PathSettings
    logging: LoggingSettings
    checkpoint: CheckpointSettings
    inference: InferenceSettings
    memory: MemorySettings
    benchmark: BenchmarkSettings


# =============================================================================
# Field validators
# =============================================================================
# Each validator names the full "section.key" in its error so the user can find
# the line in config.toml without reading this module.
# =============================================================================


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    section = document.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"config.toml must contain a [{name}] section")
    return section


def _integer(section: dict[str, Any], section_name: str, key: str, minimum: int) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(
            f"{section_name}.{key} must be an integer >= {minimum}, got {value!r}"
        )
    return value


def _number(
    section: dict[str, Any],
    section_name: str,
    key: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = section.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{section_name}.{key} must be a number, got {value!r}")
    number = float(value)
    if number < minimum or (maximum is not None and number > maximum):
        upper = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{section_name}.{key} must be >= {minimum}{upper}, got {value!r}")
    return number


def _choice(section: dict[str, Any], section_name: str, key: str, choices: tuple[str, ...]) -> str:
    value = section.get(key)
    if value not in choices:
        raise ValueError(f"{section_name}.{key} must be one of {choices}, got {value!r}")
    return value


def _positive_number(section: dict[str, Any], section_name: str, key: str) -> float:
    number = _number(section, section_name, key, minimum=0.0)
    if number == 0.0:
        raise ValueError(f"{section_name}.{key} must be greater than zero")
    return number


def _boolean(section: dict[str, Any], section_name: str, key: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{section_name}.{key} must be a boolean, got {value!r}")
    return value


def _path(section: dict[str, Any], section_name: str, key: str, root: Path) -> Path:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section_name}.{key} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


# =============================================================================
# Section parsers
# =============================================================================


def _parse_paths(document: dict[str, Any], root: Path) -> PathSettings:
    section = _section(document, "paths")
    return PathSettings(
        weights_dir=_path(section, "paths", "weights_dir", root),
        log_dir=_path(section, "paths", "log_dir", root),
        metrics_dir=_path(section, "paths", "metrics_dir", root),
    )


def _parse_logging(document: dict[str, Any]) -> LoggingSettings:
    section = _section(document, "logging")
    level = section.get("level")
    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR")
    if not isinstance(level, str) or level.upper() not in valid_levels:
        raise ValueError(f"logging.level must be one of {valid_levels}, got {level!r}")
    return LoggingSettings(level=level.upper())


def _parse_checkpoint(document: dict[str, Any]) -> CheckpointSettings:
    section = _section(document, "checkpoint")
    timeout = _number(section, "checkpoint", "request_timeout_seconds", minimum=0.0)
    if timeout == 0.0:
        raise ValueError("checkpoint.request_timeout_seconds must be greater than zero")
    return CheckpointSettings(
        request_timeout_seconds=timeout,
        download_attempts=_integer(section, "checkpoint", "download_attempts", minimum=1),
        retry_backoff_seconds=_number(section, "checkpoint", "retry_backoff_seconds", minimum=0.0),
        stream_block_bytes=_integer(section, "checkpoint", "stream_block_bytes", minimum=1),
    )


def _parse_extensions(section: dict[str, Any]) -> tuple[str, ...]:
    raw_extensions = section.get("audio_extensions")
    if not isinstance(raw_extensions, list):
        raise ValueError("inference.audio_extensions must be a list")

    normalized_extensions: list[str] = []
    for extension in raw_extensions:
        if not isinstance(extension, str) or not extension.strip():
            raise ValueError("Every inference.audio_extensions entry must be a non-empty string")
        normalized = extension.strip().lower()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        if normalized not in normalized_extensions:
            normalized_extensions.append(normalized)
    return tuple(normalized_extensions)


def _parse_inference(document: dict[str, Any]) -> InferenceSettings:
    section = _section(document, "inference")

    output_filename = section.get("output_filename")
    if (
        not isinstance(output_filename, str)
        or not output_filename.strip()
        or Path(output_filename).name != output_filename
    ):
        raise ValueError("inference.output_filename must be a plain filename")

    max_chunk_feature_frames = _integer(section, "inference", "max_chunk_feature_frames", minimum=1)
    # Zero overlap is a legitimate benchmark setting: chunks then share no
    # context and ownership boundaries coincide with chunk boundaries.
    overlap_feature_frames = _integer(section, "inference", "overlap_feature_frames", minimum=0)
    if max_chunk_feature_frames - 2 * overlap_feature_frames <= 0:
        raise ValueError(
            "inference.max_chunk_feature_frames must exceed twice "
            "inference.overlap_feature_frames so every chunk owns some frames"
        )

    max_batch_feature_frames = _integer(section, "inference", "max_batch_feature_frames", minimum=1)
    if max_batch_feature_frames < max_chunk_feature_frames:
        raise ValueError(
            "inference.max_batch_feature_frames must be at least "
            "inference.max_chunk_feature_frames or a full chunk can never be scheduled"
        )

    progress_interval = _number(section, "inference", "progress_interval_seconds", minimum=0.0)

    untranscribed_gap_seconds = _number(section, "inference", "untranscribed_gap_seconds", minimum=0.0)
    if untranscribed_gap_seconds == 0.0:
        raise ValueError("inference.untranscribed_gap_seconds must be greater than zero")
    raw_offsets = section.get("recovery_start_offsets_feature_frames")
    if not isinstance(raw_offsets, list) or not all(
        isinstance(offset, int) and not isinstance(offset, bool) and offset != 0
        for offset in raw_offsets
    ):
        raise ValueError(
            "inference.recovery_start_offsets_feature_frames must be a list of non-zero integers"
        )
    if any(abs(offset) > overlap_feature_frames for offset in raw_offsets):
        raise ValueError(
            "inference.recovery_start_offsets_feature_frames must not exceed "
            "inference.overlap_feature_frames in size, or a shifted window would "
            "leave part of the chunk core undecoded"
        )

    return InferenceSettings(
        batch_size=_integer(section, "inference", "batch_size", minimum=1),
        recursive=_boolean(section, "inference", "recursive"),
        output_filename=output_filename,
        audio_extensions=_parse_extensions(section),
        max_chunk_feature_frames=max_chunk_feature_frames,
        overlap_feature_frames=overlap_feature_frames,
        max_batch_feature_frames=max_batch_feature_frames,
        max_padding_fraction=_number(
            section,
            "inference",
            "max_padding_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        merge_tolerance_feature_frames=_integer(
            section,
            "inference",
            "merge_tolerance_feature_frames",
            minimum=0,
        ),
        untranscribed_gap_seconds=untranscribed_gap_seconds,
        gap_silence_rms=_number(section, "inference", "gap_silence_rms", minimum=0.0),
        recovery_start_offsets_feature_frames=tuple(raw_offsets),
        progress_interval_seconds=progress_interval,
        max_open_files=_integer(section, "inference", "max_open_files", minimum=1),
        encoder_precision=_choice(section, "inference", "encoder_precision", ENCODER_PRECISIONS),
        cuda_graphs=_boolean(section, "inference", "cuda_graphs"),
        float32_matmul_precision=_choice(
            section,
            "inference",
            "float32_matmul_precision",
            FLOAT32_MATMUL_PRECISIONS,
        ),
        float16_accumulation=_boolean(section, "inference", "float16_accumulation"),
        decode_workers=_integer(section, "inference", "decode_workers", minimum=1),
    )


def _parse_memory(document: dict[str, Any]) -> MemorySettings:
    section = _section(document, "memory")
    return MemorySettings(
        cap_to_free_memory=_boolean(section, "memory", "cap_to_free_memory"),
        auto_batch_budget=_boolean(section, "memory", "auto_batch_budget"),
        reserve_mib=_integer(section, "memory", "reserve_mib", minimum=0),
    )


def _parse_benchmark(document: dict[str, Any]) -> BenchmarkSettings:
    section = _section(document, "benchmark")
    return BenchmarkSettings(
        warmup_rounds=_integer(section, "benchmark", "warmup_rounds", minimum=0),
        measured_rounds=_integer(section, "benchmark", "measured_rounds", minimum=1),
        git_timeout_seconds=_positive_number(section, "benchmark", "git_timeout_seconds"),
    )


# =============================================================================
# Public loader
# =============================================================================


def load_settings(
    settings_path: Path = DEFAULT_SETTINGS_PATH,
    project_root: Path = PROJECT_ROOT,
) -> Settings:
    """
    Load and validate every section of ``config.toml``.

    Args:
        settings_path: TOML file to read.
        project_root: Base directory for relative paths in ``[paths]``.

    Returns:
        Immutable validated settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If any required field is missing, mistyped, or conflicts
            with another field.
    """

    if not settings_path.is_file():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    with settings_path.open("rb") as input_file:
        document = tomllib.load(input_file)

    return Settings(
        paths=_parse_paths(document, project_root),
        logging=_parse_logging(document),
        checkpoint=_parse_checkpoint(document),
        inference=_parse_inference(document),
        memory=_parse_memory(document),
        benchmark=_parse_benchmark(document),
    )
