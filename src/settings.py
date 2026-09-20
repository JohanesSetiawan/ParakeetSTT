"""
Application settings loaded from the repository TOML configuration.

User-controlled inference behavior belongs in ``config.toml`` rather than model
or CLI implementation. This module validates the configuration before model
allocation so invalid batch sizes or output names fail quickly.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT


DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config.toml"


@dataclass(frozen=True)
class InferenceSettings:
    """Validated user-facing settings for file and directory transcription."""

    batch_size: int
    recursive: bool
    output_filename: str
    audio_extensions: tuple[str, ...]


def load_inference_settings(
    settings_path: Path = DEFAULT_SETTINGS_PATH,
) -> InferenceSettings:
    """
    Load and validate the ``[inference]`` TOML section.

    Args:
        settings_path: TOML file containing the inference section.

    Returns:
        Immutable validated inference settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If a required field has an invalid type or value.
    """

    if not settings_path.is_file():
        raise FileNotFoundError(f"Inference settings not found: {settings_path}")

    with settings_path.open("rb") as input_file:
        document = tomllib.load(input_file)

    raw_inference = document.get("inference")
    if not isinstance(raw_inference, dict):
        raise ValueError("config.toml must contain an [inference] section")

    batch_size = raw_inference.get("batch_size")
    recursive = raw_inference.get("recursive")
    output_filename = raw_inference.get("output_filename")
    audio_extensions = raw_inference.get("audio_extensions")

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("inference.batch_size must be a positive integer")
    if not isinstance(recursive, bool):
        raise ValueError("inference.recursive must be a boolean")
    if (
        not isinstance(output_filename, str)
        or not output_filename.strip()
        or Path(output_filename).name != output_filename
    ):
        raise ValueError("inference.output_filename must be a plain filename")
    if not isinstance(audio_extensions, list) or not audio_extensions:
        raise ValueError("inference.audio_extensions must be a non-empty list")

    normalized_extensions: list[str] = []
    for extension in audio_extensions:
        if not isinstance(extension, str) or not extension.strip():
            raise ValueError("Every inference audio extension must be a string")
        normalized = extension.lower()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        if normalized not in normalized_extensions:
            normalized_extensions.append(normalized)

    return InferenceSettings(
        batch_size=batch_size,
        recursive=recursive,
        output_filename=output_filename,
        audio_extensions=tuple(normalized_extensions),
    )
