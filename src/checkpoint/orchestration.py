"""
Checkpoint acquisition and conversion orchestration.

This module coordinates two independent infrastructure operations:
1. ensure the requested Hugging Face artifacts exist and match per-file evidence;
2. ensure ``model.pth`` corresponds to the current model safetensors and config.

Network transfer is delegated to ``checkpoint.download``. Safetensors parsing and
PyTorch serialization live in ``checkpoint.conversion``. A conversion
manifest records source and output hashes so unchanged checkpoints are not
converted repeatedly.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from ..configuration.config import DEFAULT_WEIGHTS_DIR
from .conversion import convert_checkpoint
from .artifacts import sha256_file
from .download import DownloadResult, ensure_checkpoint_files, load_manifest


CONVERSION_MANIFEST_FILENAME = "conversion_manifest.json"
CONVERSION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ConversionResult:
    """Observable outcome of model.pth freshness validation or conversion."""

    action: str
    output_path: str
    output_size_bytes: int
    output_sha256: str
    source_config_sha256: str
    source_safetensors_sha256: str


@dataclass(frozen=True)
class CheckpointPreparationResult:
    """Complete acquisition and conversion outcome for one checkpoint directory."""

    checkpoint_dir: str
    downloads: tuple[DownloadResult, ...]
    conversion: ConversionResult


# =============================================================================
# Conversion manifest persistence
# =============================================================================
# The download manifest proves each remote artifact independently. Conversion
# freshness needs a narrower dependency set: model.safetensors and config.json
# determine the PyTorch bundle; tokenizer and processor changes do not alter the
# state dictionary bundle and therefore do not trigger a 2.5 GB reconversion.
# =============================================================================


def _load_conversion_manifest(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Load a supported conversion manifest, or return None when absent."""

    path = checkpoint_dir / CONVERSION_MANIFEST_FILENAME
    if not path.is_file():
        return None

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Conversion manifest must be a JSON object: {path}")
    if value.get("schema_version") != CONVERSION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported conversion manifest version: {path}")

    return value


def _write_conversion_manifest(checkpoint_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically persist source and output conversion evidence."""

    path = checkpoint_dir / CONVERSION_MANIFEST_FILENAME
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=checkpoint_dir,
        prefix="conversion_manifest.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            json.dump(manifest, temporary_file, ensure_ascii=True, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    temporary_path.replace(path)


# =============================================================================
# Existing PTH bootstrap validation
# =============================================================================
# Older repositories may already contain model.pth but no conversion manifest.
# Loading only the bundle on CPU is expensive but establishes real compatibility:
# embedded config must equal config.json, state-dict parameter count must equal
# converter metadata, and the expected top-level bundle fields must exist.
# =============================================================================


def _validate_existing_bundle(
    output_path: Path,
    checkpoint_dir: Path,
) -> None:
    """Validate an existing pre-manifest model.pth before trusting it."""

    bundle = torch.load(output_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError(f"Converted checkpoint must be a dictionary: {output_path}")
    if set(bundle) != {"state_dict", "config", "metadata"}:
        raise ValueError(f"Converted checkpoint has unexpected top-level fields: {output_path}")

    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    if bundle["config"] != config:
        raise ValueError("Existing model.pth embeds a config that differs from config.json")

    state_dict = bundle["state_dict"]
    metadata = bundle["metadata"]
    if not isinstance(state_dict, dict) or not isinstance(metadata, dict):
        raise ValueError("Existing model.pth state_dict and metadata must be dictionaries")

    total_elements = sum(tensor.numel() for tensor in state_dict.values())
    if metadata.get("total_parameters") != total_elements:
        raise ValueError("Existing model.pth parameter metadata does not match its state_dict")


# =============================================================================
# Freshness decision and conversion
# =============================================================================


def ensure_converted_checkpoint(
    checkpoint_dir: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> ConversionResult:
    """
    Ensure ``model.pth`` matches the current config and safetensors source.

    Existing output is reused when source hashes and output hash match the
    conversion manifest. A pre-manifest output is validated and bootstrapped.
    Missing, invalid, or stale output is regenerated through the converter.
    """

    checkpoint_dir = checkpoint_dir.resolve()
    output_path = checkpoint_dir / "model.pth"
    download_manifest = load_manifest(checkpoint_dir)
    files = download_manifest["files"]

    config_entry = files.get("config.json")
    safetensors_entry = files.get("model.safetensors")
    if not isinstance(config_entry, dict) or not isinstance(safetensors_entry, dict):
        raise ValueError("Download manifest lacks config.json or model.safetensors evidence")

    config_sha256 = config_entry["sha256"]
    safetensors_sha256 = safetensors_entry["sha256"]
    manifest = _load_conversion_manifest(checkpoint_dir)

    if output_path.is_file() and manifest is not None:
        source_matches = (
            manifest.get("source_config_sha256") == config_sha256
            and manifest.get("source_safetensors_sha256") == safetensors_sha256
        )
        output_sha256 = sha256_file(output_path)
        output_matches = (
            manifest.get("output_size_bytes") == output_path.stat().st_size
            and manifest.get("output_sha256") == output_sha256
        )
        if source_matches and output_matches:
            return ConversionResult(
                action="reused",
                output_path=str(output_path),
                output_size_bytes=output_path.stat().st_size,
                output_sha256=output_sha256,
                source_config_sha256=config_sha256,
                source_safetensors_sha256=safetensors_sha256,
            )

    if output_path.is_file() and manifest is None:
        if progress_callback is not None:
            progress_callback("model.pth: validating existing pre-manifest bundle")
        try:
            _validate_existing_bundle(output_path, checkpoint_dir)
        except (OSError, ValueError, RuntimeError):
            # Invalid legacy output is replaced through the same conversion path
            # used for stale manifest outputs.
            pass
        else:
            output_sha256 = sha256_file(output_path)
            result = ConversionResult(
                action="bootstrapped",
                output_path=str(output_path),
                output_size_bytes=output_path.stat().st_size,
                output_sha256=output_sha256,
                source_config_sha256=config_sha256,
                source_safetensors_sha256=safetensors_sha256,
            )
            _write_conversion_manifest(
                checkpoint_dir,
                {"schema_version": CONVERSION_SCHEMA_VERSION, **asdict(result)},
            )
            return result

    if progress_callback is not None:
        progress_callback("model.pth: converting current safetensors and config")

    # Convert to a same-directory temporary path so a failed conversion cannot
    # corrupt the last valid model.pth. The converter creates the parent directory.
    temporary_output = checkpoint_dir / "model.pth.converting"
    temporary_output.unlink(missing_ok=True)
    try:
        convert_checkpoint(
            checkpoint_dir,
            temporary_output,
            progress_callback=progress_callback,
        )
        temporary_output.replace(output_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    output_sha256 = sha256_file(output_path)
    result = ConversionResult(
        action="converted",
        output_path=str(output_path),
        output_size_bytes=output_path.stat().st_size,
        output_sha256=output_sha256,
        source_config_sha256=config_sha256,
        source_safetensors_sha256=safetensors_sha256,
    )
    _write_conversion_manifest(
        checkpoint_dir,
        {"schema_version": CONVERSION_SCHEMA_VERSION, **asdict(result)},
    )
    return result


def prepare_checkpoint(
    checkpoint_dir: Path = DEFAULT_WEIGHTS_DIR,
    progress_callback: Callable[[str], None] | None = None,
) -> CheckpointPreparationResult:
    """Download/validate all requested artifacts, then ensure model.pth exists."""

    resolved_dir = checkpoint_dir.resolve()
    downloads = ensure_checkpoint_files(
        resolved_dir,
        progress_callback=progress_callback,
    )
    conversion = ensure_converted_checkpoint(
        resolved_dir,
        progress_callback,
    )

    return CheckpointPreparationResult(
        checkpoint_dir=str(resolved_dir),
        downloads=tuple(downloads),
        conversion=conversion,
    )
