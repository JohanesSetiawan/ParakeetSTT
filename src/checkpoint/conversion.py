"""
Safetensors-to-PyTorch checkpoint conversion.

This module converts Hugging Face safetensors artifacts into the repository's
standalone ``model.pth`` bundle. It uses only Python standard-library parsing and
PyTorch. CLI parsing, network download, freshness decisions, and synthetic test
fixtures belong to other layers.

Safetensors byte layout
-----------------------
- bytes 0..7: little-endian uint64 JSON-header length N;
- bytes 8..8+N: UTF-8 JSON tensor descriptors;
- remaining bytes: raw tensor payload addressed by descriptor-relative offsets.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable

import torch

from ..configuration.config import SOURCE_WEIGHTS_FILENAME


SAFETENSORS_PREFIX_BYTES = 8


SAFETENSORS_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


# =============================================================================
# Structural state-dict requirements
# =============================================================================
# These templates are a fast architecture smoke check, not a replacement for the
# native model's strict state-dict load. They catch the most consequential model
# family or layer-count mismatch before serialization.
# =============================================================================

REQUIRED_KEY_TEMPLATES: tuple[str, ...] = (
    "encoder.subsampling.layers.0.weight",
    "encoder.subsampling.linear.weight",
    "encoder.layers.{layer_index}.feed_forward1.linear1.weight",
    "encoder.layers.{layer_index}.feed_forward1.linear2.weight",
    "encoder.layers.{layer_index}.self_attn.q_proj.weight",
    "encoder.layers.{layer_index}.self_attn.k_proj.weight",
    "encoder.layers.{layer_index}.self_attn.v_proj.weight",
    "encoder.layers.{layer_index}.self_attn.o_proj.weight",
    "encoder.layers.{layer_index}.self_attn.relative_k_proj.weight",
    "encoder.layers.{layer_index}.self_attn.bias_u",
    "encoder.layers.{layer_index}.self_attn.bias_v",
    "encoder.layers.{layer_index}.conv.pointwise_conv1.weight",
    "encoder.layers.{layer_index}.conv.depthwise_conv.weight",
    "encoder.layers.{layer_index}.conv.pointwise_conv2.weight",
    "encoder.layers.{layer_index}.feed_forward2.linear1.weight",
    "encoder.layers.{layer_index}.norm_out.weight",
    "encoder_projector.weight",
    "decoder.embedding.weight",
    "decoder.lstm.weight_ih_l0",
    "decoder.lstm.weight_hh_l0",
    "decoder.decoder_projector.weight",
    "joint.head.weight",
)


@dataclass(frozen=True)
class SafetensorsFile:
    """Owned tensors and metadata parsed from one safetensors file."""

    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    """Structural and numerical validation outcome for one loaded state dict."""

    missing_keys: tuple[str, ...] = ()
    nan_tensor_names: tuple[str, ...] = ()
    inf_tensor_names: tuple[str, ...] = ()
    embedding_vocab_mismatch: str | None = None
    joint_head_vocab_mismatch: str | None = None

    @property
    def is_valid(self) -> bool:
        """Return true only when every validation category is empty."""

        return not (
            self.missing_keys
            or self.nan_tensor_names
            or self.inf_tensor_names
            or self.embedding_vocab_mismatch
            or self.joint_head_vocab_mismatch
        )


@dataclass(frozen=True)
class ConvertedCheckpoint:
    """Measured output metadata returned after successful serialization."""

    output_path: str
    tensor_count: int
    total_parameters: int
    parameter_counts_by_group: dict[str, int]


# =============================================================================
# Safetensors parsing
# =============================================================================


def read_safetensors_file(path: Path) -> SafetensorsFile:
    """
    Parse one safetensors file and return tensors with owned CPU storage.

    Each tensor is read straight from its file offset into its own buffer, so
    peak memory is the size of the resulting state dict plus one tensor, not
    the whole file held twice.
    """

    file_size = path.stat().st_size
    if file_size < SAFETENSORS_PREFIX_BYTES:
        raise ValueError(f"Safetensors file is shorter than its header prefix: {path}")

    with path.open("rb") as source:
        prefix = source.read(SAFETENSORS_PREFIX_BYTES)
        header_length = int.from_bytes(prefix, byteorder="little", signed=False)
        header_end = SAFETENSORS_PREFIX_BYTES + header_length
        if header_end > file_size:
            raise ValueError(f"Safetensors JSON header is truncated: {path}")

        header = json.loads(source.read(header_length).decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError(f"Safetensors header must be a JSON object: {path}")

        return _read_tensors(path, source, header, header_end, file_size)


def _read_tensors(
    path: Path,
    source: BinaryIO,
    header: dict[str, Any],
    header_end: int,
    file_size: int,
) -> SafetensorsFile:
    """Validate descriptors and read every tensor payload from ``source``."""

    raw_metadata = header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_metadata.items()
    ):
        raise ValueError(f"Safetensors metadata must be a string map: {path}")

    tensors: dict[str, torch.Tensor] = {}
    for tensor_name, tensor_info in header.items():
        if not isinstance(tensor_info, dict):
            raise ValueError(f"Invalid descriptor for tensor {tensor_name!r}")

        dtype_name = tensor_info.get("dtype")
        shape = tensor_info.get("shape")
        offsets = tensor_info.get("data_offsets")
        if dtype_name not in SAFETENSORS_DTYPE_TO_TORCH:
            raise ValueError(
                f"Unsupported dtype {dtype_name!r} for tensor {tensor_name!r}"
            )
        if not isinstance(shape, list) or not all(
            isinstance(dimension, int) and dimension >= 0
            for dimension in shape
        ):
            raise ValueError(f"Invalid shape for tensor {tensor_name!r}: {shape!r}")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(
            isinstance(offset, int) for offset in offsets
        ):
            raise ValueError(f"Invalid data offsets for tensor {tensor_name!r}")

        start_offset, end_offset = offsets
        absolute_start = header_end + start_offset
        absolute_end = header_end + end_offset
        if start_offset < 0 or end_offset < start_offset or absolute_end > file_size:
            raise ValueError(f"Out-of-range payload for tensor {tensor_name!r}")

        dtype = SAFETENSORS_DTYPE_TO_TORCH[dtype_name]
        expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if absolute_end - absolute_start != expected_bytes:
            raise ValueError(
                f"Payload size of tensor {tensor_name!r} does not match shape {shape} "
                f"and dtype {dtype_name}"
            )

        source.seek(absolute_start)
        # bytearray gives the tensor its own writable storage, so no clone.
        tensor_bytes = bytearray(source.read(absolute_end - absolute_start))
        if len(tensor_bytes) != expected_bytes:
            raise ValueError(f"Unexpected end of file while reading {tensor_name!r} from {path}")

        if expected_bytes == 0:
            tensor = torch.empty(shape, dtype=dtype)
        else:
            tensor = torch.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
        tensors[tensor_name] = tensor

    return SafetensorsFile(tensors=tensors, metadata=dict(raw_metadata))


def _shard_path(checkpoint_dir: Path, shard_name: object) -> Path:
    """
    Resolve one shard file name from an index, rejecting anything but a plain
    file name. The index is external input; ``../x`` or an absolute path must
    not make the converter read outside the checkpoint directory.
    """

    if not isinstance(shard_name, str) or not shard_name or Path(shard_name).name != shard_name:
        raise ValueError(f"Safetensors index names an invalid shard file: {shard_name!r}")
    return checkpoint_dir / shard_name


def resolve_safetensors_files(checkpoint_dir: Path) -> list[Path]:
    """Resolve a single model file or every shard declared by the index JSON."""

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            raise ValueError(f"Safetensors index has no weight_map: {index_path}")
        shard_names = sorted(set(weight_map.values()))
        shard_paths = [_shard_path(checkpoint_dir, shard_name) for shard_name in shard_names]
        missing_shards = [str(path) for path in shard_paths if not path.is_file()]
        if missing_shards:
            raise FileNotFoundError(f"Missing safetensors shards: {missing_shards}")
        return shard_paths

    single_path = checkpoint_dir / SOURCE_WEIGHTS_FILENAME
    if single_path.is_file():
        return [single_path]

    raise FileNotFoundError(
        f"No {SOURCE_WEIGHTS_FILENAME} or model.safetensors.index.json under {checkpoint_dir}"
    )


def load_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load and merge all resolved safetensors files without duplicate keys."""

    state_dict: dict[str, torch.Tensor] = {}
    for shard_path in resolve_safetensors_files(checkpoint_dir):
        shard = read_safetensors_file(shard_path)
        overlapping_keys = sorted(set(shard.tensors) & set(state_dict))
        if overlapping_keys:
            raise ValueError(
                f"Tensor keys appear in multiple shards: {overlapping_keys[:10]}"
            )
        state_dict.update(shard.tensors)
    return state_dict


def load_model_config(checkpoint_dir: Path) -> dict[str, Any]:
    """Load config.json as the plain dictionary embedded in model.pth."""

    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found under {checkpoint_dir}")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"config.json must contain an object: {config_path}")
    return value


# =============================================================================
# State validation and parameter accounting
# =============================================================================


def build_expected_keys(config: dict[str, Any]) -> tuple[str, ...]:
    """Expand required key templates using the configured encoder layer count."""

    encoder_config = config.get("encoder_config")
    if not isinstance(encoder_config, dict):
        raise ValueError("config.encoder_config must be an object")
    layer_count = encoder_config.get("num_hidden_layers")
    if not isinstance(layer_count, int) or layer_count < 0:
        raise ValueError("encoder num_hidden_layers must be a non-negative integer")

    expected: list[str] = []
    for template in REQUIRED_KEY_TEMPLATES:
        if "{layer_index}" in template:
            expected.extend(
                template.format(layer_index=layer_index)
                for layer_index in range(layer_count)
            )
        else:
            expected.append(template)
    return tuple(expected)


def validate_state_dict(
    state_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> ValidationReport:
    """Validate key presence, finite floating values, and output dimensions."""

    missing_keys = tuple(
        key for key in build_expected_keys(config) if key not in state_dict
    )
    nan_names: list[str] = []
    inf_names: list[str] = []

    for tensor_name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            continue
        if bool(torch.isnan(tensor).any()):
            nan_names.append(tensor_name)
        if bool(torch.isinf(tensor).any()):
            inf_names.append(tensor_name)

    vocab_size = config.get("vocab_size")
    embedding_mismatch = None
    embedding = state_dict.get("decoder.embedding.weight")
    if isinstance(vocab_size, int) and embedding is not None:
        if embedding.shape[0] != vocab_size:
            embedding_mismatch = (
                f"decoder.embedding.weight rows={embedding.shape[0]}, "
                f"config vocab_size={vocab_size}"
            )

    joint_mismatch = None
    durations = config.get("durations")
    joint_head = state_dict.get("joint.head.weight")
    if isinstance(vocab_size, int) and isinstance(durations, list) and joint_head is not None:
        expected_outputs = vocab_size + len(durations)
        if joint_head.shape[0] != expected_outputs:
            joint_mismatch = (
                f"joint.head.weight rows={joint_head.shape[0]}, "
                f"expected={expected_outputs}"
            )

    return ValidationReport(
        missing_keys=missing_keys,
        nan_tensor_names=tuple(nan_names),
        inf_tensor_names=tuple(inf_names),
        embedding_vocab_mismatch=embedding_mismatch,
        joint_head_vocab_mismatch=joint_mismatch,
    )


def summarize_parameter_counts(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Group element counts by top-level state-dict prefix."""

    counts: dict[str, int] = {}
    for tensor_name, tensor in state_dict.items():
        group = tensor_name.split(".", maxsplit=1)[0]
        counts[group] = counts.get(group, 0) + tensor.numel()
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


# =============================================================================
# Conversion orchestration
# =============================================================================


def convert_checkpoint(
    checkpoint_dir: Path,
    output_path: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> ConvertedCheckpoint:
    """
    Validate source artifacts and serialize the standalone PyTorch bundle.

    Args:
        checkpoint_dir: Directory containing config and safetensors files.
        output_path: Destination ``.pth`` path. Atomic final replacement is owned
            by the higher-level checkpoint freshness orchestrator.
        progress_callback: Optional plain-text progress callback.

    Returns:
        Measured conversion metadata.
    """

    def progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    progress(f"Loading config from {checkpoint_dir}")
    config = load_model_config(checkpoint_dir)
    progress(f"Loading safetensors weights from {checkpoint_dir}")
    state_dict = load_state_dict(checkpoint_dir)

    tensor_count = len(state_dict)
    total_parameters = sum(tensor.numel() for tensor in state_dict.values())
    progress(f"Loaded {tensor_count} tensors and {total_parameters} elements")

    report = validate_state_dict(state_dict, config)
    if not report.is_valid:
        raise ValueError(
            "Checkpoint validation failed: "
            f"missing={list(report.missing_keys[:5])}, "
            f"nan={list(report.nan_tensor_names[:5])}, "
            f"inf={list(report.inf_tensor_names[:5])}, "
            f"embedding={report.embedding_vocab_mismatch}, "
            f"joint={report.joint_head_vocab_mismatch}"
        )

    parameter_counts = summarize_parameter_counts(state_dict)
    metadata = {
        "source_checkpoint_dir": str(checkpoint_dir),
        "conversion_timestamp": datetime.now().isoformat(),
        "source_transformers_version": config.get("transformers_version"),
        "source_architectures": config.get("architectures"),
        "total_parameters": total_parameters,
        "parameter_counts_by_group": parameter_counts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "config": config,
            "metadata": metadata,
        },
        output_path,
    )
    progress(f"Saved converted checkpoint to {output_path}")

    return ConvertedCheckpoint(
        output_path=str(output_path),
        tensor_count=tensor_count,
        total_parameters=total_parameters,
        parameter_counts_by_group=parameter_counts,
    )
