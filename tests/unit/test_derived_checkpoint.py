"""The float16-encoder checkpoint derived from model.pth."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.checkpoint import derived as derived_module
from src.checkpoint.derived import (
    HALF_ENCODER_FILENAME,
    HALF_ENCODER_MANIFEST,
    ensure_half_encoder_checkpoint,
    remove_derived_checkpoints,
)
from src.commands.model_loading import inference_model_loader
from src.configuration.config import CHECKPOINT_FILENAME, load_config
from src.models.conformer import ConvolutionModule
from src.models.parakeet import ParakeetTDT, load_model
from support import write_tiny_checkpoint

CPU = torch.device("cpu")
SOURCE_METADATA = {"note": "converted from safetensors", "total_parameters": 1234}


@pytest.fixture
def weights_dir(tmp_path: Path) -> Path:
    """A tiny prepared checkpoint directory with model.pth."""

    directory = write_tiny_checkpoint(tmp_path / "weights")
    torch.manual_seed(0)
    model = ParakeetTDT(load_config(directory)).eval()
    torch.save(
        {"state_dict": model.state_dict(), "config": {}, "metadata": SOURCE_METADATA},
        directory / CHECKPOINT_FILENAME,
    )
    return directory


def counting_builder(weights_dir: Path, calls: list[int]):
    def build():
        calls.append(1)
        # Like the command loader: the derived file keeps the checkpoint layout.
        model, _configuration, metadata = load_model(
            weights_dir,
            CPU,
            encoder_dtype=torch.float16,
            fold_batch_norm=False,
        )
        return model.state_dict(), metadata

    return build


def test_built_once_then_reused(weights_dir: Path) -> None:
    calls: list[int] = []

    first = ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, calls))
    second = ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, calls))

    assert first == second == weights_dir / HALF_ENCODER_FILENAME
    assert calls == [1]
    manifest = json.loads((weights_dir / HALF_ENCODER_MANIFEST).read_text(encoding="utf-8"))
    source = (weights_dir / CHECKPOINT_FILENAME).stat()
    assert manifest["source_bytes"] == source.st_size
    assert manifest["source_mtime_ns"] == source.st_mtime_ns
    assert manifest["bytes"] == first.stat().st_size
    assert not list(weights_dir.glob("*.tmp"))


def test_rebuilt_when_model_pth_changes_size(weights_dir: Path) -> None:
    calls: list[int] = []
    ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, calls))

    bundle = torch.load(weights_dir / CHECKPOINT_FILENAME, weights_only=True)
    bundle["metadata"] = {"note": "re-prepared checkpoint with a different size"}
    torch.save(bundle, weights_dir / CHECKPOINT_FILENAME)
    ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, calls))

    assert calls == [1, 1]


def test_rebuilt_when_model_pth_is_replaced_by_a_file_of_the_same_size(weights_dir: Path) -> None:
    """
    Review finding: freshness was the size alone, and a repaired or replaced
    checkpoint with the same architecture has exactly the same size, so the
    stale float16 weights kept loading.
    """

    calls: list[int] = []
    ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, calls))

    source = weights_dir / CHECKPOINT_FILENAME
    original = source.stat()
    # Different weights, same shapes and therefore same size, written later.
    bundle = torch.load(source, weights_only=True)
    # In place, so the state dict keeps its structure (and its _metadata
    # attribute) and the file keeps its exact size.
    for tensor in bundle["state_dict"].values():
        if tensor.is_floating_point():
            tensor.add_(1.0)
    torch.save(bundle, source)
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns + 10_000_000))
    assert source.stat().st_size == original.st_size

    path = ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, calls))

    assert calls == [1, 1]
    rebuilt = torch.load(path, weights_only=True)["state_dict"]
    expected = load_model(weights_dir, CPU, encoder_dtype=torch.float16, fold_batch_norm=False)[0].state_dict()
    assert all(torch.equal(rebuilt[name], tensor) for name, tensor in expected.items())


def test_reconverting_model_pth_removes_derived_files(weights_dir: Path) -> None:
    ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, []))

    remove_derived_checkpoints(weights_dir)

    assert not (weights_dir / HALF_ENCODER_FILENAME).exists()
    assert not (weights_dir / HALF_ENCODER_MANIFEST).exists()
    remove_derived_checkpoints(weights_dir)  # nothing left to remove is fine


@pytest.mark.parametrize("damage", ["missing manifest", "corrupt manifest", "truncated file"])
def test_incomplete_or_damaged_files_are_rebuilt(weights_dir: Path, damage: str) -> None:
    calls: list[int] = []
    path = ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, calls))
    manifest = weights_dir / HALF_ENCODER_MANIFEST
    if damage == "missing manifest":
        manifest.unlink()
    elif damage == "corrupt manifest":
        manifest.write_text("{not json", encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes()[:100])

    ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, calls))

    assert calls == [1, 1]


def test_derived_file_loads_with_the_same_weights_and_metadata_as_casting(weights_dir: Path) -> None:
    derived = ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, []))

    from_derived, _configuration, derived_metadata = load_model(
        weights_dir,
        CPU,
        encoder_dtype=torch.float16,
        checkpoint_file=derived,
    )
    cast, _configuration, cast_metadata = load_model(weights_dir, CPU, encoder_dtype=torch.float16)

    for name, tensor in cast.state_dict().items():
        assert torch.equal(from_derived.state_dict()[name], tensor), name
    # Review finding: the derived manifest used to replace model.pth's metadata.
    assert derived_metadata == cast_metadata == SOURCE_METADATA
    assert from_derived.encoder_dtype == torch.float16
    assert from_derived.decoder.decoder_projector.weight.dtype == torch.float32
    assert all(
        module.depthwise_conv.weight.dtype == torch.float32
        for module in from_derived.encoder.modules()
        if isinstance(module, ConvolutionModule)
    )
    stored = torch.load(derived, weights_only=True)["state_dict"]
    # Schema 3: the BatchNorm stays float32 so that folding it is exact.
    assert stored["encoder.layers.0.conv.norm.running_var"].dtype == torch.float32
    assert stored["encoder.layers.0.conv.pointwise_conv1.weight"].dtype == torch.float16
    assert from_derived.encoder.encode_positions.inv_freq.dtype == torch.float32


def test_a_float16_encoder_cannot_be_widened_back(weights_dir: Path) -> None:
    derived = ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, []))

    with pytest.raises(ValueError, match="cannot be loaded as torch.float32"):
        load_model(weights_dir, CPU, encoder_dtype=torch.float32, checkpoint_file=derived)


def test_command_loader_uses_the_derived_file_for_float16(weights_dir: Path) -> None:
    model, _configuration, _metadata = inference_model_loader(CPU, torch.float16)(weights_dir)

    assert (weights_dir / HALF_ENCODER_FILENAME).is_file()
    assert model.encoder_dtype == torch.float16

    float32_model, _configuration, _metadata = inference_model_loader(CPU, torch.float32)(weights_dir)
    assert float32_model.encoder_dtype == torch.float32


def test_read_only_weights_fall_back_to_casting_in_memory(weights_dir: Path) -> None:
    """Review finding: a read-only weights directory stopped every float16 run."""

    from src.commands import model_loading

    reports: list[str] = []
    loads: list[Path | None] = []
    real_load_model = model_loading.load_model

    def counting_load_model(*args, **kwargs):
        loads.append(kwargs.get("checkpoint_file"))
        return real_load_model(*args, **kwargs)

    def refuse(*args, **kwargs):
        raise PermissionError(13, "Read-only file system")

    with (
        patch.object(derived_module.tempfile, "mkstemp", side_effect=refuse),
        patch.object(model_loading, "load_model", side_effect=counting_load_model),
    ):
        model, _configuration, metadata = inference_model_loader(CPU, torch.float16, reports.append)(weights_dir)

    assert model.encoder_dtype == torch.float16
    assert metadata == SOURCE_METADATA
    assert not (weights_dir / HALF_ENCODER_FILENAME).exists()
    assert any("cannot write" in line and "in memory" in line for line in reports)
    # Only the in-memory load ran: the unwritable directory was detected before
    # the weights were loaded and cast for a derived file that cannot be saved.
    assert loads == [None]
