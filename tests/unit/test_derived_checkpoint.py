"""The float16-encoder checkpoint derived from model.pth."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.checkpoint.derived import HALF_ENCODER_FILENAME, HALF_ENCODER_MANIFEST, ensure_half_encoder_checkpoint
from src.commands.model_loading import inference_model_loader
from src.configuration.config import load_config
from src.models.conformer import ConvolutionModule
from src.models.parakeet import ParakeetTDT, load_model
from support import write_tiny_checkpoint

CPU = torch.device("cpu")


@pytest.fixture
def weights_dir(tmp_path: Path) -> Path:
    """A tiny prepared checkpoint directory with model.pth."""

    directory = write_tiny_checkpoint(tmp_path / "weights")
    torch.manual_seed(0)
    model = ParakeetTDT(load_config(directory)).eval()
    torch.save({"state_dict": model.state_dict(), "config": {}, "metadata": {}}, directory / "model.pth")
    return directory


def counting_builder(weights_dir: Path, calls: list[int]):
    def build() -> dict[str, torch.Tensor]:
        calls.append(1)
        model, _configuration, _metadata = load_model(weights_dir, CPU, encoder_dtype=torch.float16)
        return model.state_dict()

    return build


def test_built_once_then_reused(weights_dir: Path) -> None:
    calls: list[int] = []

    first = ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, calls))
    second = ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, calls))

    assert first == second == weights_dir / HALF_ENCODER_FILENAME
    assert calls == [1]
    manifest = json.loads((weights_dir / HALF_ENCODER_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["source_bytes"] == (weights_dir / "model.pth").stat().st_size
    assert manifest["bytes"] == first.stat().st_size


def test_rebuilt_when_model_pth_changes(weights_dir: Path) -> None:
    calls: list[int] = []
    ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, calls))

    # A re-prepared model.pth of a different size invalidates the derived file.
    bundle = torch.load(weights_dir / "model.pth", weights_only=True)
    bundle["metadata"] = {"note": "re-prepared checkpoint with a different size"}
    torch.save(bundle, weights_dir / "model.pth")
    ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, calls))

    assert calls == [1, 1]


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


def test_derived_file_loads_with_the_same_weights_as_casting(weights_dir: Path) -> None:
    derived = ensure_half_encoder_checkpoint(weights_dir, counting_builder(weights_dir, []))

    from_derived, _configuration, _metadata = load_model(
        weights_dir,
        CPU,
        encoder_dtype=torch.float16,
        checkpoint_file=derived,
    )
    cast, _configuration, _metadata = load_model(weights_dir, CPU, encoder_dtype=torch.float16)

    for name, tensor in cast.state_dict().items():
        assert torch.equal(from_derived.state_dict()[name], tensor), name
    assert from_derived.encoder_dtype == torch.float16
    assert from_derived.decoder.decoder_projector.weight.dtype == torch.float32
    assert all(
        module.depthwise_conv.weight.dtype == torch.float32
        for module in from_derived.encoder.modules()
        if isinstance(module, ConvolutionModule)
    )
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
