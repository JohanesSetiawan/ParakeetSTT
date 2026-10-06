"""
Static-buffer decoding (the step the CUDA Graph captures) against the eager loop.

These run the step eagerly on CPU, so the decoding rules are checked on every
machine; the captured replay itself is checked on the GPU in the full tier.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.audio.features import ParakeetFeatureExtractor
from src.configuration.config import load_config
from src.models.graphed_decoding import GraphedGreedyDecoder
from src.models.parakeet import ParakeetTDT
from support import TINY_BLANK_ID, TINY_VOCAB_SIZE, write_tiny_checkpoint

CPU = torch.device("cpu")


def tiny_model(directory: Path, durations: list[int] | None = None) -> ParakeetTDT:
    torch.manual_seed(0)
    return ParakeetTDT(load_config(write_tiny_checkpoint(directory, durations))).eval()


def features_for(model: ParakeetTDT, sample_counts: tuple[int, ...], seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    extractor = ParakeetFeatureExtractor(model.configuration)
    waveforms = [torch.randn(count, generator=generator) * 0.1 for count in sample_counts]
    return extractor(waveforms, [16000] * len(sample_counts), CPU)


def generate_both(model: ParakeetTDT, features: torch.Tensor, mask: torch.Tensor, rows: int, frames: int):
    eager = model.generate(features, mask)
    model.graph_decoder = GraphedGreedyDecoder(model, rows, frames, use_cuda_graph=False)
    try:
        static = model.generate(features, mask)
    finally:
        model.graph_decoder = None
    return eager, static


def assert_same_generation(eager, static) -> None:
    for field in ("sequences", "durations", "frame_starts", "frame_ends", "forced_advances", "encoder_lengths"):
        assert torch.equal(getattr(eager, field), getattr(static, field)), field


@pytest.mark.parametrize("seed", range(3))
def test_static_step_matches_the_eager_loop_on_mixed_lengths(tmp_path: Path, seed: int) -> None:
    model = tiny_model(tmp_path)
    features, mask = features_for(model, (16000, 7000, 12000), seed=seed)
    frames = int(model.encoder.output_length(torch.tensor([features.shape[1]])).item())

    # Larger static buffers than the batch: extra rows and frames must not
    # change anything.
    eager, static = generate_both(model, features, mask, rows=5, frames=frames + 7)

    assert_same_generation(eager, static)


def test_static_step_matches_with_a_non_identity_duration_table(tmp_path: Path) -> None:
    model = tiny_model(tmp_path, durations=[0, 2, 4])
    features, mask = features_for(model, (16000, 9000))
    frames = int(model.encoder.output_length(torch.tensor([features.shape[1]])).item())

    eager, static = generate_both(model, features, mask, rows=2, frames=frames)

    assert_same_generation(eager, static)


def test_static_step_applies_the_per_frame_guard(tmp_path: Path) -> None:
    """A joint that always emits a token with duration 0 must trip the guard identically."""

    model = tiny_model(tmp_path)
    features, mask = features_for(model, (4000, 8000))
    frames = int(model.encoder.output_length(torch.tensor([features.shape[1]])).item())

    def stuck_joint(decoder_hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.full((decoder_hidden_states.shape[0], 1, TINY_VOCAB_SIZE + 5), -1e4)
        logits[..., TINY_BLANK_ID - 1] = 1e4  # a non-blank token
        logits[..., TINY_VOCAB_SIZE] = 1e4  # duration class 0
        return logits

    with patch.object(model.joint, "forward", stuck_joint):
        eager, static = generate_both(model, features, mask, rows=2, frames=frames)

    assert int(eager.forced_advances.sum()) > 0
    assert_same_generation(eager, static)


def test_non_finite_and_empty_rows_emit_nothing(tmp_path: Path) -> None:
    model = tiny_model(tmp_path)
    features, mask = features_for(model, (16000, 16000))
    original = model.encoder_projector.forward

    def poisoned(hidden_states: torch.Tensor) -> torch.Tensor:
        output = original(hidden_states).clone()
        output[1, 0, 0] = float("nan")
        return output

    frames = int(model.encoder.output_length(torch.tensor([features.shape[1]])).item())
    with patch.object(model.encoder_projector, "forward", poisoned):
        eager, static = generate_both(model, features, mask, rows=4, frames=frames)

    assert eager.encoder_finite.tolist() == [True, False]
    assert_same_generation(eager, static)


def test_batches_beyond_the_static_buffers_use_the_eager_loop(tmp_path: Path) -> None:
    model = tiny_model(tmp_path)
    features, mask = features_for(model, (16000, 16000, 16000))
    decoder = GraphedGreedyDecoder(model, max_batch_rows=2, max_encoder_frames=4, use_cuda_graph=False)
    encoder_states, _ = model.encode(features, mask)

    assert not decoder.accepts(encoder_states)
    with pytest.raises(ValueError, match="static buffers"):
        decoder.decode(encoder_states, torch.ones(3, dtype=torch.long), torch.ones(3, dtype=torch.bool))

    model.graph_decoder = decoder
    try:
        result = model.generate(features, mask)  # falls back to the eager loop
    finally:
        model.graph_decoder = None
    assert result.sequences.shape[0] == 3


def test_graph_decoding_stays_off_without_cuda(tmp_path: Path) -> None:
    model = tiny_model(tmp_path)

    model.enable_graph_decoding(max_batch_rows=4, max_encoder_frames=16)

    assert model.graph_decoder is None
