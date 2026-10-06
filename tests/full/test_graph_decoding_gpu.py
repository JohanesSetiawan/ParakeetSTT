"""The captured CUDA Graph decoder on the real model: bit-identical to the eager loop."""

from __future__ import annotations

import pytest
import soundfile
import torch

from src.audio.features import ParakeetFeatureExtractor
from src.models.graphed_decoding import GraphedGreedyDecoder


@pytest.mark.cuda
def test_captured_graph_decoding_is_bit_identical_to_the_eager_loop(real_settings, real_model, speech_clips) -> None:
    model, configuration = real_model
    device = next(model.parameters()).device
    extractor = ParakeetFeatureExtractor(configuration)
    # Real speech of different lengths in one padded batch; the longest clip
    # is cut to one chunk so it fits the graph's static buffers.
    max_samples = (real_settings.inference.max_chunk_feature_frames - 1) * 160
    waveforms = []
    for clip in speech_clips:
        audio, rate = soundfile.read(str(clip.path), dtype="float32")
        assert rate == 16000
        waveforms.append(torch.from_numpy(audio[:max_samples]))
    features, mask = extractor(waveforms, [16000] * len(waveforms), device)
    frames = int(model.encoder.output_length(torch.tensor([real_settings.inference.max_chunk_feature_frames])).item())

    previous = model.graph_decoder
    try:
        model.graph_decoder = None
        eager = model.generate(features, mask)
        graph_decoder = GraphedGreedyDecoder(model, real_settings.inference.batch_size, frames)
        model.graph_decoder = graph_decoder
        first = model.generate(features, mask)
        # A second, smaller batch replays the same graph.
        smaller = model.generate(features[:2], mask[:2])
        model.graph_decoder = None
        smaller_eager = model.generate(features[:2], mask[:2])
    finally:
        model.graph_decoder = previous

    assert graph_decoder.graph is not None
    for field in ("sequences", "durations", "frame_starts", "frame_ends", "forced_advances"):
        assert torch.equal(getattr(eager, field), getattr(first, field)), field
        assert torch.equal(getattr(smaller_eager, field), getattr(smaller, field)), field
