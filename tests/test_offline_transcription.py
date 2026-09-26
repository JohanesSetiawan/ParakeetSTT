"""Integration tests for bounded offline orchestration without large weights."""

from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.audio.media import (
    _codec_binary,
    inspect_media,
    open_media_session,
    read_media_segment,
)
from src.configuration.config import DEFAULT_WEIGHTS_DIR, load_config 
from src.configuration.settings import InferenceSettings 
from src.inference.offline import OfflineTranscriber 
from src.models.parakeet import GenerationResult


def _write_float_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Write a minimal mono IEEE-float WAV fixture using standard-library bytes."""

    payload = struct.pack(f"<{waveform.numel()}f", *waveform.tolist())
    fmt = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


class _FakeModel(nn.Module):
    """Return deterministic frame metadata while exercising service wiring."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def generate(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> GenerationResult:
        """Emit one non-special token per row with deterministic timing."""

        batch_size = features.shape[0]
        device = features.device
        return GenerationResult(
            sequences=torch.full((batch_size, 1), 275, dtype=torch.long, device=device),
            durations=torch.ones((batch_size, 1), dtype=torch.long, device=device),
            frame_starts=torch.zeros((batch_size, 1), dtype=torch.long, device=device),
            frame_ends=torch.ones((batch_size, 1), dtype=torch.long, device=device),
            encoder_lengths=mask.sum(dim=1),
        )


class OfflineTranscriptionTests(unittest.TestCase):
    """Verify end-to-end bounded orchestration on synthetic media."""

    def test_media_metadata_and_segment_are_bounded_and_finite(self) -> None:
        """The codec boundary exposes metadata and a finite target-rate segment."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.wav"
            _write_float_wav(path, torch.linspace(-0.2, 0.2, 32000), 16000)

            metadata = inspect_media(path)
            segment = read_media_segment(path, 0, 16000, 16000)

        self.assertEqual(metadata.frame_count, 32000)
        self.assertEqual(tuple(segment.waveform.shape), (16000,))
        self.assertTrue(segment.finite)
        self.assertEqual(segment.source_channels, 1)

    def test_ffmpeg_binary_uses_external_environment_without_hardcoded_path(self) -> None:
        """Fallback executable resolution honors external configuration only."""

        with patch.dict("os.environ", {"FFMPEG_BINARY": "C:/tools/ffmpeg.exe"}):
            with patch("src.audio.media.Path.is_file", return_value=True):
                self.assertEqual(
                    _codec_binary("FFMPEG_BINARY", "ffmpeg"),
                    str(Path("C:/tools/ffmpeg.exe")),
                )

    def test_media_session_reuses_one_file_decoder_for_multiple_segments(self) -> None:
        """A file session reads multiple bounded segments without reopening the path."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.wav"
            _write_float_wav(path, torch.linspace(-0.2, 0.2, 32000), 16000)

            with open_media_session(path) as session:
                first = session.read_segment(0, 8000, 16000)
                second = session.read_segment(8000, 16000, 16000)

        self.assertEqual(first.waveform.numel(), 8000)
        self.assertEqual(second.waveform.numel(), 8000)
        self.assertTrue(first.finite and second.finite)

    def test_offline_service_transcribes_without_reference_or_trace(self) -> None:
        """The complete planner, model boundary, and merge path execute."""

        settings = InferenceSettings(
            batch_size=2,
            recursive=True,
            output_filename="transcriptions.csv",
            audio_extensions=(".wav",),
            max_chunk_feature_frames=100,
            overlap_feature_frames=5,
            max_batch_feature_frames=150,
            max_padding_fraction=0.5,
        )
        model = _FakeModel().eval()
        configuration = load_config(DEFAULT_WEIGHTS_DIR)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "sample.wav"
            _write_float_wav(path, torch.full((32000,), 0.1), 16000)

            transcriber = OfflineTranscriber(model, configuration, settings)
            result = transcriber.transcribe([path])

        self.assertEqual(result.files[0].transcript, "s")
        self.assertGreater(len(result.plan.items), 1)


if __name__ == "__main__":
    unittest.main()