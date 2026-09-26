"""Integration tests for bounded media decoding and offline orchestration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile
import torch
from torch import nn

from src.audio.media import _codec_binary, inspect_media, open_media_session
from src.configuration.config import load_config
from src.inference.offline import ChunkResult, FileStatus, OfflineTranscriber, classify_file
from src.inference.planning import WorkItem, build_execution_plan
from src.models.parakeet import GenerationResult
from support import inference_settings, write_float_wav, write_tiny_checkpoint

TARGET_RATE = 16000
HOP_LENGTH = 160


def _read_plan_sequentially(path: Path, max_chunk: int, overlap: int):
    """Decode every chunk of one file both sequentially and fresh."""

    plan = build_execution_plan(
        [inspect_media(path)],
        target_sample_rate=TARGET_RATE,
        hop_length=HOP_LENGTH,
        max_chunk_feature_frames=max_chunk,
        overlap_feature_frames=overlap,
        batch_size=4,
        max_batch_feature_frames=4 * max_chunk,
        max_padding_fraction=1.0,
    )
    pairs = []
    with open_media_session(path) as sequential, open_media_session(path) as fresh:
        previous = None
        for item in sorted(plan.items, key=lambda work: work.chunk_index):
            segment, waveform = sequential.read_sequential_segment(
                item.source_start_frame,
                item.source_end_frame,
                TARGET_RATE,
                previous_source_end_frame=previous[0] if previous else None,
                previous_waveform=previous[1] if previous else None,
            )
            previous = (item.source_end_frame, waveform)
            reference = fresh.read_segment(item.source_start_frame, item.source_end_frame, TARGET_RATE)
            pairs.append((item, segment, reference))
    return pairs


class MediaDecodingTests(unittest.TestCase):
    """The codec boundary is bounded, exact at seams, and honest about input."""

    def test_session_reads_bounded_finite_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.wav"
            write_float_wav(path, torch.linspace(-0.2, 0.2, 32000), TARGET_RATE)

            metadata = inspect_media(path)
            with open_media_session(path) as session:
                first = session.read_segment(0, 8000, TARGET_RATE)
                second = session.read_segment(8000, 16000, TARGET_RATE)

        self.assertEqual(metadata.frame_count, 32000)
        self.assertEqual(first.waveform.numel(), 8000)
        self.assertEqual(second.waveform.numel(), 8000)
        self.assertTrue(first.finite and second.finite)

    def test_zero_overlap_does_not_accumulate_previous_chunks(self) -> None:
        """
        Regression: with zero overlap, tensor[-0:] used to prepend the entire
        previous chunk, so chunk k grew to k times the budget.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "long.wav"
            waveform = torch.sin(torch.arange(120_000, dtype=torch.float32) / 7.0)
            write_float_wav(path, waveform, TARGET_RATE)

            pairs = _read_plan_sequentially(path, max_chunk=100, overlap=0)

        self.assertGreater(len(pairs), 3)
        for item, segment, reference in pairs:
            self.assertEqual(segment.waveform.numel(), item.source_frame_count)
            self.assertTrue(torch.equal(segment.waveform, reference.waveform))

    def test_overlap_longer_than_core_still_matches_fresh_decode(self) -> None:
        """Chunk 21 frames with overlap 10 leaves a 1-frame core; still exact."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "short_cores.wav"
            write_float_wav(path, torch.linspace(-0.3, 0.3, 20_000), TARGET_RATE)

            pairs = _read_plan_sequentially(path, max_chunk=21, overlap=10)

        self.assertGreater(len(pairs), 10)
        for item, segment, reference in pairs:
            self.assertTrue(torch.equal(segment.waveform, reference.waveform))

    def test_overlap_reuse_is_bit_identical_after_resampling(self) -> None:
        """48 kHz stereo input: reused overlap equals a fresh decode exactly."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "stereo48k.wav"
            generator = np.random.default_rng(7)
            # 400_001 frames is not divisible by 3, so the last chunk ends
            # inside a source sample: the end-of-file read is one frame short.
            samples = generator.uniform(-0.5, 0.5, size=(400_001, 2)).astype(np.float32)
            soundfile.write(str(path), samples, 48000, subtype="FLOAT")

            pairs = _read_plan_sequentially(path, max_chunk=120, overlap=10)

        self.assertGreater(len(pairs), 3)
        for item, segment, reference in pairs:
            with self.subTest(chunk=item.chunk_index):
                self.assertEqual(segment.waveform.numel(), item.source_frame_count)
                self.assertTrue(torch.equal(segment.waveform, reference.waveform))

        # 48 kHz -> 16 kHz linear resampling with align_corners=False samples
        # source index 3i + 1. The final chunk must follow that grid (padded
        # with silence past end-of-file), not be stretched.
        last_item, last_segment, _ = pairs[-1]
        mono = torch.from_numpy(samples.mean(axis=1))
        padded = torch.nn.functional.pad(mono, (0, 3))
        expected = padded[3 * last_item.source_start_frame + 1 :: 3][: last_item.source_frame_count]
        self.assertTrue(torch.allclose(last_segment.waveform, expected, atol=1e-6))

    def test_non_finite_samples_are_zeroed_and_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nan.wav"
            waveform = torch.full((1600,), 0.1)
            waveform[10:20] = float("nan")
            write_float_wav(path, waveform, TARGET_RATE)

            with open_media_session(path) as session:
                segment = session.read_segment(0, 1600, TARGET_RATE)

        self.assertFalse(segment.finite)
        self.assertTrue(torch.isfinite(segment.waveform).all())

    def test_ffprobe_metadata_uses_duration_not_packet_count(self) -> None:
        """
        Regression: nb_frames counts AAC packets (217 for 13.82 s), and was
        read as a sample count, shrinking the file to 0.014 s.
        """

        probe = {
            "stream": {"sample_rate": "16000", "channels": 1, "duration": "13.820000", "nb_frames": "217"},
            "format": {"duration": "13.820000"},
        }
        with (
            patch("src.audio.media.soundfile.info", side_effect=RuntimeError("unsupported")),
            patch("src.audio.media._run_ffprobe", return_value=probe),
        ):
            metadata = inspect_media(Path("clip.m4a"))

        self.assertEqual(metadata.frame_count, 221_120)
        self.assertAlmostEqual(metadata.duration_seconds, 13.82)

    def test_ffprobe_falls_back_to_container_duration(self) -> None:
        probe = {
            "stream": {"sample_rate": "48000", "channels": 2, "duration": "N/A"},
            "format": {"duration": "2.5"},
        }
        with (
            patch("src.audio.media.soundfile.info", side_effect=RuntimeError("unsupported")),
            patch("src.audio.media._run_ffprobe", return_value=probe),
        ):
            metadata = inspect_media(Path("clip.webm"))

        self.assertEqual(metadata.frame_count, 120_000)

    def test_unreadable_media_reports_both_backends(self) -> None:
        with (
            patch("src.audio.media.soundfile.info", side_effect=RuntimeError("libsndfile says no")),
            patch("src.audio.media._run_ffprobe", side_effect=RuntimeError("ffprobe missing")),
        ):
            with self.assertRaisesRegex(ValueError, "libsndfile says no.*ffprobe missing"):
                inspect_media(Path("notes.txt"))

    def test_ffmpeg_binary_uses_external_environment_without_hardcoded_path(self) -> None:
        with patch.dict("os.environ", {"FFMPEG_BINARY": "C:/tools/ffmpeg.exe"}):
            with patch("src.audio.media.Path.is_file", return_value=True):
                self.assertEqual(
                    _codec_binary("FFMPEG_BINARY", "ffmpeg"),
                    str(Path("C:/tools/ffmpeg.exe")),
                )


class _ScriptedModel(nn.Module):
    """Emit one fixed token per row at frame 0; optionally fail the encoder."""

    def __init__(self, token_id: int, encoder_finite: bool = True) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.token_id = token_id
        self.encoder_finite = encoder_finite

    def generate(self, features: torch.Tensor, mask: torch.Tensor) -> GenerationResult:
        batch_size = features.shape[0]
        device = features.device
        return GenerationResult(
            sequences=torch.full((batch_size, 1), self.token_id, dtype=torch.long, device=device),
            durations=torch.ones((batch_size, 1), dtype=torch.long, device=device),
            frame_starts=torch.zeros((batch_size, 1), dtype=torch.long, device=device),
            frame_ends=torch.ones((batch_size, 1), dtype=torch.long, device=device),
            encoder_lengths=mask.sum(dim=1),
            forced_advances=torch.zeros(batch_size, dtype=torch.long, device=device),
            encoder_finite=torch.full((batch_size,), self.encoder_finite, device=device),
        )


class OfflineServiceTests(unittest.TestCase):
    """The planner, decoder boundary, merge, and status wiring run end to end."""

    def _transcribe(self, model: nn.Module, waveform: torch.Tensor):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            configuration = load_config(write_tiny_checkpoint(root / "checkpoint"))
            path = root / "sample.wav"
            write_float_wav(path, waveform, TARGET_RATE)
            transcriber = OfflineTranscriber(model.eval(), configuration, inference_settings())
            return transcriber.transcribe([path])

    def test_long_file_is_chunked_and_merged_once_per_core(self) -> None:
        # Token 3 is "\u2581a"; one token per chunk, owned only by chunk 0's core
        # because every chunk emits it at its own frame 0.
        result = self._transcribe(_ScriptedModel(token_id=3), torch.full((32000,), 0.1))

        self.assertGreater(len(result.plan.items), 1)
        self.assertEqual(result.files[0].status, FileStatus.OK)
        self.assertEqual(result.files[0].transcript, "a")

    def test_non_finite_encoder_discards_tokens_and_reports_status(self) -> None:
        result = self._transcribe(
            _ScriptedModel(token_id=3, encoder_finite=False),
            torch.full((16000,), 0.1),
        )

        self.assertEqual(result.files[0].status, FileStatus.NUMERICAL_FAILURE)
        self.assertEqual(result.files[0].transcript, "")

    def test_silent_file_is_no_speech(self) -> None:
        result = self._transcribe(_ScriptedModel(token_id=11), torch.zeros(16000))

        self.assertEqual(result.files[0].status, FileStatus.NO_SPEECH)


def _chunk(**overrides: object) -> ChunkResult:
    values: dict[str, object] = {
        "item": WorkItem(0, Path("a.wav"), 0, 0, 1, 0, 1, 1),
        "token_ids": (),
        "durations": (),
        "frame_starts": (),
        "frame_ends": (),
        "encoder_length": 1,
        "input_finite": True,
        "silent": False,
        "features_finite": True,
        "encoder_finite": True,
        "forced_advances": 0,
    }
    values.update(overrides)
    return ChunkResult(**values)  # type: ignore[arg-type]


class FileStatusTests(unittest.TestCase):
    """File status is derived from objective facts, most severe first."""

    def test_classification_table(self) -> None:
        cases = [
            ("text", (_chunk(),), FileStatus.OK),
            ("text", (_chunk(), _chunk(encoder_finite=False)), FileStatus.NUMERICAL_FAILURE),
            ("text", (_chunk(features_finite=False),), FileStatus.NUMERICAL_FAILURE),
            ("text", (_chunk(input_finite=False),), FileStatus.INPUT_NONFINITE),
            ("text", (_chunk(forced_advances=2),), FileStatus.DECODER_FORCED_ADVANCE),
            ("", (_chunk(silent=True), _chunk(silent=True)), FileStatus.NO_SPEECH),
            ("", (_chunk(silent=True), _chunk()), FileStatus.EMPTY_TRANSCRIPT),
            ("", (_chunk(input_finite=False, forced_advances=1),), FileStatus.INPUT_NONFINITE),
        ]
        for transcript, chunks, expected in cases:
            with self.subTest(expected=expected, transcript=transcript):
                self.assertEqual(classify_file(transcript, chunks), expected)

    def test_natural_repetition_is_not_flagged(self) -> None:
        """Repeated words are speech, not an anomaly (no gibberish heuristic)."""

        chunk = _chunk(token_ids=(5, 5, 5, 5, 5))
        self.assertEqual(classify_file("ha ha ha ha ha", (chunk,)), FileStatus.OK)


if __name__ == "__main__":
    unittest.main()
