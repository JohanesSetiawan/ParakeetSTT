"""Media decoding and offline orchestration: sessions, seams, merging, statuses."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile
import torch
from torch import nn

from src.audio.media import (
    CodecUnavailableError,
    _codec_binary,
    inspect_media,
    open_media_session,
)
from src.configuration.config import ParakeetConfig
from src.inference.offline import ChunkResult, FileStatus, OfflineTranscriber, classify_file
from src.inference.planning import WorkItem, build_execution_plan
from support import TINY_BLANK_ID, ScriptedModel, inference_settings, write_float_wav

TARGET_RATE = 16000
HOP_LENGTH = 160


def read_plan_sequentially_and_fresh(path: Path, max_chunk: int, overlap: int):
    """Decode every chunk twice: through overlap reuse and as a fresh read."""

    plan = build_execution_plan(
        [inspect_media(path)],
        target_sample_rate=TARGET_RATE,
        hop_length=HOP_LENGTH,
        max_chunk_feature_frames=max_chunk,
        overlap_feature_frames=overlap,
        batch_size=4,
        max_batch_feature_frames=4 * max_chunk,
        max_padding_fraction=1.0,
        max_open_files=8,
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


# =============================================================================
# Media decoding
# =============================================================================


def test_session_reads_bounded_finite_segments(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    write_float_wav(path, torch.linspace(-0.2, 0.2, 32000), TARGET_RATE)

    metadata = inspect_media(path)
    with open_media_session(path) as session:
        first = session.read_segment(0, 8000, TARGET_RATE)
        second = session.read_segment(8000, 16000, TARGET_RATE)

    assert metadata.frame_count == 32000
    assert (first.waveform.numel(), second.waveform.numel()) == (8000, 8000)
    assert first.finite and second.finite
    assert first.rms > 0.0


# Resampled pieces are computed from different source blocks, so their float32
# sums can differ in the last bit (measured up to 1.2e-7); anything larger
# would mean the pieces are on different sample grids.
RESAMPLING_TOLERANCE = 1e-6


@pytest.mark.parametrize("rate", [48000, 44100, 22050])
@pytest.mark.parametrize(("max_chunk", "overlap"), [(120, 10), (100, 0), (21, 10), (300, 50)])
def test_overlap_reuse_matches_fresh_decode(tmp_path: Path, rate: int, max_chunk: int, overlap: int) -> None:
    """Stereo input at common rates, including overlap longer than the core (21, 10)."""

    path = tmp_path / "stereo.wav"
    samples = np.random.default_rng(7).uniform(-0.5, 0.5, size=(rate * 5 + 1, 2)).astype(np.float32)
    soundfile.write(str(path), samples, rate, subtype="FLOAT")

    pairs = read_plan_sequentially_and_fresh(path, max_chunk=max_chunk, overlap=overlap)

    assert len(pairs) > 1
    for item, segment, reference in pairs:
        assert segment.waveform.numel() == item.source_frame_count
        assert torch.allclose(segment.waveform, reference.waveform, atol=RESAMPLING_TOLERANCE), item.chunk_index


def test_native_rate_input_is_read_exactly(tmp_path: Path) -> None:
    """At 16 kHz there is no resampling, so overlap reuse is bit-identical."""

    path = tmp_path / "native.wav"
    write_float_wav(path, torch.sin(torch.arange(80_000, dtype=torch.float32) / 5.0), TARGET_RATE)

    for item, segment, reference in read_plan_sequentially_and_fresh(path, max_chunk=120, overlap=10):
        assert torch.equal(segment.waveform, reference.waveform), item.chunk_index


def test_multichannel_input_is_averaged_to_mono(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    left = np.full(1600, 0.4, dtype=np.float32)
    right = np.full(1600, -0.2, dtype=np.float32)
    soundfile.write(str(path), np.stack([left, right], axis=1), TARGET_RATE, subtype="FLOAT")

    with open_media_session(path) as session:
        segment = session.read_segment(0, 1600, TARGET_RATE)

    assert segment.source_channels == 2
    assert torch.allclose(segment.waveform, torch.full((1600,), 0.1))


def test_non_finite_samples_are_zeroed_and_flagged(tmp_path: Path) -> None:
    path = tmp_path / "nan.wav"
    waveform = torch.full((1600,), 0.1)
    waveform[10:20] = float("nan")
    write_float_wav(path, waveform, TARGET_RATE)

    with open_media_session(path) as session:
        segment = session.read_segment(0, 1600, TARGET_RATE)

    assert not segment.finite
    assert torch.isfinite(segment.waveform).all()


def test_ffprobe_duration_falls_back_to_container(tmp_path: Path) -> None:
    path = tmp_path / "clip.webm"
    path.write_bytes(b"webm payload")
    probe = {"stream": {"sample_rate": "48000", "channels": 2, "duration": "N/A"}, "format": {"duration": "2.5"}}

    with (
        patch("src.audio.media.soundfile.info", side_effect=RuntimeError("unsupported")),
        patch("src.audio.media._run_ffprobe", return_value=probe),
    ):
        metadata = inspect_media(path)

    assert metadata.frame_count == 120_000
    assert metadata.sample_rate == 48000


def test_unreadable_media_reports_both_backends(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not audio", encoding="utf-8")

    with (
        patch("src.audio.media.soundfile.info", side_effect=RuntimeError("libsndfile says no")),
        patch("src.audio.media._run_ffprobe", side_effect=CodecUnavailableError("ffprobe missing")),
    ):
        with pytest.raises(ValueError, match="libsndfile says no.*ffprobe missing"):
            inspect_media(path)


def test_codec_binary_resolves_from_environment_without_hardcoded_path() -> None:
    with patch.dict("os.environ", {"FFMPEG_BINARY": "C:/tools/ffmpeg.exe"}):
        with patch("src.audio.media.Path.is_file", return_value=True):
            assert _codec_binary("FFMPEG_BINARY", "ffmpeg") == str(Path("C:/tools/ffmpeg.exe"))


# =============================================================================
# Offline orchestration with a scripted model
# =============================================================================


def transcribe_waveforms(
    tmp_path: Path,
    configuration: ParakeetConfig,
    model: nn.Module,
    waveforms: list[torch.Tensor],
    **settings_overrides: object,
):
    paths = []
    for index, waveform in enumerate(waveforms):
        path = tmp_path / f"clip_{index}.wav"
        write_float_wav(path, waveform, TARGET_RATE)
        paths.append(path)
    transcriber = OfflineTranscriber(model.eval(), configuration, inference_settings(**settings_overrides))
    return transcriber.transcribe(paths)


def test_long_file_is_chunked_and_each_position_owned_once(tmp_path, tiny_configuration) -> None:
    # Token 3 is "\u2581a". Every chunk emits it at its own frame 0; only chunk 0's
    # core owns that position, later ones fall in their left overlap.
    result = transcribe_waveforms(tmp_path, tiny_configuration, ScriptedModel(3), [torch.full((32000,), 0.1)])

    assert len(result.plan.items) > 1
    assert result.files[0].status is FileStatus.OK
    assert result.files[0].transcript == "a"


def test_results_follow_input_order_across_mixed_batches(tmp_path, tiny_configuration) -> None:
    waveforms = [torch.full((count,), 0.1) for count in (32000, 1600, 8000)]

    result = transcribe_waveforms(tmp_path, tiny_configuration, ScriptedModel(3), waveforms)

    assert [file_result.path.name for file_result in result.files] == ["clip_0.wav", "clip_1.wav", "clip_2.wav"]
    assert all(file_result.transcript == "a" for file_result in result.files)
    assert result.total_audio_seconds == pytest.approx((32000 + 1600 + 8000) / TARGET_RATE)


def test_non_finite_encoder_discards_tokens(tmp_path, tiny_configuration) -> None:
    result = transcribe_waveforms(
        tmp_path, tiny_configuration, ScriptedModel(3, encoder_finite=False), [torch.full((16000,), 0.1)]
    )

    assert result.files[0].status is FileStatus.NUMERICAL_FAILURE
    assert result.files[0].transcript == ""


def test_silent_file_is_no_speech(tmp_path, tiny_configuration) -> None:
    result = transcribe_waveforms(tmp_path, tiny_configuration, ScriptedModel(TINY_BLANK_ID), [torch.zeros(16000)])

    assert result.files[0].status is FileStatus.NO_SPEECH


def test_duplicate_input_paths_are_rejected(tmp_path, tiny_configuration) -> None:
    path = tmp_path / "clip.wav"
    write_float_wav(path, torch.zeros(1600), TARGET_RATE)
    transcriber = OfflineTranscriber(ScriptedModel(3).eval(), tiny_configuration, inference_settings())

    with pytest.raises(ValueError, match="only once"):
        transcriber.transcribe([path, path])


# =============================================================================
# File status classification
# =============================================================================


def chunk(**overrides: object) -> ChunkResult:
    values: dict[str, object] = {
        "item": WorkItem(0, Path("a.wav"), 0, 0, 1, 0, 1, 1, 0),
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


@pytest.mark.parametrize(
    ("transcript", "chunks", "expected"),
    [
        ("text", (chunk(),), FileStatus.OK),
        ("text", (chunk(), chunk(encoder_finite=False)), FileStatus.NUMERICAL_FAILURE),
        ("text", (chunk(features_finite=False),), FileStatus.NUMERICAL_FAILURE),
        ("text", (chunk(input_finite=False),), FileStatus.INPUT_NONFINITE),
        ("text", (chunk(forced_advances=2),), FileStatus.DECODER_FORCED_ADVANCE),
        ("", (chunk(silent=True), chunk(silent=True)), FileStatus.NO_SPEECH),
        ("", (chunk(silent=True), chunk()), FileStatus.EMPTY_TRANSCRIPT),
        ("", (chunk(input_finite=False, forced_advances=1),), FileStatus.INPUT_NONFINITE),
    ],
)
def test_file_status_is_most_severe_objective_fact(transcript, chunks, expected) -> None:
    assert classify_file(transcript, chunks) is expected


# =============================================================================
# Decoding one batch ahead
# =============================================================================


def test_next_batch_is_decoded_while_the_model_runs(tmp_path, tiny_configuration) -> None:
    """The second batch's audio must be read before the first batch's inference ends."""

    import threading

    paths = []
    for index in range(3):
        path = tmp_path / f"clip_{index}.wav"
        write_float_wav(path, torch.full((8000,), 0.1), 16000)
        paths.append(path)

    second_batch_decoding = threading.Event()
    original_decode = OfflineTranscriber._decode_batch
    decode_calls = []

    def recording_decode(self, items, *args):
        decode_calls.append(items[0].path.name)
        if len(decode_calls) == 2:
            second_batch_decoding.set()
        return original_decode(self, items, *args)

    class WaitingModel(ScriptedModel):
        def __init__(self) -> None:
            super().__init__(3)
            self.saw_prefetch: list[bool] = []

        def generate(self, features, mask):
            if not self.saw_prefetch:
                # Without prefetching, the second decode cannot start until
                # this call returns, and the wait times out.
                self.saw_prefetch.append(second_batch_decoding.wait(timeout=5.0))
            return super().generate(features, mask)

    model = WaitingModel().eval()
    transcriber = OfflineTranscriber(model, tiny_configuration, inference_settings(batch_size=1))
    with patch.object(OfflineTranscriber, "_decode_batch", recording_decode):
        result = transcriber.transcribe(paths)

    assert model.saw_prefetch == [True]
    assert [file.path.name for file in result.files] == [path.name for path in paths]


def test_decode_failure_in_the_worker_reaches_the_caller(tmp_path, tiny_configuration) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"clip_{index}.wav"
        write_float_wav(path, torch.full((8000,), 0.1), 16000)
        paths.append(path)
    original_decode = OfflineTranscriber._decode_batch_audio
    calls = [0]

    def failing_on_second(self, items, *args):
        calls[0] += 1
        if calls[0] == 2:
            raise ValueError("decoder broke on the second batch")
        return original_decode(self, items, *args)

    transcriber = OfflineTranscriber(ScriptedModel(3).eval(), tiny_configuration, inference_settings(batch_size=1))
    with patch.object(OfflineTranscriber, "_decode_batch_audio", failing_on_second):
        with pytest.raises(ValueError, match="second batch"):
            transcriber.transcribe(paths)
