"""Parallel decode streams: same audio as one stream, real parallelism, bounded sessions."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.audio.media import _downmix, open_media_session
from src.inference import offline as offline_module
from src.inference.offline import OfflineTranscriber
from support import ScriptedModel, inference_settings, write_float_wav

TARGET_RATE = 16000


def tone(seconds: float, rate: int) -> torch.Tensor:
    time = torch.arange(int(seconds * rate), dtype=torch.float64) / rate
    return (0.3 * torch.sin(2 * np.pi * 330 * time) + 0.1 * torch.sin(2 * np.pi * 1210 * time)).to(torch.float32)


def stream_settings(decode_workers: int, **overrides: object):
    values: dict[str, object] = {
        "batch_size": 4,
        "max_chunk_feature_frames": 100,
        "overlap_feature_frames": 5,
        "max_batch_feature_frames": 400,
        "max_padding_fraction": 1.0,
        "decode_workers": decode_workers,
        # The recovery reach widens every read; it must not change the audio
        # the model sees.
        "recovery_start_offsets_feature_frames": (-3, 3, -5, 5),
    }
    values.update(overrides)
    return inference_settings(**values)


def model_inputs(path: Path, configuration, decode_workers: int) -> dict[int, torch.Tensor]:
    """Waveform of every planned row the model receives, by chunk index."""

    seen: dict[int, torch.Tensor] = {}
    original = OfflineTranscriber._infer_segments

    def recording(self, items, segments):
        for item, segment in zip(items, segments, strict=True):
            seen[item.chunk_index] = segment.waveform.clone()
            assert segment.source_start_frame == item.source_start_frame
            assert segment.source_end_frame == item.source_end_frame
        return original(self, items, segments)

    transcriber = OfflineTranscriber(ScriptedModel(3).eval(), configuration, stream_settings(decode_workers))
    with patch.object(OfflineTranscriber, "_infer_segments", recording):
        transcriber.transcribe([path])
    return seen


@pytest.mark.parametrize("decode_workers", [2, 4])
def test_streams_feed_the_model_the_same_audio_as_one_stream(
    tmp_path: Path,
    tiny_configuration,
    decode_workers: int,
) -> None:
    path = tmp_path / "speech.wav"
    write_float_wav(path, tone(6.0, TARGET_RATE), TARGET_RATE)

    single = model_inputs(path, tiny_configuration, decode_workers=1)
    parallel = model_inputs(path, tiny_configuration, decode_workers=decode_workers)

    assert len(single) >= 6
    assert single.keys() == parallel.keys()
    # At the native rate a fresh read and overlap reuse are bit-identical.
    for chunk_index, waveform in single.items():
        assert torch.equal(parallel[chunk_index], waveform), chunk_index


def test_resampled_streams_match_fresh_reads(tmp_path: Path, tiny_configuration) -> None:
    """48 kHz input: every chunk equals a fresh resampled read of its window."""

    path = tmp_path / "speech48k.wav"
    write_float_wav(path, tone(6.0, 48_000), 48_000)

    parallel = model_inputs(path, tiny_configuration, decode_workers=4)

    transcriber = OfflineTranscriber(ScriptedModel(3).eval(), tiny_configuration, stream_settings(4))
    plan = transcriber._plan((offline_module.inspect_media(path),))
    with open_media_session(path) as session:
        for item in plan.items:
            fresh = session.read_segment(item.source_start_frame, item.source_end_frame, TARGET_RATE)
            assert torch.allclose(parallel[item.chunk_index], fresh.waveform, atol=1e-6), item.chunk_index


def test_streams_of_one_batch_run_on_several_threads(tmp_path: Path, tiny_configuration) -> None:
    path = tmp_path / "speech.wav"
    write_float_wav(path, tone(6.0, TARGET_RATE), TARGET_RATE)
    thread_names: set[str] = set()
    inside = [0]
    overlapping = [False]
    lock = threading.Lock()
    original = offline_module._StreamDecoder.decode

    def recording(self, items):
        with lock:
            thread_names.add(threading.current_thread().name)
            inside[0] += 1
            overlapping[0] = overlapping[0] or inside[0] > 1
        try:
            # Long enough that the other streams of the batch start meanwhile.
            threading.Event().wait(0.05)
            return original(self, items)
        finally:
            with lock:
                inside[0] -= 1

    transcriber = OfflineTranscriber(ScriptedModel(3).eval(), tiny_configuration, stream_settings(4))
    with patch.object(offline_module._StreamDecoder, "decode", recording):
        transcriber.transcribe([path])

    assert len(thread_names) > 1
    assert all(name.startswith("media-decode") for name in thread_names)
    assert overlapping[0]


def test_every_stream_session_is_closed(tmp_path: Path, tiny_configuration) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"speech_{index}.wav"
        write_float_wav(path, tone(4.0, TARGET_RATE), TARGET_RATE)
        paths.append(path)
    open_sessions: list[object] = []
    opened = [0]
    lock = threading.Lock()
    original_open = offline_module.open_media_session

    def tracking_open(media_path: Path):
        session = original_open(media_path)
        original_close = session.close
        with lock:
            open_sessions.append(session)
            opened[0] += 1

        def tracking_close() -> None:
            with lock:
                if session in open_sessions:
                    open_sessions.remove(session)
            original_close()

        session.close = tracking_close
        return session

    transcriber = OfflineTranscriber(ScriptedModel(3).eval(), tiny_configuration, stream_settings(4))
    plan = transcriber._plan(tuple(offline_module.inspect_media(path) for path in paths))
    with patch.object(offline_module, "open_media_session", tracking_open):
        result = transcriber.transcribe(paths)

    assert len(result.files) == 3
    assert opened[0] == len({item.stream_index for item in plan.items})
    assert open_sessions == []


def test_decode_failure_in_a_stream_reaches_the_caller(tmp_path: Path, tiny_configuration) -> None:
    path = tmp_path / "speech.wav"
    write_float_wav(path, tone(6.0, TARGET_RATE), TARGET_RATE)
    original = offline_module._StreamDecoder.decode
    calls = [0]
    lock = threading.Lock()

    def failing(self, items):
        with lock:
            calls[0] += 1
            fail = calls[0] == 3
        if fail:
            raise ValueError("stream decoder broke")
        return original(self, items)

    transcriber = OfflineTranscriber(ScriptedModel(3).eval(), tiny_configuration, stream_settings(4))
    with patch.object(offline_module._StreamDecoder, "decode", failing):
        with pytest.raises(ValueError, match="stream decoder broke"):
            transcriber.transcribe([path])


# =============================================================================
# Downmix
# =============================================================================


def test_stereo_downmix_equals_the_channel_mean_exactly() -> None:
    generator = np.random.default_rng(0)
    interleaved = generator.standard_normal((10_000, 2)).astype(np.float32)

    mono = _downmix(interleaved)

    assert mono.flags["C_CONTIGUOUS"] and mono.dtype == np.float32
    expected = torch.from_numpy(interleaved).mean(dim=1).numpy()
    assert np.array_equal(mono, expected)


def test_downmix_of_mono_and_many_channels() -> None:
    generator = np.random.default_rng(1)
    mono_input = generator.standard_normal((500, 1)).astype(np.float32)
    six_channels = generator.standard_normal((500, 6)).astype(np.float32)

    assert np.array_equal(_downmix(mono_input), mono_input[:, 0])
    assert _downmix(mono_input).flags["C_CONTIGUOUS"]
    assert np.allclose(_downmix(six_channels), six_channels.mean(axis=1), atol=1e-6)
