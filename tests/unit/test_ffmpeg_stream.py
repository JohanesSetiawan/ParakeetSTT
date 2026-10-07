"""Persistent FFmpeg decoding for formats libsndfile cannot read."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.audio import media as media_module
from src.audio.media import inspect_media, open_media_session
from src.inference.planning import build_execution_plan
from support import write_float_wav

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")

TARGET_RATE = 16000


@pytest.fixture
def m4a(tmp_path: Path) -> Path:
    """Five seconds of a chirp encoded as AAC, which libsndfile cannot decode."""

    time = torch.arange(TARGET_RATE * 5, dtype=torch.float64) / TARGET_RATE
    chirp = (0.3 * torch.sin(2 * np.pi * (200 + 300 * time) * time)).to(torch.float32)
    wav = tmp_path / "source.wav"
    write_float_wav(wav, chirp, TARGET_RATE)
    path = tmp_path / "clip.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "128k", str(path)],
        check=True,
        timeout=120,
    )
    return path


def full_decode(path: Path) -> torch.Tensor:
    completed = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-ac", "1",
         "-ar", str(TARGET_RATE), "-f", "f32le", "pipe:1"],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return torch.frombuffer(bytearray(completed.stdout), dtype=torch.float32).clone()


def plan_items(path: Path, overlap: int):
    plan = build_execution_plan(
        [inspect_media(path)],
        target_sample_rate=TARGET_RATE,
        hop_length=160,
        max_chunk_feature_frames=120,
        overlap_feature_frames=overlap,
        batch_size=4,
        max_batch_feature_frames=480,
        max_padding_fraction=1.0,
        max_open_files=8,
    )
    return sorted(plan.items, key=lambda item: item.chunk_index)


@pytest.mark.parametrize("overlap", [0, 10])
def test_sequential_chunks_are_slices_of_one_continuous_decode(m4a: Path, overlap: int) -> None:
    reference = full_decode(m4a)
    items = plan_items(m4a, overlap)

    with open_media_session(m4a) as session:
        previous = None
        for item in items:
            segment, waveform = session.read_sequential_segment(
                item.source_start_frame,
                item.source_end_frame,
                TARGET_RATE,
                previous_source_end_frame=previous[0] if previous else None,
                previous_waveform=previous[1] if previous else None,
            )
            previous = (item.source_end_frame, waveform)
            expected = torch.nn.functional.pad(
                reference[item.source_start_frame : item.source_end_frame],
                (0, max(0, item.source_end_frame - reference.numel())),
            )

            assert torch.equal(segment.waveform, expected), item.chunk_index

    assert len(items) > 3


def test_one_ffmpeg_process_decodes_the_whole_file(m4a: Path) -> None:
    items = plan_items(m4a, overlap=10)
    spawned: list[object] = []
    real_popen = subprocess.Popen

    def counting_popen(*args, **kwargs):
        spawned.append(args)
        return real_popen(*args, **kwargs)

    with (
        patch.object(media_module.subprocess, "Popen", counting_popen),
        patch.object(media_module.subprocess, "run", side_effect=AssertionError("per-chunk ffmpeg call")),
        open_media_session(m4a) as session,
    ):
        previous = None
        for item in items:
            _segment, waveform = session.read_sequential_segment(
                item.source_start_frame,
                item.source_end_frame,
                TARGET_RATE,
                previous_source_end_frame=previous[0] if previous else None,
                previous_waveform=previous[1] if previous else None,
            )
            previous = (item.source_end_frame, waveform)

    assert len(spawned) == 1


def test_reading_behind_the_stream_uses_a_one_off_decode(m4a: Path) -> None:
    items = plan_items(m4a, overlap=10)

    with open_media_session(m4a) as session:
        session.read_sequential_segment(items[0].source_start_frame, items[0].source_end_frame, TARGET_RATE)
        session.read_sequential_segment(items[1].source_start_frame, items[1].source_end_frame, TARGET_RATE)
        # Chunk 0 again: behind the stream, e.g. a collapse-recovery window.
        again, _waveform = session.read_sequential_segment(
            items[0].source_start_frame,
            items[0].source_end_frame,
            TARGET_RATE,
        )

    assert again.waveform.numel() == items[0].source_frame_count
    assert again.finite


def test_closing_the_session_stops_the_process(m4a: Path) -> None:
    session = open_media_session(m4a)
    session.read_sequential_segment(0, 8000, TARGET_RATE)
    process = session._stream._process

    session.close()

    assert process.poll() is not None


def test_closing_mid_file_joins_the_stderr_thread_cleanly(m4a: Path) -> None:
    """The drain thread must finish before its pipe is closed, without an exception."""

    import threading

    thread_errors: list[BaseException] = []
    previous_hook = threading.excepthook
    threading.excepthook = lambda arguments: thread_errors.append(arguments.exc_value)
    try:
        session = open_media_session(m4a)
        session.read_sequential_segment(0, 8000, TARGET_RATE)
        stream = session._stream
        session.close()
    finally:
        threading.excepthook = previous_hook

    assert not stream._stderr_thread.is_alive()
    assert stream._process.stderr.closed
    assert thread_errors == []


def test_a_long_jump_ahead_restarts_ffmpeg_at_the_new_position(m4a: Path) -> None:
    """A decode stream moving to its next block must not decode the audio in between."""

    reference = full_decode(m4a)
    spawned: list[list[str]] = []
    real_popen = subprocess.Popen

    def counting_popen(command, *args, **kwargs):
        spawned.append(list(command))
        return real_popen(command, *args, **kwargs)

    with patch.object(media_module.subprocess, "Popen", counting_popen), open_media_session(m4a) as session:
        session.read_sequential_segment(0, 8000, TARGET_RATE)
        # Short jump (less than the piece read): the stream skips ahead.
        session.read_sequential_segment(12_000, 20_000, TARGET_RATE)
        # Long jump: a new process starts at the requested position.
        jumped, _waveform = session.read_sequential_segment(56_000, 64_000, TARGET_RATE)

    assert len(spawned) == 2
    assert "-ss" not in spawned[0]
    assert spawned[1][spawned[1].index("-ss") + 1] == f"{56_000 / TARGET_RATE:.9f}"
    assert jumped.waveform.numel() == 8000
    # FFmpeg seeks accurately; only the AAC decoder's start-up at the seek
    # point differs (measured: the first 32 ms). A decoder's first chunk of a
    # block begins with the recovery reach and the overlap before the core,
    # so that stretch never reaches a transcribed core.
    settle = 512
    assert torch.allclose(jumped.waveform[settle:], reference[56_000 + settle : 64_000], atol=1e-4)
