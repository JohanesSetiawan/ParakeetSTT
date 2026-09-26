"""
The real command, run exactly as a user runs it: `python inference.py --transcribe ...`
in a separate process, against the prepared checkpoint.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.configuration.config import PROJECT_ROOT
from support import SpeechClip

COMMAND_TIMEOUT_SECONDS = 600


def run_command(target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "inference.py", "--transcribe", str(target)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=COMMAND_TIMEOUT_SECONDS,
    )


def single_chunk_clip(clips: tuple[SpeechClip, ...]) -> SpeechClip:
    return next(clip for clip in clips if clip.chunks == 1)


def test_single_file_prints_the_expected_transcript(real_settings, speech_clips) -> None:
    clip = single_chunk_clip(speech_clips)

    completed = run_command(clip.path)

    assert completed.returncode == 0, completed.stderr
    assert "Status: ok" in completed.stdout
    assert f"Transcript: {clip.expected_transcript}" in completed.stdout
    for label in ("Device:", "Precision: float32", "Model load seconds:", "Real-time factor:"):
        assert label in completed.stdout


def test_folder_run_writes_one_row_per_file_with_status(real_settings, speech_clips, tmp_path: Path) -> None:
    folder = tmp_path / "inbox"
    (folder / "nested").mkdir(parents=True)
    for clip in speech_clips:
        shutil.copy(clip.path, folder / clip.path.name)
    shutil.copy(single_chunk_clip(speech_clips).path, folder / "nested" / "copy.flac")
    (folder / "notes.txt").write_text("not audio", encoding="utf-8")

    completed = run_command(folder)
    with (folder / real_settings.inference.output_filename).open(encoding="utf-8", newline="") as handle:
        rows = {row["filename_audio"]: row for row in csv.DictReader(handle)}

    assert completed.returncode == 0, completed.stderr
    assert "Skipped unreadable file: notes.txt" in completed.stdout
    assert rows["notes.txt"]["status"] == "unreadable"
    assert rows["copy.flac"]["transcription"] == single_chunk_clip(speech_clips).expected_transcript
    for clip in speech_clips:
        row = rows[clip.path.name]
        assert row["status"] == clip.expected_status
        assert row["transcription"] == clip.expected_transcript
        assert float(row["duration_audio"]) == pytest.approx(clip.duration_seconds, abs=0.01)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_ffmpeg_only_format_is_decoded_through_the_fallback(real_settings, speech_clips, tmp_path: Path) -> None:
    """AAC in M4A is not a libsndfile format; it must go through ffprobe/ffmpeg."""

    clip = single_chunk_clip(speech_clips)
    m4a = tmp_path / "clip.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(clip.path), "-c:a", "aac", "-b:a", "128k", str(m4a)],
        check=True,
        timeout=120,
    )

    completed = run_command(m4a)

    assert completed.returncode == 0, completed.stderr
    assert "Status: ok" in completed.stdout
    # AAC is lossy, so the text is compared loosely; the duration must not be
    # misread (it once came out as 0.014 s).
    duration_line = next(line for line in completed.stdout.splitlines() if line.startswith("Audio duration seconds:"))
    assert float(duration_line.split(":")[1]) == pytest.approx(clip.duration_seconds, abs=0.1)


def test_missing_input_exits_with_error_and_points_to_the_log(real_settings, tmp_path: Path) -> None:
    completed = run_command(tmp_path / "does_not_exist")

    assert completed.returncode == 1
    assert "Input path does not exist" in completed.stderr
    assert "Details:" in completed.stderr
