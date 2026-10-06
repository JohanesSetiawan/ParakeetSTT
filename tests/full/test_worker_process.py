"""The worker as a real process: one model load serves several requests."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from src.configuration.config import PROJECT_ROOT


def single_chunk_clips(speech_clips):
    return [clip for clip in speech_clips if clip.chunks == 1]


def test_worker_serves_several_requests_with_one_load(real_settings, speech_clips) -> None:
    clips = single_chunk_clips(speech_clips)[:2]
    requests = "".join(json.dumps({"path": str(clip.path), "id": clip.clip_id}) + "\n" for clip in clips)
    # This session keeps a model on the GPU; give the worker the cached memory.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    completed = subprocess.run(
        [sys.executable, "-m", "src.commands.worker"],
        cwd=PROJECT_ROOT,
        input=requests,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )

    assert completed.returncode == 0, completed.stderr
    replies = [json.loads(line) for line in completed.stdout.splitlines()]
    assert replies[0]["event"] == "ready"
    transcripts = {reply["id"]: reply for reply in replies if reply["event"] == "transcript"}
    for clip in clips:
        assert transcripts[clip.clip_id]["transcript"] == clip.expected_transcript
        assert transcripts[clip.clip_id]["status"] == "ok"
    assert [reply["event"] for reply in replies].count("done") == len(clips)
    # Startup messages stay off the protocol stream.
    assert "Run id:" in completed.stderr
    if not torch.cuda.is_available():
        pytest.skip("timing is only meaningful on the GPU")


def test_worker_handles_non_ascii_paths_through_real_pipes(real_settings, speech_clips, tmp_path) -> None:
    """
    Review finding: Windows pipes are cp1252, so a UTF-8 request for a path
    with "\u00c1" (bytes C3 81; 0x81 is undefined in cp1252) killed the worker.
    """

    import shutil

    clip = single_chunk_clips(speech_clips)[0]
    folder = tmp_path / "\u00c1udio \u65e5\u672c"
    folder.mkdir()
    target = folder / f"\u00e9t\u00e9 {clip.path.name}"
    shutil.copyfile(clip.path, target)
    request = json.dumps({"path": str(target), "id": "unicode"}, ensure_ascii=False) + "\n"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    completed = subprocess.run(
        [sys.executable, "-m", "src.commands.worker"],
        cwd=PROJECT_ROOT,
        input=request.encode("utf-8"),
        capture_output=True,
        timeout=600,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert b"Logging error" not in completed.stderr
    assert completed.stdout.isascii()
    replies = [json.loads(line) for line in completed.stdout.decode("ascii").splitlines()]
    transcript = next(reply for reply in replies if reply["event"] == "transcript")
    assert transcript["path"] == str(target.resolve())
    assert transcript["transcript"] == clip.expected_transcript
