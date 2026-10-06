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
