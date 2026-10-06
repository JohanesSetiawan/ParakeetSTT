"""The JSON Lines transcription worker, with a stand-in model."""

from __future__ import annotations

import io
import json
from pathlib import Path

import torch

from src.commands.worker import Worker
from src.configuration.settings import load_settings
from src.inference.offline import OfflineTranscriber
from support import ScriptedModel, inference_settings, write_float_wav


def make_worker(tiny_configuration) -> tuple[Worker, io.StringIO]:
    output = io.StringIO()
    transcriber = OfflineTranscriber(ScriptedModel(3).eval(), tiny_configuration, inference_settings())
    return Worker(transcriber, load_settings(), output), output


def events(output: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_a_file_request_returns_its_transcript_then_done(tmp_path: Path, tiny_configuration) -> None:
    path = tmp_path / "clip.wav"
    write_float_wav(path, torch.full((16000,), 0.1), 16000)
    worker, output = make_worker(tiny_configuration)

    worker.handle(json.dumps({"path": str(path), "id": 7}))

    transcript, done = events(output)
    assert transcript["event"] == "transcript"
    assert transcript["id"] == 7
    assert Path(transcript["path"]) == path.resolve()
    assert transcript["status"] in {"ok", "empty_transcript", "no_speech"}
    assert transcript["duration_seconds"] == 1.0
    assert done == {"event": "done", "id": 7, "files": 1, "wall_seconds": done["wall_seconds"]}


def test_a_folder_request_lists_unreadable_files_too(tmp_path: Path, tiny_configuration) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    write_float_wav(folder / "a.wav", torch.full((8000,), 0.1), 16000)
    (folder / "notes.txt").write_text("not audio", encoding="utf-8")
    worker, output = make_worker(tiny_configuration)

    worker.handle(json.dumps({"path": str(folder)}))

    replies = events(output)
    statuses = {Path(reply["path"]).name: reply["status"] for reply in replies if reply["event"] == "transcript"}
    assert statuses["notes.txt"] == "unreadable"
    assert "a.wav" in statuses
    assert replies[-1]["event"] == "done" and replies[-1]["files"] == 2


def test_bad_requests_are_reported_and_the_worker_keeps_serving(tmp_path: Path, tiny_configuration) -> None:
    path = tmp_path / "clip.wav"
    write_float_wav(path, torch.full((8000,), 0.1), 16000)
    worker, output = make_worker(tiny_configuration)

    worker.handle("not json")
    worker.handle(json.dumps({"id": 1}))
    worker.handle(json.dumps({"path": str(tmp_path / "missing.wav"), "id": 2}))
    worker.handle(json.dumps({"path": str(path), "id": 3}))

    replies = events(output)
    assert [reply["event"] for reply in replies] == ["error", "error", "error", "transcript", "done"]
    assert replies[0]["id"] is None and "invalid request" in replies[0]["error"]
    assert replies[2]["id"] == 2
    assert replies[-1]["id"] == 3
