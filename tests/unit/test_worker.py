"""The JSON Lines transcription worker, with a stand-in model."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

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


def test_unreadable_only_requests_still_report_each_file(tmp_path: Path, tiny_configuration) -> None:
    """Review finding: a request with no readable file got one generic error and no per-file status."""

    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"RIFF not really audio")
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "a.txt").write_text("not audio", encoding="utf-8")
    (folder / "b.txt").write_text("not audio either", encoding="utf-8")
    worker, output = make_worker(tiny_configuration)

    worker.handle(json.dumps({"path": str(corrupt), "id": "file"}))
    worker.handle(json.dumps({"path": str(folder), "id": "folder"}))

    replies = events(output)
    assert [reply["event"] for reply in replies] == ["transcript", "done", "transcript", "transcript", "done"]
    assert replies[0]["status"] == "unreadable" and replies[0]["error"]
    assert replies[1] == {"event": "done", "id": "file", "files": 1, "wall_seconds": replies[1]["wall_seconds"]}
    assert {Path(reply["path"]).name for reply in replies[2:4]} == {"a.txt", "b.txt"}
    assert replies[4]["files"] == 2


def test_non_ascii_paths_round_trip_as_ascii_json(tmp_path: Path, tiny_configuration) -> None:
    """
    Review finding: on Windows pipes stdin/stdout are cp1252, so a UTF-8 path
    with "\u00c1" (bytes C3 81; 0x81 is undefined in cp1252) killed the worker.
    """

    folder = tmp_path / "\u00c1udio \u65e5\u672c"
    folder.mkdir()
    clip = folder / "\u00c1bc.wav"
    write_float_wav(clip, torch.full((8000,), 0.1), 16000)
    worker, output = make_worker(tiny_configuration)

    worker.handle(json.dumps({"path": str(clip), "id": 1}, ensure_ascii=False).encode("utf-8"))

    text = output.getvalue()
    assert text.isascii()
    transcript, done = events(output)
    assert Path(transcript["path"]) == clip.resolve()
    assert done["files"] == 1


def test_invalid_utf8_is_reported_and_the_worker_keeps_serving(tmp_path: Path, tiny_configuration) -> None:
    clip = tmp_path / "clip.wav"
    write_float_wav(clip, torch.full((8000,), 0.1), 16000)
    worker, output = make_worker(tiny_configuration)

    worker.handle(b"\xff\xfe not utf-8 \x81\n")
    worker.handle(("\ufeff" + json.dumps({"path": str(clip), "id": 2})).encode("utf-8"))

    replies = events(output)
    assert replies[0]["event"] == "error" and "invalid request" in replies[0]["error"]
    assert [reply["event"] for reply in replies[1:]] == ["transcript", "done"]


def test_main_serves_byte_input_until_it_ends(tmp_path: Path, tiny_configuration) -> None:
    from types import SimpleNamespace

    from src.commands import worker as worker_module
    from src.runtime.device import describe_runtime

    clip = tmp_path / "\u00e9t\u00e9.wav"
    write_float_wav(clip, torch.full((8000,), 0.1), 16000)
    prepared = SimpleNamespace(
        model=ScriptedModel(3).eval(),
        configuration=tiny_configuration,
        inference=inference_settings(),
        runtime=describe_runtime(torch.device("cpu"), torch.float32),
        precision="float32",
        graph_decoding=False,
        load_seconds=0.0,
    )
    requests = io.BytesIO(
        json.dumps({"path": str(clip), "id": "x"}, ensure_ascii=False).encode("utf-8") + b"\n\n"
    )
    output = io.StringIO()

    with patch.object(worker_module, "prepare_inference_model", return_value=prepared):
        exit_code = worker_module.main(requests, output)

    replies = events(output)
    assert exit_code == 0
    assert [reply["event"] for reply in replies] == ["ready", "transcript", "done"]
    assert Path(replies[1]["path"]) == clip.resolve()
