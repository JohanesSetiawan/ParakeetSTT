"""
Long-running transcription worker: load the model once, then serve requests.

    venv\\Scripts\\python.exe -m src.commands.worker

A one-off ``inference.py`` run spends about 5 s before the first sample is
processed (importing PyTorch, loading 1.2 to 2.4 GB of weights, first-call
CUDA setup), while transcribing a 15 s clip takes about 0.2 s. Programs that
send many short files (a labeling pipeline, an editor plugin) keep this
worker running instead and pay the startup once.

Protocol: JSON Lines over standard input and output, one object per line.

Requests (stdin, UTF-8)::

    {"path": "C:/audio/clip.wav", "id": "anything"}

``path`` is a file or a folder (folders follow ``inference.recursive`` and
``inference.audio_extensions``); ``id`` is optional and echoed back. Several
files in one request are batched together.

Responses (stdout), in order:

* ``{"event": "ready", ...}`` once, after the model is loaded;
* one ``{"event": "transcript", "path", "status", "transcript",
  "duration_seconds", "processing_seconds", "id"}`` per file, including
  unreadable files with ``status`` "unreadable" and an ``error``;
* ``{"event": "done", "id", "files", "wall_seconds"}`` after each request;
* ``{"event": "error", "id", "error"}`` when a request cannot be served
  (not UTF-8, not a JSON object with a string ``path``, a path that does not
  exist, an out-of-memory batch).

Responses are ASCII: characters outside it are written as JSON ``\\u``
escapes, so a console code page (cp1252 on Windows pipes) can never corrupt
or reject them. Requests are read as bytes and decoded as UTF-8 per line, so
one bad line is reported instead of ending the worker. Startup messages go to
standard error, so standard output carries only the protocol. The worker
stops at end of input. Everything is also written to the dated run log.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, Iterable, TextIO

from ..configuration.settings import Settings, load_settings
from ..inference.offline import OfflineTranscriber
from ..runtime.logging_setup import configure_run_logging
from .inference import discover_audio_files
from .model_loading import prepare_inference_model
from .reporting import line_reporter

# Literal name: under `python -m` __name__ is "__main__", outside the run log.
logger = logging.getLogger("src.commands.worker")

# Characters of an unreadable reason kept in a response; the full reason is
# in the run log.
UNREADABLE_REASON_CHARACTERS = 300


class Worker:
    """Serve transcription requests with one loaded model."""

    def __init__(self, transcriber: OfflineTranscriber, settings: Settings, output: TextIO) -> None:
        self.transcriber = transcriber
        self.settings = settings
        self.output = output

    def emit(self, payload: dict[str, Any]) -> None:
        """Write one ASCII response line and flush, so the caller sees it immediately."""

        self.output.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.output.flush()

    def handle(self, line: bytes | str) -> None:
        """Serve one request line; report every failure as an error event."""

        try:
            # utf-8-sig also accepts a byte order mark, which some Windows
            # shells put at the start of piped text.
            text = line.decode("utf-8-sig") if isinstance(line, bytes) else line
            request = json.loads(text)
            if not isinstance(request, dict) or not isinstance(request.get("path"), str):
                raise ValueError('a request is a JSON object with a string "path"')
        except ValueError as error:
            # UnicodeDecodeError and json.JSONDecodeError are ValueErrors.
            self.emit({"event": "error", "id": None, "error": f"invalid request: {error}"})
            return

        request_id = request.get("id")
        started = time.perf_counter()
        try:
            self._transcribe(Path(request["path"]), request_id)
        except Exception as error:
            # The worker keeps serving: one bad request (a missing folder, an
            # out-of-memory batch) must not end the session for the next one.
            logger.exception("request %r failed", request_id)
            self.emit({"event": "error", "id": request_id, "error": str(error)})
            return
        logger.info("request %r served in %.3f s", request_id, time.perf_counter() - started)

    def _transcribe(self, path: Path, request_id: object) -> None:
        inference = self.settings.inference
        started = time.perf_counter()
        discovered = discover_audio_files(
            path.expanduser().resolve(),
            inference.audio_extensions,
            inference.recursive,
            excluded_names=frozenset({inference.output_filename}),
            require_audio=False,
        )
        transcribed_files = ()
        if discovered.audio:
            result = self.transcriber.transcribe(
                [record.path for record in discovered.audio],
                metadata=discovered.audio,
            )
            transcribed_files = result.files
        for file_result in transcribed_files:
            self.emit(
                {
                    "event": "transcript",
                    "id": request_id,
                    "path": str(file_result.path),
                    "status": file_result.status.value,
                    "transcript": file_result.transcript,
                    "duration_seconds": round(file_result.duration_seconds, 3),
                    "processing_seconds": round(file_result.processing_seconds, 3),
                }
            )
        for unreadable_path, reason in discovered.unreadable:
            self.emit(
                {
                    "event": "transcript",
                    "id": request_id,
                    "path": str(unreadable_path),
                    "status": "unreadable",
                    "transcript": "",
                    "error": reason.splitlines()[0][:UNREADABLE_REASON_CHARACTERS],
                }
            )
        self.emit(
            {
                "event": "done",
                "id": request_id,
                "files": len(transcribed_files) + len(discovered.unreadable),
                "wall_seconds": round(time.perf_counter() - started, 3),
            }
        )


def serve(worker: Worker, requests: Iterable[bytes | str]) -> None:
    """Handle every non-blank request line until the input ends."""

    for line in requests:
        if line.strip():
            worker.handle(line)


def main(input_stream: BinaryIO | None = None, output_stream: TextIO | None = None) -> int:
    """Load the model, announce readiness, and serve until end of input."""

    # Binary input: each line is decoded as UTF-8 by the worker, whatever the
    # console code page. Startup lines on standard error may contain paths;
    # characters it cannot encode are escaped instead of raising.
    requests = input_stream if input_stream is not None else sys.stdin.buffer
    responses = output_stream if output_stream is not None else sys.stdout
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")

    settings = load_settings()
    run_id, log_path = configure_run_logging(settings.paths.log_dir, settings.logging.level)
    report = line_reporter(logger, sys.stderr)
    report(f"Run id: {run_id}")
    report(f"Log file: {log_path}")
    try:
        prepared = prepare_inference_model(settings, report)
    except Exception as error:
        logger.exception("worker startup failed")
        print(f"Error: {error}", file=sys.stderr)
        return 1

    transcriber = OfflineTranscriber(prepared.model, prepared.configuration, prepared.inference)
    worker = Worker(transcriber, settings, responses)
    worker.emit(
        {
            "event": "ready",
            "run_id": run_id,
            "device": prepared.runtime.device,
            "precision": prepared.precision,
            "graph_decoding": prepared.graph_decoding,
            "load_seconds": round(prepared.load_seconds, 3),
        }
    )
    serve(worker, requests)
    logger.info("worker input ended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
