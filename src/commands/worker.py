"""
Long-running transcription worker: load the model once, then serve requests.

    venv\\Scripts\\python.exe -m src.commands.worker

A one-off ``inference.py`` run spends about 5 s before the first sample is
processed (importing PyTorch, loading 1.2 to 2.4 GB of weights, first-call
CUDA setup), while transcribing a 15 s clip takes about 0.2 s. Programs that
send many short files (a labeling pipeline, an editor plugin) keep this
worker running instead and pay the startup once.

Protocol: JSON Lines over standard input and output, one object per line.

Requests (stdin)::

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
* ``{"event": "error", "id", "error"}`` when a request cannot be served.

Startup messages go to standard error, so standard output carries only the
protocol. The worker stops at end of input. Everything is also written to
the dated run log.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from ..configuration.settings import Settings, load_settings
from ..inference.offline import OfflineTranscriber
from ..runtime.logging_setup import configure_run_logging
from .inference import discover_audio_files
from .model_loading import prepare_inference_model

# Literal name: under `python -m` __name__ is "__main__", outside the run log.
logger = logging.getLogger("src.commands.worker")


class Worker:
    """Serve transcription requests with one loaded model."""

    def __init__(self, transcriber: OfflineTranscriber, settings: Settings, output: TextIO) -> None:
        self.transcriber = transcriber
        self.settings = settings
        self.output = output

    def emit(self, payload: dict[str, Any]) -> None:
        """Write one response line and flush, so the caller sees it immediately."""

        self.output.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.output.flush()

    def handle(self, line: str) -> None:
        """Serve one request line; report every failure as an error event."""

        try:
            request = json.loads(line)
            if not isinstance(request, dict) or not isinstance(request.get("path"), str):
                raise ValueError('a request is a JSON object with a string "path"')
        except ValueError as error:
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
        )
        result = self.transcriber.transcribe(
            [record.path for record in discovered.audio],
            metadata=discovered.audio,
        )
        for file_result in result.files:
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
                    "error": reason.splitlines()[0][:300],
                }
            )
        self.emit(
            {
                "event": "done",
                "id": request_id,
                "files": len(result.files) + len(discovered.unreadable),
                "wall_seconds": round(time.perf_counter() - started, 3),
            }
        )


def main(input_stream: TextIO | None = None, output_stream: TextIO | None = None) -> int:
    """Load the model, announce readiness, and serve until end of input."""

    requests = input_stream if input_stream is not None else sys.stdin
    responses = output_stream if output_stream is not None else sys.stdout
    settings = load_settings()
    run_id, log_path = configure_run_logging(settings.paths.log_dir, settings.logging.level)

    def report(line: str) -> None:
        print(line, file=sys.stderr)
        logger.info(line)

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
    for line in requests:
        if line.strip():
            worker.handle(line)
    logger.info("worker input ended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
