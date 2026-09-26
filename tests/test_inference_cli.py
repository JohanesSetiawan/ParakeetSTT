"""Regression tests for the intentionally minimal public inference command."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from src.commands.inference import (
    CSV_COLUMNS,
    csv_rows,
    discover_audio_files,
    main,
    parse_arguments,
    persist_transcriptions,
    write_transcription_csv,
)
from src.commands.reporting import ProgressReporter
from src.configuration.config import PROJECT_ROOT
from src.configuration.settings import load_settings
from src.inference.offline import FileStatus, OfflineFileResult, OfflineRunResult
from src.inference.planning import ExecutionPlan
from support import write_float_wav

ROW = {
    "path_audio": "C:/audio/sample.wav",
    "filename_audio": "sample.wav",
    "duration_audio": 1.25,
    "status": "ok",
    "transcription": "hello, world",
}


def _run_result(paths: list[Path]) -> OfflineRunResult:
    files = tuple(
        OfflineFileResult(path=path, duration_seconds=1.0, transcript="text", status=FileStatus.OK, chunks=())
        for path in paths
    )
    plan = ExecutionPlan(metadata=(), items=(), batches=(), target_sample_rate=16000,
                         max_feature_frames=1, max_batch_feature_frames=1)
    return OfflineRunResult(files=files, plan=plan, elapsed_seconds=1.0, peak_memory={},
                            media_decode_seconds=0.0, feature_seconds=0.0, generation_seconds=0.0)


class InferenceCliTests(unittest.TestCase):
    """Keep the root command easy to use and free of maintenance flags."""

    def test_parser_accepts_only_transcription_input(self) -> None:
        """One --transcribe value is the complete public command contract."""

        arguments = parse_arguments(["--transcribe", "docs"])

        self.assertEqual(vars(arguments), {"transcribe": Path("docs")})

    def test_parser_rejects_removed_maintenance_flags(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_arguments(["--transcribe", "docs", "--repair"])

    def test_folder_csv_includes_status_column(self) -> None:
        """Every row states its status so anomalies are not silent successes."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "transcriptions.csv"
            write_transcription_csv(output_path, [ROW])
            lines = output_path.read_text(encoding="utf-8").splitlines()
            leftovers = list(Path(temporary_directory).glob("*.tmp"))

        self.assertEqual(lines[0], ",".join(CSV_COLUMNS))
        self.assertEqual(lines[1], 'C:/audio/sample.wav,sample.wav,1.25,ok,"hello, world"')
        self.assertEqual(leftovers, [])


class DiscoveryTests(unittest.TestCase):
    """Unreadable files are reported, never silently dropped."""

    def test_unreadable_files_are_listed_not_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_float_wav(root / "b.wav", torch.zeros(1600), 16000)
            write_float_wav(root / "a.wav", torch.zeros(1600), 16000)
            (root / "notes.txt").write_text("not audio", encoding="utf-8")
            (root / "transcriptions.csv").write_text("path_audio\n", encoding="utf-8")

            discovered = discover_audio_files(
                root,
                extensions=(),
                recursive=False,
                excluded_names=frozenset({"transcriptions.csv"}),
            )

        self.assertEqual([record.path.name for record in discovered.audio], ["a.wav", "b.wav"])
        self.assertEqual([path.name for path, _reason in discovered.unreadable], ["notes.txt"])

    def test_unreadable_files_get_a_csv_row_in_path_order(self) -> None:
        root = Path("C:/audio")
        rows = csv_rows(
            _run_result([root / "a.wav", root / "c.wav"]),
            [(root / "b.m4a", "no decoder")],
        )

        self.assertEqual([row["filename_audio"] for row in rows], ["a.wav", "b.m4a", "c.wav"])
        self.assertEqual(rows[1]["status"], FileStatus.UNREADABLE.value)
        self.assertEqual(rows[1]["transcription"], "")

    def test_missing_ffprobe_marks_file_unreadable(self) -> None:
        """No ffprobe installed: the file is unreadable, the run continues."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_float_wav(root / "a.wav", torch.zeros(1600), 16000)
            (root / "clip.m4a").write_bytes(b"not decodable by libsndfile")

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("src.audio.media.shutil.which", return_value=None),
            ):
                os.environ.pop("FFPROBE_BINARY", None)
                discovered = discover_audio_files(root, extensions=(), recursive=False)

        self.assertEqual([path.name for path, _ in discovered.unreadable], ["clip.m4a"])
        self.assertIn("ffprobe was not found", discovered.unreadable[0][1])

    def test_misconfigured_ffprobe_path_stops_the_run(self) -> None:
        """A wrong FFPROBE_BINARY is a setup error, not an unreadable file."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "clip.m4a").write_bytes(b"not decodable by libsndfile")

            with patch.dict(os.environ, {"FFPROBE_BINARY": str(root / "missing" / "ffprobe.exe")}):
                with self.assertRaisesRegex(RuntimeError, "FFPROBE_BINARY points to a missing executable"):
                    discover_audio_files(root, extensions=(), recursive=False)

    def test_single_unreadable_file_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notes.txt"
            path.write_text("not audio", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "notes.txt"):
                discover_audio_files(path, extensions=(), recursive=False)


class PersistenceTests(unittest.TestCase):
    """A locked output CSV must not cost the transcripts of the run."""

    def test_fallback_location_keeps_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            blocked = root / "transcriptions.csv"
            blocked.mkdir()  # a directory cannot be replaced by a file: the write fails
            fallback = root / "logs" / "transcriptions_run.csv"

            written, used_fallback = persist_transcriptions(blocked, fallback, [ROW])

            self.assertTrue(used_fallback)
            self.assertEqual(written, fallback)
            self.assertIn("hello, world", fallback.read_text(encoding="utf-8"))

    def test_both_locations_failing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "a.csv").mkdir()
            (root / "b.csv").mkdir()

            with self.assertRaises(OSError):
                persist_transcriptions(root / "a.csv", root / "b.csv", [ROW])


class CommandFlowTests(unittest.TestCase):
    """Cheap input errors fail before any checkpoint or model work."""

    def test_missing_input_fails_before_loading_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "typo_folder"

            with (
                patch("src.commands.inference.ensure_first_run_ready") as bootstrap_mock,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as errors,
            ):
                exit_code = main(["--transcribe", str(missing)])

        self.assertEqual(exit_code, 1)
        bootstrap_mock.assert_not_called()
        self.assertIn("Input path does not exist", errors.getvalue())

    def test_module_entry_point_writes_failures_to_the_run_log(self) -> None:
        """
        Regression: under `python -m`, a __name__ logger is "__main__", so the
        traceback never reached the log file the error message points to.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "typo_folder"
            completed = subprocess.run(
                [sys.executable, "-m", "src.commands.inference", "--transcribe", str(missing)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )

        self.assertEqual(completed.returncode, 1, completed.stderr)
        run_id = next(
            line.split(": ", 1)[1]
            for line in completed.stdout.splitlines()
            if line.startswith("Run id: ")
        )
        log_path = next(
            Path(line.split(": ", 1)[1])
            for line in completed.stdout.splitlines()
            if line.startswith("Log file: ")
        )
        self.assertEqual(log_path.parent, load_settings().paths.log_dir)
        run_lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if f"run={run_id}" in line]
        self.assertTrue(any("run failed" in line for line in run_lines), run_lines)
        self.assertTrue(any("command=transcribe" in line for line in run_lines), run_lines)


class ProgressReporterTests(unittest.TestCase):
    """Terminal progress is plain text, rate-limited, and always final."""

    def test_rate_limit_and_final_line(self) -> None:
        lines: list[str] = []
        now = [0.0]
        reporter = ProgressReporter("Batch", 10.0, emit=lines.append, clock=lambda: now[0])

        for completed in range(1, 101):
            now[0] = completed * 0.5
            reporter(completed, 100)

        self.assertLessEqual(len(lines), 7)
        self.assertEqual(
            lines[-1],
            "Batch: 100 / 100, Progress: 100.00 percent, Elapsed: 50.0 s, ETA: 0.0 s",
        )
        self.assertTrue(all(line.isascii() and "#" not in line for line in lines))

    def test_eta_uses_measured_rate(self) -> None:
        lines: list[str] = []
        now = [0.0]
        reporter = ProgressReporter("Batch", 0.0, emit=lines.append, clock=lambda: now[0])

        now[0] = 4.0
        reporter(1, 5)

        self.assertIn("ETA: 16.0 s", lines[-1])
        self.assertIn("Progress: 20.00 percent", lines[-1])


if __name__ == "__main__":
    unittest.main()
