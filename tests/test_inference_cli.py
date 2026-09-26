"""Regression tests for the intentionally minimal public inference command."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.commands.inference import (
    CSV_COLUMNS,
    discover_audio_files,
    parse_arguments,
    write_transcription_csv,
)
from src.commands.reporting import ProgressReporter
from support import write_float_wav


class InferenceCliTests(unittest.TestCase):
    """Keep the root command easy to use and free of maintenance flags."""

    def test_parser_accepts_only_transcription_input(self) -> None:
        """One --transcribe value is the complete public command contract."""

        arguments = parse_arguments(["--transcribe", "docs"])

        self.assertEqual(vars(arguments), {"transcribe": Path("docs")})

    def test_parser_rejects_removed_maintenance_flags(self) -> None:
        """Checkpoint maintenance internals cannot leak back into the CLI."""

        with self.assertRaises(SystemExit):
            parse_arguments(["--transcribe", "docs", "--repair"])

    def test_folder_csv_includes_status_column(self) -> None:
        """Every row states its status so anomalies are not silent successes."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "transcriptions.csv"
            write_transcription_csv(
                output_path,
                [
                    {
                        "path_audio": "C:/audio/sample.wav",
                        "filename_audio": "sample.wav",
                        "duration_audio": 1.25,
                        "status": "ok",
                        "transcription": "hello, world",
                    }
                ],
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()
            leftovers = list(Path(temporary_directory).glob("*.tmp"))

        self.assertEqual(lines[0], ",".join(CSV_COLUMNS))
        self.assertEqual(lines[1], 'C:/audio/sample.wav,sample.wav,1.25,ok,"hello, world"')
        self.assertEqual(leftovers, [])

    def test_discovery_skips_non_audio_and_previous_output(self) -> None:
        """Text files and an earlier transcriptions.csv never become inputs."""

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

        self.assertEqual([record.path.name for record in discovered], ["a.wav", "b.wav"])
        self.assertEqual(discovered[0].frame_count, 1600)

    def test_single_unreadable_file_fails_clearly(self) -> None:
        """Naming one non-audio file raises an error that names the file."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notes.txt"
            path.write_text("not audio", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "notes.txt"):
                discover_audio_files(path, extensions=(), recursive=False)


class ProgressReporterTests(unittest.TestCase):
    """Terminal progress is plain text, rate-limited, and always final."""

    def test_rate_limit_and_final_line(self) -> None:
        lines: list[str] = []
        now = [0.0]
        reporter = ProgressReporter("Batch", 10.0, emit=lines.append, clock=lambda: now[0])

        for completed in range(1, 101):
            now[0] = completed * 0.5  # 50 s total, one line allowed per 10 s
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
