"""Regression tests for the intentionally minimal public inference command."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.commands.inference import parse_arguments, write_transcription_csv


class InferenceCliTests(unittest.TestCase):
    """Keep the root command easy to use and free of maintenance flags."""

    def test_parser_accepts_only_transcription_input(self) -> None:
        """One --transcribe value is the complete public command contract."""

        with patch.object(sys, "argv", ["inference.py", "--transcribe", "docs"]):
            arguments = parse_arguments()

        self.assertEqual(arguments.transcribe, Path("docs"))
        self.assertEqual(vars(arguments), {"transcribe": Path("docs")})

    def test_parser_rejects_removed_maintenance_flags(self) -> None:
        """Checkpoint maintenance internals cannot leak back into the CLI."""

        with patch.object(
            sys,
            "argv",
            ["inference.py", "--transcribe", "docs", "--repair"],
        ):
            with self.assertRaises(SystemExit):
                parse_arguments()

    def test_folder_csv_uses_requested_columns(self) -> None:
        """Folder output preserves the path, filename, duration, and text fields."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "transcriptions.csv"
            write_transcription_csv(
                output_path,
                [
                    {
                        "path_audio": "C:/audio/sample.wav",
                        "filename_audio": "sample.wav",
                        "duration_audio": 1.25,
                        "transcription": "hello",
                    }
                ],
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            lines[0],
            "path_audio,filename_audio,duration_audio,transcription",
        )
        self.assertEqual(lines[1], "C:/audio/sample.wav,sample.wav,1.25,hello")


if __name__ == "__main__":
    unittest.main()