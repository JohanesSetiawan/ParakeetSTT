"""Tests for settings validation, crash-safe writes, and dated run logs."""

from __future__ import annotations

import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from src.configuration.settings import DEFAULT_SETTINGS_PATH, load_settings
from src.runtime.filesystem import write_json_atomic, write_text_atomic
from src.runtime.logging_setup import configure_run_logging, log_file_path

VALID_TOML = """
[paths]
weights_dir = "weights/model"
log_dir = "logs"

[logging]
level = "info"

[checkpoint]
request_timeout_seconds = 30
download_attempts = 2
retry_backoff_seconds = 0.5
stream_block_bytes = 4096

[inference]
batch_size = 4
recursive = false
output_filename = "out.csv"
audio_extensions = ["WAV", ".mp3", "wav"]
max_chunk_feature_frames = 1000
overlap_feature_frames = 0
max_batch_feature_frames = 2000
max_padding_fraction = 0.3
progress_interval_seconds = 1
"""


class SettingsTests(unittest.TestCase):
    """Every config.toml field is required, typed, and cross-checked."""

    def _load(self, text: str):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        path = root / "config.toml"
        path.write_text(text, encoding="utf-8")
        return load_settings(path, project_root=root), root

    def test_repository_config_is_valid(self) -> None:
        settings = load_settings(DEFAULT_SETTINGS_PATH)

        self.assertTrue(settings.paths.weights_dir.is_absolute())
        self.assertGreaterEqual(settings.inference.overlap_feature_frames, 0)

    def test_valid_file_is_normalized(self) -> None:
        settings, root = self._load(VALID_TOML)

        self.assertEqual(settings.paths.weights_dir, (root / "weights/model").resolve())
        self.assertEqual(settings.logging.level, "INFO")
        self.assertEqual(settings.inference.audio_extensions, (".wav", ".mp3"))
        self.assertEqual(settings.inference.overlap_feature_frames, 0)
        self.assertEqual(settings.checkpoint.request_timeout_seconds, 30.0)

    def test_invalid_values_name_the_key(self) -> None:
        cases = [
            ("overlap_feature_frames = 0", "overlap_feature_frames = 500", "twice"),
            ("overlap_feature_frames = 0", "overlap_feature_frames = -1", "overlap_feature_frames"),
            ("max_batch_feature_frames = 2000", "max_batch_feature_frames = 999", "max_batch_feature_frames"),
            ('output_filename = "out.csv"', 'output_filename = "../out.csv"', "output_filename"),
            ("batch_size = 4", "batch_size = true", "batch_size"),
            ('level = "info"', 'level = "loud"', "logging.level"),
            ("download_attempts = 2", "download_attempts = 0", "download_attempts"),
            ("request_timeout_seconds = 30", "request_timeout_seconds = 0", "request_timeout_seconds"),
            ("max_padding_fraction = 0.3", "max_padding_fraction = 1.5", "max_padding_fraction"),
        ]
        for old, new, message in cases:
            with self.subTest(new=new):
                with self.assertRaisesRegex(ValueError, message):
                    self._load(VALID_TOML.replace(old, new))

    def test_missing_section_is_rejected(self) -> None:
        text = VALID_TOML.replace("[checkpoint]", "[not_checkpoint]")
        with self.assertRaisesRegex(ValueError, r"\[checkpoint\]"):
            self._load(text)


class AtomicWriteTests(unittest.TestCase):
    """A failed write never truncates the previous file."""

    def test_writes_and_leaves_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "data.json"
            write_json_atomic(path, {"b": 1, "a": 2})

            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "a": 2,\n  "b": 1\n}\n')
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_failure_keeps_previous_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "result.csv"
            path.write_text("previous", encoding="utf-8")

            with patch("src.runtime.filesystem.os.fsync", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    write_text_atomic(path, "new content")

            self.assertEqual(path.read_text(encoding="utf-8"), "previous")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


class RunLoggingTests(unittest.TestCase):
    """Run logs follow logs/log_<YYYY-MM-DD>.txt and carry the run id."""

    def test_dated_file_and_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_dir = Path(temporary_directory) / "logs"
            run_id, path = configure_run_logging(log_dir, "INFO")
            logging.getLogger("src.tests").info("hello from test")
            second_run_id, second_path = configure_run_logging(log_dir, "INFO")
            logging.getLogger("src.tests").info("second run")

            for handler in list(logging.getLogger("src").handlers):
                handler.close()
                logging.getLogger("src").removeHandler(handler)
            text = path.read_text(encoding="utf-8")

        self.assertEqual(path.name, f"log_{datetime.now():%Y-%m-%d}.txt")
        self.assertEqual(path, second_path)
        self.assertIn(f"run={run_id}", text)
        self.assertIn(f"run={second_run_id}", text)
        self.assertEqual(text.count("hello from test"), 1)
        self.assertEqual(text.count("second run"), 1)

    def test_log_file_name_uses_given_date(self) -> None:
        path = log_file_path(Path("logs"), datetime(2026, 1, 5, 23, 59))
        self.assertEqual(path, Path("logs") / "log_2026-01-05.txt")


if __name__ == "__main__":
    unittest.main()
