"""Regression tests for first-run readiness marker behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.checkpoint.bootstrap import ensure_first_run_ready, readiness_marker_path


class BootstrapTests(unittest.TestCase):
    """Verify marker-only fast path and explicit repair behavior."""

    def test_ready_marker_skips_checkpoint_preparation(self) -> None:
        """An existing marker returns immediately without calling preparation."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            marker_path = readiness_marker_path(checkpoint_dir)
            marker_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            (checkpoint_dir / "model.pth").write_bytes(b"prepared")

            with patch("src.checkpoint.bootstrap.prepare_checkpoint") as prepare_mock:
                result = ensure_first_run_ready(checkpoint_dir)

            self.assertEqual(result.action, "ready")
            prepare_mock.assert_not_called()

    def test_repair_removes_marker_and_runs_preparation(self) -> None:
        """Repair explicitly invalidates the marker and rebuilds readiness."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            marker_path = readiness_marker_path(checkpoint_dir)
            marker_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            (checkpoint_dir / "model.pth").write_bytes(b"prepared")

            preparation = object()
            with (
                patch("src.checkpoint.bootstrap.prepare_checkpoint", return_value=preparation),
                patch("src.checkpoint.bootstrap.load_model", return_value=(object(), object(), {})),
            ):
                result = ensure_first_run_ready(checkpoint_dir, force_repair=True)

            self.assertEqual(result.action, "prepared")
            self.assertIs(result.preparation, preparation)
            self.assertTrue(marker_path.is_file())


if __name__ == "__main__":
    unittest.main()
