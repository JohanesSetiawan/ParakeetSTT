"""Regression tests for first-run readiness marker behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.checkpoint.bootstrap import ensure_first_run_ready, readiness_marker_path
from support import checkpoint_settings


def _write_marker(checkpoint_dir: Path, recorded_size: int) -> Path:
    marker_path = readiness_marker_path(checkpoint_dir)
    marker_path.write_text(
        json.dumps({"schema_version": 1, "model_pth_size_bytes": recorded_size}),
        encoding="utf-8",
    )
    return marker_path


class BootstrapTests(unittest.TestCase):
    """Verify the fast path, stale-marker detection, single load, and repair."""

    def test_current_marker_loads_once_without_preparation(self) -> None:
        """A marker whose recorded size matches model.pth skips preparation."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            (checkpoint_dir / "model.pth").write_bytes(b"prepared")
            _write_marker(checkpoint_dir, recorded_size=len(b"prepared"))
            loader = MagicMock(return_value="model")

            with patch("src.checkpoint.bootstrap.prepare_checkpoint") as prepare_mock:
                result, loaded = ensure_first_run_ready(checkpoint_dir, checkpoint_settings(), loader)

            self.assertEqual(result.action, "ready")
            self.assertEqual(loaded, "model")
            prepare_mock.assert_not_called()
            loader.assert_called_once_with(checkpoint_dir.resolve())

    def test_first_run_loads_checkpoint_exactly_once(self) -> None:
        """
        Regression: the first run used to strict-load model.pth for
        validation, discard it, and load it again for inference.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            (checkpoint_dir / "model.pth").write_bytes(b"prepared")
            loader = MagicMock(return_value="model")

            with patch("src.checkpoint.bootstrap.prepare_checkpoint", return_value=object()):
                result, loaded = ensure_first_run_ready(checkpoint_dir, checkpoint_settings(), loader)

            self.assertEqual(result.action, "prepared")
            self.assertEqual(loaded, "model")
            loader.assert_called_once()
            self.assertTrue(readiness_marker_path(checkpoint_dir).is_file())

    def test_failed_strict_load_does_not_mark_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            (checkpoint_dir / "model.pth").write_bytes(b"prepared")
            loader = MagicMock(side_effect=RuntimeError("missing keys"))

            with patch("src.checkpoint.bootstrap.prepare_checkpoint", return_value=object()):
                with self.assertRaises(RuntimeError):
                    ensure_first_run_ready(checkpoint_dir, checkpoint_settings(), loader)

            self.assertFalse(readiness_marker_path(checkpoint_dir).exists())

    def test_resized_checkpoint_invalidates_marker(self) -> None:
        """A truncated or replaced model.pth forces full preparation again."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            (checkpoint_dir / "model.pth").write_bytes(b"truncated")
            _write_marker(checkpoint_dir, recorded_size=123_456)

            with patch("src.checkpoint.bootstrap.prepare_checkpoint", return_value=object()) as prepare_mock:
                result, _loaded = ensure_first_run_ready(
                    checkpoint_dir,
                    checkpoint_settings(),
                    MagicMock(return_value="model"),
                )

            self.assertEqual(result.action, "prepared")
            prepare_mock.assert_called_once()
            marker = json.loads(readiness_marker_path(checkpoint_dir).read_text(encoding="utf-8"))
            self.assertEqual(marker["model_pth_size_bytes"], len(b"truncated"))

    def test_missing_checkpoint_invalidates_marker(self) -> None:
        """A marker without model.pth is stale, not a reason to crash later."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            _write_marker(checkpoint_dir, recorded_size=8)

            def fake_prepare(**_kwargs: object) -> object:
                (checkpoint_dir / "model.pth").write_bytes(b"prepared")
                return object()

            with patch("src.checkpoint.bootstrap.prepare_checkpoint", side_effect=fake_prepare):
                result, _loaded = ensure_first_run_ready(
                    checkpoint_dir,
                    checkpoint_settings(),
                    MagicMock(return_value="model"),
                )

            self.assertEqual(result.action, "prepared")

    def test_repair_removes_marker_and_runs_preparation(self) -> None:
        """Repair explicitly invalidates the marker and rebuilds readiness."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            (checkpoint_dir / "model.pth").write_bytes(b"prepared")
            marker_path = _write_marker(checkpoint_dir, recorded_size=len(b"prepared"))

            preparation = object()
            with patch("src.checkpoint.bootstrap.prepare_checkpoint", return_value=preparation):
                result, _loaded = ensure_first_run_ready(
                    checkpoint_dir,
                    checkpoint_settings(),
                    MagicMock(return_value="model"),
                    force_repair=True,
                )

            self.assertEqual(result.action, "prepared")
            self.assertIs(result.preparation, preparation)
            self.assertTrue(marker_path.is_file())


if __name__ == "__main__":
    unittest.main()
