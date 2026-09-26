"""Regression tests for first-run readiness marker behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    """Verify the fast path, stale-marker detection, and explicit repair."""

    def test_current_marker_skips_checkpoint_preparation(self) -> None:
        """A marker whose recorded size matches model.pth returns immediately."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            (checkpoint_dir / "model.pth").write_bytes(b"prepared")
            _write_marker(checkpoint_dir, recorded_size=len(b"prepared"))

            with patch("src.checkpoint.bootstrap.prepare_checkpoint") as prepare_mock:
                result = ensure_first_run_ready(checkpoint_dir, checkpoint_settings())

            self.assertEqual(result.action, "ready")
            prepare_mock.assert_not_called()

    def test_resized_checkpoint_invalidates_marker(self) -> None:
        """A truncated or replaced model.pth forces full preparation again."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            (checkpoint_dir / "model.pth").write_bytes(b"truncated")
            _write_marker(checkpoint_dir, recorded_size=123_456)

            with (
                patch("src.checkpoint.bootstrap.prepare_checkpoint", return_value=object()) as prepare_mock,
                patch("src.checkpoint.bootstrap.load_model", return_value=(object(), object(), {})),
            ):
                result = ensure_first_run_ready(checkpoint_dir, checkpoint_settings())

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

            with (
                patch("src.checkpoint.bootstrap.prepare_checkpoint", side_effect=fake_prepare),
                patch("src.checkpoint.bootstrap.load_model", return_value=(object(), object(), {})),
            ):
                result = ensure_first_run_ready(checkpoint_dir, checkpoint_settings())

            self.assertEqual(result.action, "prepared")

    def test_validation_load_stays_on_cpu(self) -> None:
        """The readiness strict-load must not claim accelerator memory."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            (checkpoint_dir / "model.pth").write_bytes(b"prepared")

            with (
                patch("src.checkpoint.bootstrap.prepare_checkpoint", return_value=object()),
                patch(
                    "src.checkpoint.bootstrap.load_model",
                    return_value=(object(), object(), {}),
                ) as load_mock,
            ):
                ensure_first_run_ready(checkpoint_dir, checkpoint_settings())

            self.assertEqual(load_mock.call_args.kwargs["device"].type, "cpu")

    def test_repair_removes_marker_and_runs_preparation(self) -> None:
        """Repair explicitly invalidates the marker and rebuilds readiness."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            (checkpoint_dir / "model.pth").write_bytes(b"prepared")
            marker_path = _write_marker(checkpoint_dir, recorded_size=len(b"prepared"))

            preparation = object()
            with (
                patch("src.checkpoint.bootstrap.prepare_checkpoint", return_value=preparation),
                patch("src.checkpoint.bootstrap.load_model", return_value=(object(), object(), {})),
            ):
                result = ensure_first_run_ready(
                    checkpoint_dir,
                    checkpoint_settings(),
                    force_repair=True,
                )

            self.assertEqual(result.action, "prepared")
            self.assertIs(result.preparation, preparation)
            self.assertTrue(marker_path.is_file())


if __name__ == "__main__":
    unittest.main()
