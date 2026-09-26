"""
Smoke test of the real primary path: config, readiness marker, full-weight
model load, media decode, planning, generation, merge, and CSV output.

It needs the prepared checkpoint and the local sample audio, neither of which
is committed, so it is skipped where they are absent (for example in CI).
"""

from __future__ import annotations

import csv
import io
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.commands.inference import main
from src.configuration.config import PROJECT_ROOT
from src.configuration.settings import load_settings

SETTINGS = load_settings()
SAMPLE_DIR = PROJECT_ROOT / "docs"
SAMPLES = sorted(SAMPLE_DIR.glob("*.wav"))[:2] if SAMPLE_DIR.is_dir() else []
CHECKPOINT_READY = (SETTINGS.paths.weights_dir / ".ready").is_file()


@unittest.skipUnless(CHECKPOINT_READY and len(SAMPLES) == 2, "prepared checkpoint and docs/*.wav required")
class RealWeightsSmokeTest(unittest.TestCase):
    """One real folder run through the public command."""

    def test_folder_command_writes_ok_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            for sample in SAMPLES:
                shutil.copy(sample, folder / sample.name)
            (folder / "readme.txt").write_text("not audio", encoding="utf-8")

            captured = io.StringIO()
            with redirect_stdout(captured):
                exit_code = main(["--transcribe", str(folder)])

            output = captured.getvalue()
            csv_path = folder / SETTINGS.inference.output_filename
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0, output)
        self.assertIn("Device:", output)
        self.assertIn("Batch: ", output)
        self.assertIn("Skipped unreadable file: readme.txt", output)

        # The non-audio file is reported with its own row, never dropped.
        by_name = {row["filename_audio"]: row for row in rows}
        self.assertEqual(set(by_name), {sample.name for sample in SAMPLES} | {"readme.txt"})
        self.assertEqual(by_name["readme.txt"]["status"], "unreadable")
        for sample in SAMPLES:
            row = by_name[sample.name]
            self.assertEqual(row["status"], "ok")
            self.assertGreater(len(row["transcription"].split()), 3)
        print(output, file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
