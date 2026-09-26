"""The public command: arguments, discovery, CSV rendering, and progress output."""

from __future__ import annotations

import io
from contextlib import redirect_stderr
from pathlib import Path

import pytest
import torch

from src.commands.inference import (
    CSV_COLUMNS,
    csv_rows,
    discover_audio_files,
    parse_arguments,
    persist_transcriptions,
    render_transcription_csv,
)
from src.commands.reporting import ProgressReporter
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


def run_result(paths: list[Path]) -> OfflineRunResult:
    files = tuple(
        OfflineFileResult(path=path, duration_seconds=1.0, transcript="text", status=FileStatus.OK, chunks=())
        for path in paths
    )
    plan = ExecutionPlan(
        metadata=(),
        items=(),
        batches=(),
        target_sample_rate=16000,
        max_feature_frames=1,
        max_batch_feature_frames=1,
    )
    return OfflineRunResult(
        files=files,
        plan=plan,
        elapsed_seconds=1.0,
        peak_memory={},
        media_decode_seconds=0.0,
        feature_seconds=0.0,
        generation_seconds=0.0,
    )


# =============================================================================
# Arguments
# =============================================================================


def test_transcribe_is_the_only_argument() -> None:
    assert vars(parse_arguments(["--transcribe", "docs"])) == {"transcribe": Path("docs")}


@pytest.mark.parametrize("argv", [[], ["--transcribe", "docs", "--repair"], ["--batch-size", "4"]])
def test_other_arguments_are_rejected(argv: list[str]) -> None:
    with redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        parse_arguments(argv)


# =============================================================================
# Discovery
# =============================================================================


def test_single_file_is_probed(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    write_float_wav(path, torch.zeros(1600), 16000)

    discovered = discover_audio_files(path, extensions=(), recursive=False)

    assert [record.path for record in discovered.audio] == [path.resolve()]
    assert discovered.unreadable == ()


def test_extension_filter_limits_candidates(tmp_path: Path) -> None:
    write_float_wav(tmp_path / "a.wav", torch.zeros(1600), 16000)
    write_float_wav(tmp_path / "b.data", torch.zeros(1600), 16000)

    discovered = discover_audio_files(tmp_path, extensions=(".wav",), recursive=False)

    assert [record.path.name for record in discovered.audio] == ["a.wav"]


def test_recursive_discovery_is_sorted_and_deterministic(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    write_float_wav(tmp_path / "sub" / "c.wav", torch.zeros(1600), 16000)
    write_float_wav(tmp_path / "b.wav", torch.zeros(1600), 16000)
    write_float_wav(tmp_path / "a.wav", torch.zeros(1600), 16000)

    flat = discover_audio_files(tmp_path, extensions=(), recursive=False)
    deep = discover_audio_files(tmp_path, extensions=(), recursive=True)

    assert [record.path.name for record in flat.audio] == ["a.wav", "b.wav"]
    assert [record.path.name for record in deep.audio] == ["a.wav", "b.wav", "c.wav"]


def test_missing_path_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_audio_files(tmp_path / "typo", extensions=(), recursive=False)


def test_folder_without_audio_is_reported(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not audio", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No readable audio"):
        discover_audio_files(tmp_path, extensions=(), recursive=False)


def test_single_unreadable_file_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not audio", encoding="utf-8")

    with pytest.raises(ValueError, match="notes.txt"):
        discover_audio_files(path, extensions=(), recursive=False)


# =============================================================================
# CSV output
# =============================================================================


def test_csv_has_fixed_columns_and_quotes_commas() -> None:
    lines = render_transcription_csv([ROW]).splitlines()

    assert lines[0] == ",".join(CSV_COLUMNS)
    assert lines[1] == 'C:/audio/sample.wav,sample.wav,1.25,ok,"hello, world"'


def test_csv_rows_merge_results_and_unreadable_in_path_order() -> None:
    root = Path("C:/audio")

    rows = csv_rows(run_result([root / "a.wav", root / "c.wav"]), [(root / "b.m4a", "no decoder")])

    assert [row["filename_audio"] for row in rows] == ["a.wav", "b.m4a", "c.wav"]
    assert rows[1]["status"] == FileStatus.UNREADABLE.value
    assert rows[1]["transcription"] == "" and rows[1]["duration_audio"] == ""


def test_persist_writes_primary_location_when_possible(tmp_path: Path) -> None:
    output = tmp_path / "transcriptions.csv"

    written, used_fallback = persist_transcriptions(output, tmp_path / "fallback.csv", [ROW])

    assert (written, used_fallback) == (output, False)
    assert not (tmp_path / "fallback.csv").exists()
    assert list(tmp_path.glob("*.tmp")) == []


# =============================================================================
# Progress reporting
# =============================================================================


def test_progress_is_rate_limited_plain_text_with_final_line() -> None:
    lines: list[str] = []
    now = [0.0]
    reporter = ProgressReporter("Batch", 10.0, emit=lines.append, clock=lambda: now[0])

    for completed in range(1, 101):
        now[0] = completed * 0.5
        reporter(completed, 100)

    assert len(lines) <= 7
    assert lines[-1] == "Batch: 100 / 100, Progress: 100.00 percent, Elapsed: 50.0 s, ETA: 0.0 s"
    assert all(line.isascii() and "#" not in line for line in lines)


def test_eta_uses_the_measured_rate() -> None:
    lines: list[str] = []
    now = [0.0]
    reporter = ProgressReporter("Batch", 0.0, emit=lines.append, clock=lambda: now[0])

    now[0] = 4.0
    reporter(1, 5)

    assert "ETA: 16.0 s" in lines[-1]
    assert "Progress: 20.00 percent" in lines[-1]
