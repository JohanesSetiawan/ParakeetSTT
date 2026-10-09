"""Readiness gate: fast path, preparation, marker contents, and repair."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import dataclasses

from src.checkpoint.bootstrap import (
    clear_readiness_marker,
    ensure_first_run_ready,
    readiness_marker_path,
)
from src.checkpoint.orchestration import remove_source_weights
from support import checkpoint_settings

PREPARE = "src.checkpoint.bootstrap.prepare_checkpoint"


def write_marker(checkpoint_dir: Path, recorded_size: int) -> Path:
    marker_path = readiness_marker_path(checkpoint_dir)
    marker_path.write_text(
        json.dumps({"schema_version": 1, "model_pth_size_bytes": recorded_size}),
        encoding="utf-8",
    )
    return marker_path


@pytest.fixture
def prepared_checkpoint(tmp_path: Path) -> Path:
    (tmp_path / "model.pth").write_bytes(b"prepared")
    return tmp_path


def test_current_marker_skips_preparation_and_loads(prepared_checkpoint: Path) -> None:
    write_marker(prepared_checkpoint, recorded_size=len(b"prepared"))
    loader = MagicMock(return_value="model")

    with patch(PREPARE) as prepare_mock:
        result, loaded = ensure_first_run_ready(prepared_checkpoint, checkpoint_settings(), loader)

    assert (result.action, loaded) == ("ready", "model")
    prepare_mock.assert_not_called()
    loader.assert_called_once_with(prepared_checkpoint.resolve())


def test_marker_records_checkpoint_size_after_preparation(prepared_checkpoint: Path) -> None:
    with patch(PREPARE, return_value=object()):
        result, _loaded = ensure_first_run_ready(prepared_checkpoint, checkpoint_settings(), MagicMock())

    marker = json.loads(readiness_marker_path(prepared_checkpoint).read_text(encoding="utf-8"))
    assert result.action == "prepared"
    assert marker["schema_version"] == 1
    assert marker["model_pth_size_bytes"] == len(b"prepared")


def test_failed_strict_load_leaves_no_marker(prepared_checkpoint: Path) -> None:
    loader = MagicMock(side_effect=RuntimeError("missing keys"))

    with patch(PREPARE, return_value=object()), pytest.raises(RuntimeError, match="missing keys"):
        ensure_first_run_ready(prepared_checkpoint, checkpoint_settings(), loader)

    assert not readiness_marker_path(prepared_checkpoint).exists()


def test_missing_checkpoint_makes_marker_stale(tmp_path: Path) -> None:
    write_marker(tmp_path, recorded_size=8)

    def fake_prepare(**_kwargs: object) -> object:
        (tmp_path / "model.pth").write_bytes(b"prepared")
        return object()

    with patch(PREPARE, side_effect=fake_prepare):
        result, _loaded = ensure_first_run_ready(tmp_path, checkpoint_settings(), MagicMock())

    assert result.action == "prepared"


@pytest.mark.parametrize("marker_text", ["not json", '{"schema_version": 99, "model_pth_size_bytes": 8}'])
def test_unreadable_or_unknown_marker_triggers_preparation(prepared_checkpoint: Path, marker_text: str) -> None:
    readiness_marker_path(prepared_checkpoint).write_text(marker_text, encoding="utf-8")

    with patch(PREPARE, return_value=object()) as prepare_mock:
        result, _loaded = ensure_first_run_ready(prepared_checkpoint, checkpoint_settings(), MagicMock())

    assert result.action == "prepared"
    prepare_mock.assert_called_once()


def test_repair_and_clear_force_full_preparation(prepared_checkpoint: Path) -> None:
    marker_path = write_marker(prepared_checkpoint, recorded_size=len(b"prepared"))
    preparation = object()

    with patch(PREPARE, return_value=preparation):
        repaired, _loaded = ensure_first_run_ready(
            prepared_checkpoint,
            checkpoint_settings(),
            MagicMock(),
            force_repair=True,
        )
    clear_readiness_marker(prepared_checkpoint)

    assert repaired.action == "prepared"
    assert repaired.preparation is preparation
    assert not marker_path.exists()


# =============================================================================
# Releasing model.safetensors after a proven model.pth
# =============================================================================

SOURCE_BYTES = b"verified safetensors payload"
SOURCE_SHA256 = "a" * 64


def write_converted_checkpoint(directory: Path, converted_from: str = SOURCE_SHA256) -> Path:
    """model.pth, model.safetensors, and the two manifests that tie them together."""

    (directory / "model.pth").write_bytes(b"prepared")
    (directory / "model.safetensors").write_bytes(SOURCE_BYTES)
    (directory / "download_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {"model.safetensors": {"sha256": SOURCE_SHA256, "size_bytes": len(SOURCE_BYTES)}},
            }
        ),
        encoding="utf-8",
    )
    (directory / "conversion_manifest.json").write_text(
        json.dumps({"schema_version": 1, "source_safetensors_sha256": converted_from}),
        encoding="utf-8",
    )
    return directory


def releasing_settings(keep: bool):
    return dataclasses.replace(checkpoint_settings(), keep_safetensors=keep)


def test_safetensors_is_removed_after_a_successful_load(tmp_path: Path) -> None:
    write_converted_checkpoint(tmp_path)
    write_marker(tmp_path, recorded_size=len(b"prepared"))
    messages: list[str] = []

    result, _loaded = ensure_first_run_ready(
        tmp_path,
        releasing_settings(keep=False),
        MagicMock(return_value="model"),
        progress_callback=messages.append,
    )

    assert result.action == "ready"
    assert not (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "model.pth").is_file()
    assert any("Removed model.safetensors" in message for message in messages)


def test_safetensors_is_kept_when_configured(tmp_path: Path) -> None:
    write_converted_checkpoint(tmp_path)
    write_marker(tmp_path, recorded_size=len(b"prepared"))

    ensure_first_run_ready(tmp_path, releasing_settings(keep=True), MagicMock())

    assert (tmp_path / "model.safetensors").read_bytes() == SOURCE_BYTES


def test_safetensors_is_kept_when_the_load_fails(tmp_path: Path) -> None:
    write_converted_checkpoint(tmp_path)
    write_marker(tmp_path, recorded_size=len(b"prepared"))
    loader = MagicMock(side_effect=RuntimeError("missing keys"))

    with pytest.raises(RuntimeError, match="missing keys"):
        ensure_first_run_ready(tmp_path, releasing_settings(keep=False), loader)

    assert (tmp_path / "model.safetensors").is_file()


@pytest.mark.parametrize("mismatch", ["other source", "resized file", "no conversion manifest"])
def test_safetensors_is_kept_without_matching_evidence(tmp_path: Path, mismatch: str) -> None:
    """Never delete a file that model.pth was not provably converted from."""

    write_converted_checkpoint(tmp_path, converted_from="b" * 64 if mismatch == "other source" else SOURCE_SHA256)
    if mismatch == "resized file":
        (tmp_path / "model.safetensors").write_bytes(SOURCE_BYTES + b"x")
    if mismatch == "no conversion manifest":
        (tmp_path / "conversion_manifest.json").unlink()

    assert remove_source_weights(tmp_path) == 0
    assert (tmp_path / "model.safetensors").is_file()


def test_a_failed_removal_is_a_warning_not_a_failure(tmp_path: Path) -> None:
    write_converted_checkpoint(tmp_path)
    write_marker(tmp_path, recorded_size=len(b"prepared"))
    messages: list[str] = []

    with patch("src.checkpoint.bootstrap.remove_source_weights", side_effect=PermissionError("in use")):
        result, loaded = ensure_first_run_ready(
            tmp_path,
            releasing_settings(keep=False),
            MagicMock(return_value="model"),
            progress_callback=messages.append,
        )

    assert (result.action, loaded) == ("ready", "model")
    assert any(message.startswith("Warning: could not remove model.safetensors") for message in messages)
