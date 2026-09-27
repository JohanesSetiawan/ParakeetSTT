"""Settings validation, crash-safe writes, dated run logs, and device report."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.configuration.settings import DEFAULT_SETTINGS_PATH, Settings, load_settings
from src.runtime.device import describe_runtime, select_device
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
merge_tolerance_feature_frames = 100
untranscribed_gap_seconds = 4.0
gap_silence_rms = 0.001
recovery_start_offsets_feature_frames = []
progress_interval_seconds = 1
"""


def load_toml(tmp_path: Path, text: str) -> Settings:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return load_settings(path, project_root=tmp_path)


# =============================================================================
# Settings
# =============================================================================


def test_repository_config_is_valid() -> None:
    settings = load_settings(DEFAULT_SETTINGS_PATH)

    assert settings.paths.weights_dir.is_absolute()
    assert settings.paths.log_dir.is_absolute()
    assert settings.inference.max_batch_feature_frames >= settings.inference.max_chunk_feature_frames


def test_valid_file_is_normalized(tmp_path: Path) -> None:
    settings = load_toml(tmp_path, VALID_TOML)

    assert settings.paths.weights_dir == (tmp_path / "weights/model").resolve()
    assert settings.logging.level == "INFO"
    assert settings.inference.audio_extensions == (".wav", ".mp3")
    assert settings.inference.overlap_feature_frames == 0
    assert settings.checkpoint.request_timeout_seconds == 30.0


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("overlap_feature_frames = 0", "overlap_feature_frames = 500", "twice"),
        ("overlap_feature_frames = 0", "overlap_feature_frames = -1", "overlap_feature_frames"),
        ("max_batch_feature_frames = 2000", "max_batch_feature_frames = 999", "max_batch_feature_frames"),
        ('output_filename = "out.csv"', 'output_filename = "../out.csv"', "output_filename"),
        ("batch_size = 4", "batch_size = true", "batch_size"),
        ('level = "info"', 'level = "loud"', "logging.level"),
        ("download_attempts = 2", "download_attempts = 0", "download_attempts"),
        ("request_timeout_seconds = 30", "request_timeout_seconds = 0", "request_timeout_seconds"),
        ("max_padding_fraction = 0.3", "max_padding_fraction = 1.5", "max_padding_fraction"),
        ('audio_extensions = ["WAV", ".mp3", "wav"]', 'audio_extensions = ".wav"', "audio_extensions"),
        ("recursive = false", 'recursive = "no"', "recursive"),
        ("recovery_start_offsets_feature_frames = []", "recovery_start_offsets_feature_frames = [5]", "must not exceed"),
        ("recovery_start_offsets_feature_frames = []", "recovery_start_offsets_feature_frames = [0]", "non-zero"),
        ("untranscribed_gap_seconds = 4.0", "untranscribed_gap_seconds = 0", "untranscribed_gap_seconds"),
    ],
)
def test_invalid_values_name_the_offending_key(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_toml(tmp_path, VALID_TOML.replace(old, new))


@pytest.mark.parametrize("section", ["paths", "logging", "checkpoint", "inference"])
def test_every_section_is_required(tmp_path: Path, section: str) -> None:
    with pytest.raises(ValueError, match=rf"\[{section}\]"):
        load_toml(tmp_path, VALID_TOML.replace(f"[{section}]", "[renamed]"))


def test_missing_settings_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "absent.toml")


# =============================================================================
# Atomic writes
# =============================================================================


def test_atomic_json_is_sorted_and_leaves_no_temporary(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "data.json"

    write_json_atomic(path, {"b": 1, "a": 2})

    assert path.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "b": 1\n}\n'
    assert list(path.parent.glob("*.tmp")) == []


def test_failed_write_keeps_previous_content(tmp_path: Path) -> None:
    path = tmp_path / "result.csv"
    path.write_text("previous", encoding="utf-8")

    with patch("src.runtime.filesystem.os.fsync", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            write_text_atomic(path, "new content")

    assert path.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob("*.tmp")) == []


# =============================================================================
# Run logging
# =============================================================================


@pytest.fixture
def isolated_package_logger():
    """Detach whatever file handlers a test attaches to the "src" logger."""

    yield
    package_logger = logging.getLogger("src")
    for handler in list(package_logger.handlers):
        handler.close()
        package_logger.removeHandler(handler)


def test_run_log_is_dated_and_stamped_with_run_id(tmp_path: Path, isolated_package_logger) -> None:
    log_dir = tmp_path / "logs"

    run_id, path = configure_run_logging(log_dir, "INFO")
    logging.getLogger("src.tests").info("first run line")
    second_run_id, second_path = configure_run_logging(log_dir, "INFO")
    logging.getLogger("src.tests").info("second run line")
    for handler in list(logging.getLogger("src").handlers):
        handler.flush()
    text = path.read_text(encoding="utf-8")

    assert path.name == f"log_{datetime.now():%Y-%m-%d}.txt"
    assert second_path == path
    assert f"run={run_id}" in text and f"run={second_run_id}" in text
    # Reconfiguring must not attach a second handler that duplicates lines.
    assert text.count("first run line") == 1
    assert text.count("second run line") == 1


def test_log_file_name_uses_the_given_date() -> None:
    assert log_file_path(Path("logs"), datetime(2026, 1, 5, 23, 59)) == Path("logs") / "log_2026-01-05.txt"


# =============================================================================
# Device report
# =============================================================================


def test_device_report_describes_the_running_process() -> None:
    device = select_device()
    report = describe_runtime(device, torch.float32)

    assert device.type in {"cpu", "cuda", "mps"}
    assert report.precision == "float32"
    assert report.torch_version == torch.__version__
    assert all(line.isascii() for line in report.lines())
