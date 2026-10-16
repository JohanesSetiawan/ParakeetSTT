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
metrics_dir = "metrics"

[logging]
level = "info"
terminal_details = false

[checkpoint]
request_timeout_seconds = 30
download_attempts = 2
retry_backoff_seconds = 0.5
stream_block_bytes = 4096
keep_safetensors = false

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
max_open_files = 4
encoder_precision = "float16"
cuda_graphs = true
float32_matmul_precision = "high"
float16_accumulation = true
decode_workers = 3

[memory]
cap_to_free_memory = true
auto_batch_budget = true
reserve_mib = 256
auto_batch_memory_share = 0.7
auto_batch_max_rows = 64
unmeasured_batch_rows = 16

[benchmark]
warmup_rounds = 0
measured_rounds = 2
git_timeout_seconds = 15.0
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
    # The shipped configuration adapts to the machine.
    assert set(settings.inference.unresolved_fields()) == {
        "batch_size",
        "max_batch_feature_frames",
        "encoder_precision",
        "float16_accumulation",
        "decode_workers",
    }


def test_valid_file_is_normalized(tmp_path: Path) -> None:
    settings = load_toml(tmp_path, VALID_TOML)

    assert settings.paths.weights_dir == (tmp_path / "weights/model").resolve()
    assert settings.logging.level == "INFO"
    assert settings.logging.terminal_details is False
    assert settings.checkpoint.keep_safetensors is False
    assert settings.inference.audio_extensions == (".wav", ".mp3")
    assert settings.inference.overlap_feature_frames == 0
    assert settings.checkpoint.request_timeout_seconds == 30.0
    assert settings.inference.max_open_files == 4
    assert settings.inference.encoder_precision == "float16"
    assert settings.inference.float16_accumulation is True
    assert settings.inference.decode_workers == 3
    assert settings.memory.cap_to_free_memory is True
    assert settings.memory.auto_batch_budget is True
    assert settings.memory.reserve_mib == 256


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
        ("measured_rounds = 2", "measured_rounds = 0", "measured_rounds"),
        ("warmup_rounds = 0", "warmup_rounds = -1", "warmup_rounds"),
        ("git_timeout_seconds = 15.0", "git_timeout_seconds = 0", "git_timeout_seconds"),
        ("max_open_files = 4", "max_open_files = 0", "max_open_files"),
        ('encoder_precision = "float16"', 'encoder_precision = "bfloat16"', "encoder_precision"),
        ("batch_size = 4", 'batch_size = "automatic"', "batch_size"),
        ("decode_workers = 3", 'decode_workers = "Auto"', "decode_workers"),
        ("float16_accumulation = true", 'float16_accumulation = "yes"', "float16_accumulation"),
        ("cuda_graphs = true", 'cuda_graphs = "on"', "cuda_graphs"),
        ('float32_matmul_precision = "high"', 'float32_matmul_precision = "medium"', "float32_matmul_precision"),
        ("float16_accumulation = true", "float16_accumulation = 1", "float16_accumulation"),
        ("max_batch_feature_frames = 2000", 'max_batch_feature_frames = "max"', "max_batch_feature_frames"),
        ("decode_workers = 3", "decode_workers = 0", "decode_workers"),
        ("terminal_details = false", "terminal_details = 0", "terminal_details"),
        ("keep_safetensors = false", 'keep_safetensors = "no"', "keep_safetensors"),
        ("reserve_mib = 256", "reserve_mib = -1", "reserve_mib"),
        ("auto_batch_memory_share = 0.7", "auto_batch_memory_share = 0", "auto_batch_memory_share"),
        ("auto_batch_memory_share = 0.7", "auto_batch_memory_share = 1.5", "auto_batch_memory_share"),
        ("auto_batch_memory_share = 0.7", "auto_batch_memory_share = nan", "finite"),
        ("gap_silence_rms = 0.001", "gap_silence_rms = inf", "finite"),
        ("auto_batch_max_rows = 64", "auto_batch_max_rows = 0", "auto_batch_max_rows"),
        ("unmeasured_batch_rows = 16", 'unmeasured_batch_rows = "auto"', "unmeasured_batch_rows"),
        ("auto_batch_budget = true", "auto_batch_budget = 1", "auto_batch_budget"),
        ("cap_to_free_memory = true", 'cap_to_free_memory = "yes"', "cap_to_free_memory"),
    ],
)
def test_invalid_values_name_the_offending_key(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_toml(tmp_path, VALID_TOML.replace(old, new))


@pytest.mark.parametrize("section", ["paths", "logging", "checkpoint", "inference", "memory", "benchmark"])
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


def test_quiet_line_reporter_only_logs(caplog) -> None:
    import io
    import logging

    from src.commands.reporting import line_reporter

    stream = io.StringIO()
    report = line_reporter(logging.getLogger("src.commands.test"), stream, echo=False)
    with caplog.at_level(logging.INFO, logger="src.commands.test"):
        report("Encoder precision: float16")

    assert stream.getvalue() == ""
    assert [record.getMessage() for record in caplog.records] == ["Encoder precision: float16"]


def test_runtime_summary_names_everything_on_one_line() -> None:
    from src.runtime.device import RuntimeReport

    cpu = RuntimeReport("cpu", "none", 1, "x86_64", "not applicable", "float32", "2.x", "not available", "3.13")
    gpu = RuntimeReport("cuda:0", "CUDA", 2, "Some GPU", "8.6", "float16", "2.x", "13.x", "3.13")

    assert cpu.summary("float32") == (
        "Running on: x86_64 (CPU, 1 device), float32, PyTorch 2.x, CUDA not available"
    )
    assert gpu.summary("float16 encoder") == (
        "Running on: Some GPU (CUDA, 2 devices), float16 encoder, PyTorch 2.x, CUDA 13.x"
    )


def test_line_reporter_prints_and_logs_each_line(caplog) -> None:
    import io
    import logging

    from src.commands.reporting import line_reporter

    stream = io.StringIO()
    report = line_reporter(logging.getLogger("src.commands.test"), stream)
    with caplog.at_level(logging.INFO, logger="src.commands.test"):
        report("Device: cpu")

    assert stream.getvalue() == "Device: cpu\n"
    assert [record.getMessage() for record in caplog.records] == ["Device: cpu"]


def test_one_checkpoint_file_name_everywhere(tmp_path: Path) -> None:
    from src.checkpoint import bootstrap
    from src.configuration.config import CHECKPOINT_FILENAME

    assert bootstrap.CHECKPOINT_FILENAME is CHECKPOINT_FILENAME
    for module in ("src/checkpoint/derived.py", "src/checkpoint/orchestration.py", "src/checkpoint/bootstrap.py"):
        source = (Path(__file__).resolve().parents[2] / module).read_text(encoding="utf-8")
        assert '"model.pth"' not in source, module


# =============================================================================
# Float16 accumulation
# =============================================================================


def test_float16_accumulation_follows_the_setting_and_the_encoder_dtype() -> None:
    from src.runtime.device import apply_float16_accumulation

    backend = torch.backends.cuda.matmul
    if not hasattr(backend, "allow_fp16_accumulation"):
        pytest.skip("this PyTorch build has no float16 accumulation switch")
    original = backend.allow_fp16_accumulation
    try:
        assert apply_float16_accumulation(True, torch.float16) == "on"
        assert backend.allow_fp16_accumulation is True
        assert apply_float16_accumulation(True, torch.float32) == "off (the encoder runs in float32)"
        assert backend.allow_fp16_accumulation is False
        apply_float16_accumulation(True, torch.float16)
        assert apply_float16_accumulation(False, torch.float16) == "off"
        assert backend.allow_fp16_accumulation is False
    finally:
        backend.allow_fp16_accumulation = original


def test_float16_accumulation_reports_an_unsupported_build(caplog) -> None:
    from types import SimpleNamespace

    from src.runtime import device as device_module

    with patch.object(device_module.torch.backends.cuda, "matmul", SimpleNamespace()):
        with caplog.at_level(logging.WARNING, logger="src.runtime.device"):
            state = device_module.apply_float16_accumulation(True, torch.float16)

    assert state == "off (not supported by this PyTorch build)"
    assert "does not support" in caplog.text


# =============================================================================
# "auto" settings and device decisions
# =============================================================================


def test_every_auto_setting_parses_and_must_be_resolved(tmp_path: Path) -> None:
    text = (
        VALID_TOML.replace("batch_size = 4", 'batch_size = "auto"')
        .replace("max_batch_feature_frames = 2000", 'max_batch_feature_frames = "auto"')
        .replace('encoder_precision = "float16"', 'encoder_precision = "auto"')
        .replace("float16_accumulation = true", 'float16_accumulation = "auto"')
        .replace("decode_workers = 3", 'decode_workers = "auto"')
    )
    inference = load_toml(tmp_path, text).inference

    assert (inference.batch_size, inference.max_batch_feature_frames) == (None, None)
    assert (inference.float16_accumulation, inference.decode_workers) == (None, None)
    assert inference.encoder_precision == "auto"
    with pytest.raises(ValueError, match="batch_size.*still 'auto'"):
        inference.require_resolved()


def test_explicit_settings_are_resolved_as_written(tmp_path: Path) -> None:
    inference = load_toml(tmp_path, VALID_TOML).inference

    inference.require_resolved()
    assert inference.unresolved_fields() == ()


@pytest.mark.parametrize(("cpus", "workers"), [(1, 1), (4, 1), (8, 2), (16, 4), (64, 8), (128, 8)])
def test_auto_decode_workers_follow_the_cpu_count(cpus: int, workers: int) -> None:
    from src.runtime.device import default_decode_workers

    assert default_decode_workers(cpus) == workers


@pytest.mark.parametrize(
    ("precision", "capability", "dtype", "text"),
    [
        ("auto", (8, 6), torch.float16, "auto: tensor cores, compute capability 8.6"),
        ("auto", (7, 0), torch.float16, "auto: tensor cores"),
        ("auto", (6, 1), torch.float32, "auto: no tensor cores, compute capability 6.1"),
        ("float16", (6, 1), torch.float16, "no tensor cores"),
        ("float16", (8, 9), torch.float16, "float16 encoder, float32 decoder and joint"),
        ("float32", (8, 6), torch.float32, "float32"),
    ],
)
def test_encoder_precision_on_cuda_follows_tensor_cores(precision, capability, dtype, text) -> None:
    from src.runtime import device as device_module

    with patch.object(device_module, "cuda_capability", return_value=capability):
        resolved, description = device_module.encoder_dtype_for(precision, torch.device("cuda"))

    assert resolved == dtype
    assert text in description


@pytest.mark.parametrize("precision", ["auto", "float16"])
def test_encoder_runs_float32_off_nvidia_gpus(precision: str) -> None:
    from src.runtime.device import encoder_dtype_for

    dtype, description = encoder_dtype_for(precision, torch.device("cpu"))

    assert dtype == torch.float32
    assert "float16 is used on NVIDIA GPUs only, this is cpu" in description


@pytest.mark.parametrize(
    ("capability", "architectures", "supported"),
    [
        ((8, 6), ["sm_75", "sm_80", "sm_86"], True),
        ((8, 9), ["sm_80", "sm_86"], True),  # same major version, newer minor
        ((9, 0), ["sm_86", "compute_86"], True),  # portable code compiles for newer GPUs
        ((6, 1), ["sm_75", "sm_80"], False),  # Pascal on a build that dropped it
        ((12, 0), ["sm_75", "sm_80", "sm_86", "sm_90"], False),
    ],
)
def test_build_support_for_a_gpu_architecture(capability, architectures, supported) -> None:
    from src.runtime import device as device_module

    with (
        patch.object(device_module, "cuda_capability", return_value=capability),
        patch.object(device_module.torch.cuda, "get_arch_list", return_value=architectures),
    ):
        assert device_module.build_supports_device(torch.device("cuda")) is supported


def test_build_support_is_not_a_question_off_cuda() -> None:
    from src.runtime.device import build_supports_device

    assert build_supports_device(torch.device("cpu")) is None


@pytest.mark.parametrize(("accelerator", "expected"), [(None, "cpu"), (torch.device("mps"), "mps"), (torch.device("xpu"), "xpu")])
def test_device_selection_follows_torch_accelerator(accelerator, expected: str) -> None:
    from src.runtime import device as device_module

    with patch.object(device_module.torch.accelerator, "current_accelerator", return_value=accelerator) as current:
        assert device_module.select_device() == torch.device(expected)
    current.assert_called_once_with(check_available=True)


def test_float16_accumulation_is_not_measured_off_cuda() -> None:
    from src.runtime.device import measure_float16_accumulation_speedup

    assert measure_float16_accumulation_speedup(torch.device("cpu"), 64, 256) is None


@pytest.mark.parametrize(
    ("setting", "speedup", "dtype", "enabled", "reason"),
    [
        (True, None, torch.float16, True, ""),
        (False, None, torch.float16, False, ""),
        (None, None, torch.float32, False, "the encoder runs in float32"),
        (None, 1.60, torch.float16, True, "1.60x faster"),
        (None, 1.02, torch.float16, False, "no clear gain"),
        (None, 1.10, torch.float16, False, "no clear gain"),  # below the 25 percent margin
        (None, None, torch.float16, False, "not measurable"),
    ],
)
def test_auto_float16_accumulation_needs_a_clear_measured_gain(setting, speedup, dtype, enabled, reason) -> None:
    from types import SimpleNamespace

    from src.commands import model_loading

    configuration = SimpleNamespace(encoder={"hidden_size": 64, "intermediate_size": 256})
    with patch.object(model_loading, "measure_float16_accumulation_speedup", return_value=speedup) as measure:
        decided, why = model_loading._decide_float16_accumulation(
            setting, torch.device("cuda"), dtype, configuration
        )

    assert decided is enabled
    assert reason in why
    if setting is not None or dtype != torch.float16:
        measure.assert_not_called()


@pytest.mark.parametrize(
    ("name", "tensor_cores"),
    [
        ("NVIDIA GeForce RTX 2060", True),
        ("Tesla T4", True),
        ("NVIDIA GeForce GTX 1650", False),
        ("NVIDIA GeForce GTX 1660 Ti", False),
        ("NVIDIA GeForce MX450", False),
        ("Quadro T2000 with Max-Q Design", False),
        ("NVIDIA T400 4GB", False),
        ("NVIDIA T500", False),
        ("NVIDIA T550 Laptop GPU", False),
    ],
)
def test_turing_cards_without_tensor_cores_are_recognized(name: str, tensor_cores: bool) -> None:
    """Review finding: compute capability 7.5 also covers TU116/TU117 cards without tensor cores."""

    from src.runtime import device as device_module

    with (
        patch.object(device_module, "cuda_capability", return_value=(7, 5)),
        patch.object(device_module.torch.cuda, "get_device_name", return_value=name),
    ):
        assert device_module.has_tensor_cores(torch.device("cuda")) is tensor_cores
        dtype, _description = device_module.encoder_dtype_for("auto", torch.device("cuda"))

    assert dtype == (torch.float16 if tensor_cores else torch.float32)


def test_rocm_gpus_are_not_judged_by_nvidia_rules() -> None:
    """Review finding: ROCm builds list gfx architectures, which made every AMD GPU look unsupported."""

    from src.runtime import device as device_module

    with patch.object(device_module.torch.version, "hip", "6.2"):
        rocm = torch.device("cuda")
        assert device_module.is_nvidia_gpu(rocm) is False
        assert device_module.cuda_capability(rocm) is None
        assert device_module.build_supports_device(rocm) is None
        assert device_module.has_tensor_cores(rocm) is False
        dtype, description = device_module.encoder_dtype_for("auto", rocm)

    assert dtype == torch.float32
    assert "NVIDIA GPUs only" in description

    with patch.object(device_module.torch.version, "hip", "6.2"):
        explicit, explicit_description = device_module.encoder_dtype_for("float16", torch.device("cuda"))
        speedup = device_module.measure_float16_accumulation_speedup(torch.device("cuda"), 64, 256)

    assert explicit == torch.float32 and "this is ROCm" in explicit_description
    assert speedup is None


def test_graph_decoding_is_nvidia_only(tmp_path: Path) -> None:
    from src.configuration.config import load_config
    from src.models import parakeet as parakeet_module
    from support import write_tiny_checkpoint

    model = parakeet_module.ParakeetTDT(load_config(write_tiny_checkpoint(tmp_path)))
    with patch.object(parakeet_module, "is_nvidia_gpu", return_value=False):
        model.enable_graph_decoding(4, 10)

    assert model.graph_decoder is None
