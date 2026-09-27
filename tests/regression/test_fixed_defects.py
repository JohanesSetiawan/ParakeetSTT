"""
One test per defect that was found and fixed.

Each test reproduces the original failure scenario and names the commit that
fixed it. If one of these fails, a fixed bug has come back. Do not weaken a
test here to make a change pass; fix the change instead.
"""

from __future__ import annotations

import errno
import io
import json
import pickle
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile
import torch

from src.audio.features import ParakeetFeatureExtractor
from src.audio.media import CodecUnavailableError, inspect_media, open_media_session
from src.checkpoint.bootstrap import ensure_first_run_ready, readiness_marker_path
from src.checkpoint.conversion import resolve_safetensors_files
from src.checkpoint.download import DownloadSpec, ensure_checkpoint_files
from src.commands.inference import csv_rows, discover_audio_files, main, persist_transcriptions
from src.configuration.config import PROJECT_ROOT, load_config
from src.configuration.settings import load_settings
from src.inference import offline as offline_module
from src.inference.offline import FileStatus, OfflineTranscriber, classify_file
from src.inference.planning import AudioMetadata, WorkItem, build_execution_plan, padded_batch_frames
from src.models.parakeet import ParakeetTDT, load_model
from support import (
    TINY_VOCAB_SIZE,
    ScriptedModel,
    checkpoint_settings,
    inference_settings,
    write_float_wav,
    write_tiny_checkpoint,
)

CPU = torch.device("cpu")
TARGET_RATE = 16000
HOP_LENGTH = 160


def sequential_chunks(path: Path, max_chunk: int, overlap: int):
    """Plan one file and read every chunk through the overlap-reuse path."""

    plan = build_execution_plan(
        [inspect_media(path)],
        target_sample_rate=TARGET_RATE,
        hop_length=HOP_LENGTH,
        max_chunk_feature_frames=max_chunk,
        overlap_feature_frames=overlap,
        batch_size=4,
        max_batch_feature_frames=4 * max_chunk,
        max_padding_fraction=1.0,
    )
    results = []
    with open_media_session(path) as session:
        previous = None
        for item in sorted(plan.items, key=lambda work: work.chunk_index):
            segment, waveform = session.read_sequential_segment(
                item.source_start_frame,
                item.source_end_frame,
                TARGET_RATE,
                previous_source_end_frame=previous[0] if previous else None,
                previous_waveform=previous[1] if previous else None,
            )
            previous = (item.source_end_frame, waveform)
            results.append((item, segment))
    return results


# =============================================================================
# Fixed in 8ce0cc6 (P0 long-form defects)
# =============================================================================


def test_zero_overlap_does_not_accumulate_previous_chunks(tmp_path: Path) -> None:
    """
    tensor[-0:] selects the whole tensor, so with zero overlap every chunk
    prepended all earlier audio: 12001 frames against a 1500-frame budget.
    """

    path = tmp_path / "long.wav"
    write_float_wav(path, torch.sin(torch.arange(120_000, dtype=torch.float32) / 7.0), TARGET_RATE)

    chunks = sequential_chunks(path, max_chunk=100, overlap=0)

    assert len(chunks) > 3
    for item, segment in chunks:
        assert segment.waveform.numel() == item.source_frame_count


def test_ffprobe_packet_count_is_not_used_as_sample_count(tmp_path: Path) -> None:
    """nb_frames counts AAC packets; a 13.82 s M4A was read as 0.014 s."""

    path = tmp_path / "clip.m4a"
    path.write_bytes(b"aac payload")
    probe = {
        "stream": {"sample_rate": "16000", "channels": 1, "duration": "13.820000", "nb_frames": "217"},
        "format": {"duration": "13.820000"},
    }
    with (
        patch("src.audio.media.soundfile.info", side_effect=RuntimeError("unsupported")),
        patch("src.audio.media._run_ffprobe", return_value=probe),
    ):
        metadata = inspect_media(path)

    assert metadata.frame_count == 221_120
    assert metadata.duration_seconds == pytest.approx(13.82)


@pytest.mark.parametrize("samples", [0, 1, 159, 217, 320])
def test_tiny_inputs_produce_finite_features(tiny_configuration, samples: int) -> None:
    """Fewer than two valid frames divided by n - 1 = 0 and fed NaN to the encoder."""

    features, mask = ParakeetFeatureExtractor(tiny_configuration)(
        [torch.full((samples,), 0.1)],
        [TARGET_RATE],
        CPU,
    )

    assert torch.isfinite(features).all()
    assert int(mask.sum()) == samples // HOP_LENGTH


def test_stuck_decoder_is_forced_forward_instead_of_raising(tmp_path: Path) -> None:
    """A joint always emitting a real token with duration 0 used to hit the step bound and raise."""

    torch.manual_seed(0)
    model = ParakeetTDT(load_config(write_tiny_checkpoint(tmp_path))).eval()
    features, mask = ParakeetFeatureExtractor(model.configuration)(
        [torch.randn(16000) * 0.1, torch.randn(8000) * 0.1],
        [TARGET_RATE, TARGET_RATE],
        CPU,
    )

    def stuck_joint(decoder_hidden_states, encoder_hidden_states):
        logits = torch.full((decoder_hidden_states.shape[0], 1, TINY_VOCAB_SIZE + 5), -1e4)
        logits[..., 5] = 1e4
        logits[..., TINY_VOCAB_SIZE] = 1e4
        return logits

    with patch.object(model.joint, "forward", stuck_joint):
        result = model.generate(features, mask)

    lengths = result.encoder_lengths.tolist()
    assert result.forced_advances.tolist() == lengths
    for row, length in enumerate(lengths):
        emitted = [token for token in result.sequences[row, 1:].tolist() if token != 2]
        assert len(emitted) == length * model.max_symbols_per_step


@pytest.mark.parametrize("cores", [1, 2, 3, 7])
def test_chunk_at_exact_budget_boundary_is_not_one_frame_over(cores: int) -> None:
    """A window of exactly N * hop samples needs N + 1 frames: the 1500 vs 1501 bug."""

    max_chunk, overlap = 1500, 50
    total_samples = cores * (max_chunk - 2 * overlap) * HOP_LENGTH
    plan = build_execution_plan(
        [AudioMetadata(Path("a.wav"), TARGET_RATE, 1, total_samples, "WAV")],
        target_sample_rate=TARGET_RATE,
        hop_length=HOP_LENGTH,
        max_chunk_feature_frames=max_chunk,
        overlap_feature_frames=overlap,
        batch_size=8,
        max_batch_feature_frames=3000,
        max_padding_fraction=0.25,
    )

    assert max(item.feature_frames for item in plan.items) <= max_chunk


def test_natural_repetition_is_not_flagged_as_gibberish() -> None:
    """A repeated-token heuristic marked real speech and both long test recordings as gibberish."""

    item = WorkItem(0, Path("a.wav"), 0, 0, 1, 0, 1, 1)
    chunk = offline_module.ChunkResult(
        item=item,
        token_ids=(5, 5, 5, 5, 5),
        durations=(1,) * 5,
        frame_starts=tuple(range(5)),
        frame_ends=tuple(range(1, 6)),
        encoder_length=5,
        input_finite=True,
        silent=False,
        features_finite=True,
        encoder_finite=True,
        forced_advances=0,
    )

    assert classify_file("ha ha ha ha ha", (chunk,)) is FileStatus.OK


def test_end_of_file_chunk_is_not_stretched(tmp_path: Path) -> None:
    """
    A 48 kHz file not divisible by 3 ended one source sample short and the last
    chunk was time-stretched. The last chunk must equal the same interval of the
    whole file resampled in one piece.
    """

    from src.audio.resampling import plan_block, resample_block

    path = tmp_path / "stereo48k.wav"
    samples = np.random.default_rng(7).uniform(-0.5, 0.5, size=(400_001, 2)).astype(np.float32)
    soundfile.write(str(path), samples, 48000, subtype="FLOAT")

    last_item, last_segment = sequential_chunks(path, max_chunk=120, overlap=10)[-1]

    mono = torch.from_numpy(samples.mean(axis=1))
    total = round(len(mono) * TARGET_RATE / 48000)
    plan = plan_block(0, total, 48000, TARGET_RATE)
    block = torch.nn.functional.pad(mono, (-plan.block_start, plan.block_end - len(mono)))
    whole = resample_block(block, plan, 48000, TARGET_RATE)
    expected = whole[last_item.source_start_frame : last_item.source_end_frame]
    assert torch.allclose(last_segment.waveform, expected, atol=1e-6)


def test_csv_rows_carry_the_file_status(tmp_path: Path) -> None:
    """Anomalies were invisible in folder output; every row now states its status."""

    transcriber = OfflineTranscriber(
        ScriptedModel(3, encoder_finite=False).eval(),
        load_config(write_tiny_checkpoint(tmp_path / "checkpoint")),
        inference_settings(),
    )
    path = tmp_path / "clip.wav"
    write_float_wav(path, torch.full((16000,), 0.1), TARGET_RATE)

    rows = csv_rows(transcriber.transcribe([path]))

    assert rows[0]["status"] == FileStatus.NUMERICAL_FAILURE.value


class _Unsafe:
    """Stand-in for an attacker-controlled pickled object."""


def test_tampered_model_pth_cannot_run_code_on_load(tmp_path: Path) -> None:
    """model.pth was loaded with weights_only=False, which executes arbitrary pickles."""

    directory = write_tiny_checkpoint(tmp_path)
    torch.save({"state_dict": {}, "payload": _Unsafe()}, directory / "model.pth")

    with pytest.raises(pickle.UnpicklingError):
        load_model(directory, device=CPU)


@pytest.mark.parametrize("shard_name", ["../outside.safetensors", "sub/inner.safetensors", "C:/abs.safetensors", ""])
def test_index_shard_names_cannot_escape_the_checkpoint_dir(tmp_path: Path, shard_name: str) -> None:
    """Shard names from an external index were joined to the checkpoint path unchecked."""

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": shard_name}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid shard"):
        resolve_safetensors_files(tmp_path)


def test_download_progress_is_per_percent_not_per_block(tmp_path: Path) -> None:
    """Progress printed one line per 1 MB block: about 2400 lines for the weights."""

    import threading
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    root = tmp_path / "server"
    root.mkdir()
    payload = bytes(range(256)) * 40
    (root / "blob.bin").write_bytes(payload)

    class Quiet(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, format, *args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    messages: list[str] = []
    try:
        import hashlib

        ensure_checkpoint_files(
            tmp_path / "checkpoint",
            checkpoint_settings(),
            specifications=(
                DownloadSpec(
                    "blob.bin",
                    f"http://127.0.0.1:{httpd.server_port}/blob.bin",
                    len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                ),
            ),
            progress_callback=messages.append,
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()

    progress_lines = [message for message in messages if "Progress:" in message]
    assert len(progress_lines) <= 101  # 2560 blocks of 4 bytes
    assert "Progress: 100 percent" in progress_lines[-1]


def test_resized_checkpoint_invalidates_the_readiness_marker(tmp_path: Path) -> None:
    """The marker was trusted by existence alone, even after model.pth was truncated."""

    (tmp_path / "model.pth").write_bytes(b"truncated")
    readiness_marker_path(tmp_path).write_text(
        json.dumps({"schema_version": 1, "model_pth_size_bytes": 123_456}),
        encoding="utf-8",
    )

    with patch("src.checkpoint.bootstrap.prepare_checkpoint", return_value=object()) as prepare_mock:
        result, _loaded = ensure_first_run_ready(tmp_path, checkpoint_settings(), MagicMock())

    assert result.action == "prepared"
    prepare_mock.assert_called_once()


def test_out_of_memory_stops_once_without_retry_or_partial_result(tmp_path: Path) -> None:
    """P0 policy: no retry, no smaller batch, no CPU fallback, no partial output."""

    model = ScriptedModel(3).eval()
    model.generate = MagicMock(side_effect=torch.OutOfMemoryError("simulated"))
    transcriber = OfflineTranscriber(model, load_config(write_tiny_checkpoint(tmp_path / "ckpt")), inference_settings())
    paths = []
    for index in range(3):
        path = tmp_path / f"clip_{index}.wav"
        write_float_wav(path, torch.full((8000,), 0.1), TARGET_RATE)
        paths.append(path)

    with pytest.raises(RuntimeError, match="No fallback or retry"):
        transcriber.transcribe(paths)

    assert model.generate.call_count == 1


# =============================================================================
# Fixed in bb2bc96 (code review findings)
# =============================================================================


def test_module_entry_point_writes_failures_to_the_run_log(tmp_path: Path) -> None:
    """Under `python -m`, a __name__ logger is "__main__" and never reached the log file."""

    completed = subprocess.run(
        [sys.executable, "-m", "src.commands.inference", "--transcribe", str(tmp_path / "typo_folder")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 1, completed.stderr
    stdout_lines = completed.stdout.splitlines()
    run_id = next(line.split(": ", 1)[1] for line in stdout_lines if line.startswith("Run id: "))
    log_path = Path(next(line.split(": ", 1)[1] for line in stdout_lines if line.startswith("Log file: ")))
    assert log_path.parent == load_settings().paths.log_dir
    run_lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if f"run={run_id}" in line]
    assert any("run failed" in line for line in run_lines)


def test_unreadable_files_are_listed_not_silently_dropped(tmp_path: Path) -> None:
    write_float_wav(tmp_path / "a.wav", torch.zeros(1600), TARGET_RATE)
    (tmp_path / "notes.txt").write_text("not audio", encoding="utf-8")
    (tmp_path / "transcriptions.csv").write_text("path_audio\n", encoding="utf-8")

    discovered = discover_audio_files(
        tmp_path,
        extensions=(),
        recursive=False,
        excluded_names=frozenset({"transcriptions.csv"}),
    )

    assert [record.path.name for record in discovered.audio] == ["a.wav"]
    assert [path.name for path, _reason in discovered.unreadable] == ["notes.txt"]


def test_missing_ffprobe_is_per_file_but_misconfigured_path_stops(tmp_path: Path, monkeypatch) -> None:
    write_float_wav(tmp_path / "a.wav", torch.zeros(1600), TARGET_RATE)
    (tmp_path / "clip.m4a").write_bytes(b"not decodable by libsndfile")
    monkeypatch.delenv("FFPROBE_BINARY", raising=False)

    with patch("src.audio.media.shutil.which", return_value=None):
        discovered = discover_audio_files(tmp_path, extensions=(), recursive=False)
    assert [path.name for path, _ in discovered.unreadable] == ["clip.m4a"]
    assert "ffprobe was not found" in discovered.unreadable[0][1]

    monkeypatch.setenv("FFPROBE_BINARY", str(tmp_path / "missing" / "ffprobe.exe"))
    with pytest.raises(RuntimeError, match="points to a missing executable"):
        discover_audio_files(tmp_path, extensions=(), recursive=False)


def test_os_open_failure_is_not_mistaken_for_an_unsupported_format() -> None:
    """libsndfile reports "too many open files" like "unknown format"; readable files went to FFmpeg."""

    too_many_files = OSError(errno.EMFILE, "Too many open files")
    with (
        patch("src.audio.media.soundfile.SoundFile", side_effect=RuntimeError("System error")),
        patch("src.audio.media.Path.open", side_effect=too_many_files),
        pytest.raises(OSError),
    ):
        open_media_session(Path("a.wav"))
    with (
        patch("src.audio.media.soundfile.info", side_effect=RuntimeError("System error")),
        patch("src.audio.media.Path.open", side_effect=too_many_files),
        pytest.raises(OSError),
    ):
        inspect_media(Path("a.wav"))


def test_batch_budget_counts_padded_frames() -> None:
    """1333 + 1000 + 667 = 3000 passed a 3000 budget but allocates 3 x 1333 = 3999 frames."""

    metadata = tuple(
        AudioMetadata(Path(f"f{index}.wav"), TARGET_RATE, 1, (frames - 1) * HOP_LENGTH, "WAV")
        for index, frames in enumerate((1333, 1000, 667))
    )
    plan = build_execution_plan(
        metadata,
        target_sample_rate=TARGET_RATE,
        hop_length=HOP_LENGTH,
        max_chunk_feature_frames=1500,
        overlap_feature_frames=0,
        batch_size=8,
        max_batch_feature_frames=3000,
        max_padding_fraction=0.5,
    )

    assert [item.feature_frames for item in plan.items] == [1333, 1000, 667]
    assert len(plan.batches) > 1
    assert all(padded_batch_frames(batch) <= 3000 for batch in plan.batches)


def test_decoder_handles_are_opened_lazily_and_released(tmp_path: Path) -> None:
    """Every file's handle was opened before the first batch and could exhaust the file limit."""

    open_sessions: list[object] = []
    peak = [0]
    original_open = offline_module.open_media_session

    def tracking_open(path: Path):
        session = original_open(path)
        original_close = session.close
        open_sessions.append(session)
        peak[0] = max(peak[0], len(open_sessions))

        def tracking_close() -> None:
            if session in open_sessions:
                open_sessions.remove(session)
            original_close()

        session.close = tracking_close
        return session

    paths = []
    for index in range(6):
        path = tmp_path / f"clip_{index}.wav"
        write_float_wav(path, torch.full((8000,), 0.1), TARGET_RATE)
        paths.append(path)
    transcriber = OfflineTranscriber(
        ScriptedModel(3).eval(),
        load_config(write_tiny_checkpoint(tmp_path / "checkpoint")),
        inference_settings(batch_size=1),
    )
    with patch.object(offline_module, "open_media_session", tracking_open):
        result = transcriber.transcribe(paths)

    assert len(result.files) == 6
    assert peak[0] == 1
    assert open_sessions == []


@pytest.mark.parametrize("samples", [8_000, 12_800, 32_000])
def test_token_at_end_of_audio_is_kept(tmp_path: Path, samples: int) -> None:
    """
    A token on the last encoder frame has its midpoint past the end of the
    audio and was dropped by the half-open ownership bound.

    The scripted model emits one token at the end of every chunk, so a
    multi-chunk file may also keep earlier chunks' tokens; what must hold is
    that the final chunk's end token is present.
    """

    path = tmp_path / "clip.wav"
    write_float_wav(path, torch.full((samples,), 0.1), TARGET_RATE)
    transcriber = OfflineTranscriber(
        ScriptedModel(3, at_last_frame=True).eval(),
        load_config(write_tiny_checkpoint(tmp_path / "checkpoint")),
        inference_settings(),
    )

    file_result = transcriber.transcribe([path]).files[0]
    words = file_result.transcript.split()

    if len(file_result.chunks) == 1:
        assert words == ["a"]
    else:
        assert words and words[-1] == "a"
        assert len(words) <= len(file_result.chunks)


def test_locked_csv_does_not_lose_the_transcripts(tmp_path: Path) -> None:
    """A CSV open in Excel made the final write fail and every transcript was lost."""

    blocked = tmp_path / "transcriptions.csv"
    blocked.mkdir()  # a directory cannot be replaced by a file, like a locked file
    fallback = tmp_path / "logs" / "transcriptions_run.csv"
    row = {"path_audio": "a.wav", "filename_audio": "a.wav", "duration_audio": 1.0, "status": "ok", "transcription": "kept"}

    written, used_fallback = persist_transcriptions(blocked, fallback, [row])

    assert (written, used_fallback) == (fallback, True)
    assert "kept" in fallback.read_text(encoding="utf-8")


def test_both_csv_locations_failing_raises(tmp_path: Path) -> None:
    (tmp_path / "a.csv").mkdir()
    (tmp_path / "b.csv").mkdir()

    with pytest.raises(OSError):
        persist_transcriptions(tmp_path / "a.csv", tmp_path / "b.csv", [])


def test_mistyped_input_fails_before_any_checkpoint_work(tmp_path: Path) -> None:
    """The input was validated only after a possible 2.5 GB download and a model load."""

    with (
        patch("src.commands.inference.ensure_first_run_ready") as bootstrap_mock,
        redirect_stdout(io.StringIO()),
        redirect_stderr(io.StringIO()) as errors,
    ):
        exit_code = main(["--transcribe", str(tmp_path / "typo_folder")])

    assert exit_code == 1
    bootstrap_mock.assert_not_called()
    assert "Input path does not exist" in errors.getvalue()


def test_first_run_loads_the_checkpoint_exactly_once(tmp_path: Path) -> None:
    """The first run strict-loaded model.pth, discarded it, and loaded it again."""

    (tmp_path / "model.pth").write_bytes(b"prepared")
    loader = MagicMock(return_value="model")

    with patch("src.checkpoint.bootstrap.prepare_checkpoint", return_value=object()):
        result, loaded = ensure_first_run_ready(tmp_path, checkpoint_settings(), loader)

    assert (result.action, loaded) == ("prepared", "model")
    loader.assert_called_once()
    assert readiness_marker_path(tmp_path).is_file()


def test_codec_unavailable_is_a_runtime_error_subclass() -> None:
    """Callers that catch RuntimeError for codec problems must keep working."""

    assert issubclass(CodecUnavailableError, RuntimeError)


# =============================================================================
# Fixed by word-level chunk merging and collapse recovery (PR #3)
# =============================================================================


def test_drifted_seam_timestamps_do_not_split_a_word() -> None:
    """
    Chunk 0 placed "smile at one" at 13.52-14.00 s and chunk 1 at 13.74-14.46 s
    (tests/data/librispeech/1272-128104-0004.flac). Token-midpoint ownership
    kept "sm ile at" from one chunk and "ile at one" from the other, giving
    "smile atile at one".
    """

    from src.inference.merging import ChunkWords, TimedToken, group_words, merge_chunks

    pieces = {10: "\u2581sm", 11: "ile", 12: "\u2581at", 13: "\u2581one", 14: "\u2581much"}

    def seam_chunk(timed: list[tuple[int, float]], source, core) -> ChunkWords:
        tokens = [TimedToken(token, round(start * 16000), round((start + 0.08) * 16000)) for token, start in timed]
        return ChunkWords(
            words=group_words(tokens, pieces.get),
            source_start=round(source[0] * 16000),
            source_end=round(source[1] * 16000),
            core_start=round(core[0] * 16000),
            core_end=round(core[1] * 16000),
        )

    earlier = seam_chunk([(10, 13.52), (11, 13.68), (12, 13.84), (13, 14.00)], (0.0, 14.5), (0.0, 14.0))
    later = seam_chunk([(10, 13.74), (11, 13.98), (12, 14.22), (13, 14.46), (14, 14.70)], (13.5, 28.5), (14.0, 28.0))

    merged = merge_chunks([earlier, later], tolerance_samples=16000)

    assert "".join(pieces[token] for token in merged).replace("\u2581", " ").strip() == "smile at one much"


def test_decoder_collapse_inside_a_chunk_is_re_decoded(tmp_path: Path, tiny_configuration) -> None:
    """
    The model skipped 9.5 s of clear speech on one window of a long recording
    (Hugging Face does the same on those samples); the words were lost.
    """

    from support import SteadySpeechModel

    path = tmp_path / "speech.wav"
    write_float_wav(path, torch.full((48_000,), 0.1), TARGET_RATE)
    transcriber = OfflineTranscriber(
        SteadySpeechModel(3, collapse_calls=frozenset({1})).eval(),
        tiny_configuration,
        inference_settings(
            batch_size=1,
            max_chunk_feature_frames=100,
            overlap_feature_frames=5,
            untranscribed_gap_seconds=0.5,
            recovery_start_offsets_feature_frames=(-3, 3),
        ),
    )

    file_result = transcriber.transcribe([path]).files[0]

    assert file_result.chunks[1].recovered
    assert file_result.status is FileStatus.OK
