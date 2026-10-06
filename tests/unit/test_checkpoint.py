"""Checkpoint acquisition and conversion against a real local HTTP server."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from src.checkpoint.conversion import convert_checkpoint, read_safetensors_file
from src.checkpoint.download import DownloadSpec, ensure_checkpoint_files
from src.checkpoint.orchestration import ensure_converted_checkpoint
from support import checkpoint_settings


class FixtureServer:
    """A threaded HTTP server over one directory; can fail the next N GETs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.failures_remaining = 0
        self.get_count = 0
        server = self

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, directory=str(root), **kwargs)  # type: ignore[arg-type]

            def do_GET(self) -> None:  # noqa: N802 (http.server naming)
                server.get_count += 1
                if server.failures_remaining > 0:
                    server.failures_remaining -= 1
                    self.send_error(503, "temporarily unavailable")
                    return
                super().do_GET()

            def log_message(self, format: str, *args: object) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}/{name}"

    def spec(self, name: str, payload: bytes | None = None) -> DownloadSpec:
        data = payload if payload is not None else (self.root / name).read_bytes()
        return DownloadSpec(name, self.url(name), len(data), expected_sha256=hashlib.sha256(data).hexdigest())


@pytest.fixture
def server(tmp_path: Path) -> Iterator[FixtureServer]:
    root = tmp_path / "server"
    root.mkdir()
    fixture = FixtureServer(root)
    fixture.thread.start()
    yield fixture
    fixture.httpd.shutdown()
    fixture.httpd.server_close()
    fixture.thread.join()


# =============================================================================
# Download
# =============================================================================


def test_only_missing_or_corrupted_files_are_downloaded(server: FixtureServer, tmp_path: Path) -> None:
    (server.root / "first.json").write_bytes(b'{"first": 1}\n')
    (server.root / "second.json").write_bytes(b'{"second": 2}\n')
    specs = (server.spec("first.json"), server.spec("second.json"))
    checkpoint = tmp_path / "checkpoint"

    first = ensure_checkpoint_files(checkpoint, checkpoint_settings(), specifications=specs)
    second = ensure_checkpoint_files(checkpoint, checkpoint_settings(), specifications=specs)
    (checkpoint / "second.json").write_bytes(b"corrupted")
    third = ensure_checkpoint_files(checkpoint, checkpoint_settings(), specifications=specs)

    assert [result.action for result in first] == ["downloaded", "downloaded"]
    assert [result.action for result in second] == ["reused", "reused"]
    assert [result.action for result in third] == ["reused", "downloaded"]
    assert (checkpoint / "second.json").read_bytes() == b'{"second": 2}\n'
    manifest = json.loads((checkpoint / "download_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"first.json", "second.json"}


def test_transient_server_error_is_retried(server: FixtureServer, tmp_path: Path) -> None:
    (server.root / "weights.bin").write_bytes(bytes(range(256)) * 8)
    server.failures_remaining = 1

    results = ensure_checkpoint_files(
        tmp_path / "checkpoint",
        checkpoint_settings(),
        specifications=(server.spec("weights.bin"),),
    )

    assert results[0].action == "downloaded"
    assert server.get_count == 2


def test_persistent_server_error_fails_without_partial_file(server: FixtureServer, tmp_path: Path) -> None:
    (server.root / "weights.bin").write_bytes(b"payload")
    server.failures_remaining = 10
    checkpoint = tmp_path / "checkpoint"

    with pytest.raises(OSError):
        ensure_checkpoint_files(checkpoint, checkpoint_settings(), specifications=(server.spec("weights.bin"),))

    assert server.get_count == checkpoint_settings().download_attempts
    assert not (checkpoint / "weights.bin").exists()
    assert list(checkpoint.glob("*.part")) == []


def test_content_that_does_not_match_the_pin_is_rejected(server: FixtureServer, tmp_path: Path) -> None:
    """A tampered upstream file must never replace the local copy."""

    (server.root / "config.json").write_bytes(b'{"tampered": true}')
    pinned = server.spec("config.json", payload=b'{"genuine": true}')
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(b"previous")

    with pytest.raises(ValueError, match="Remote size changed|identity mismatch"):
        ensure_checkpoint_files(checkpoint, checkpoint_settings(), specifications=(pinned,))

    assert (checkpoint / "config.json").read_bytes() == b"previous"


# =============================================================================
# Conversion
# =============================================================================


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header: dict[str, object] = {}
    payloads: list[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        payload = tensor.numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(payload)]}
        payloads.append(payload)
        offset += len(payload)
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes + b"".join(payloads))


MINIMAL_CONFIG = {
    "architectures": ["ParakeetForTDT"],
    "model_type": "parakeet_tdt",
    "vocab_size": 2,
    "durations": [0, 1],
    "encoder_config": {"num_hidden_layers": 0},
}

REQUIRED_TENSORS = {
    "encoder.subsampling.layers.0.weight": torch.ones(1),
    "encoder.subsampling.linear.weight": torch.ones(1),
    "encoder_projector.weight": torch.ones(1),
    "decoder.embedding.weight": torch.ones(2, 1),
    "decoder.lstm.weight_ih_l0": torch.ones(1),
    "decoder.lstm.weight_hh_l0": torch.ones(1),
    "decoder.decoder_projector.weight": torch.ones(1),
    "joint.head.weight": torch.ones(4, 1),
}


def test_conversion_round_trips_every_tensor(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    tensors = {name: tensor * (index + 1) for index, (name, tensor) in enumerate(REQUIRED_TENSORS.items())}
    write_safetensors(tmp_path / "model.safetensors", tensors)

    result = convert_checkpoint(tmp_path, tmp_path / "model.pth")
    bundle = torch.load(tmp_path / "model.pth", weights_only=True)

    assert result.tensor_count == len(tensors)
    assert set(bundle["state_dict"]) == set(tensors)
    for name, expected in tensors.items():
        assert torch.equal(bundle["state_dict"][name], expected)


def test_conversion_rejects_missing_keys_and_non_finite_weights(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    tensors = dict(REQUIRED_TENSORS)
    tensors.pop("joint.head.weight")
    tensors["decoder.embedding.weight"] = torch.tensor([[float("nan")], [1.0]])
    write_safetensors(tmp_path / "model.safetensors", tensors)

    with pytest.raises(ValueError, match="missing=.*joint.head.weight.*nan=.*decoder.embedding.weight"):
        convert_checkpoint(tmp_path, tmp_path / "model.pth")


def test_payload_size_must_match_shape_and_dtype(tmp_path: Path) -> None:
    payload = torch.ones(4).numpy().tobytes()
    header = json.dumps({"w": {"dtype": "F32", "shape": [3], "data_offsets": [0, len(payload)]}}).encode("utf-8")
    path = tmp_path / "model.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)

    with pytest.raises(ValueError, match="does not match shape"):
        read_safetensors_file(path)


def test_legacy_model_pth_gains_conversion_manifest(tmp_path: Path) -> None:
    config_bytes = json.dumps(MINIMAL_CONFIG).encode("utf-8")
    safetensors_bytes = b"synthetic-safetensors-source"
    (tmp_path / "config.json").write_bytes(config_bytes)
    (tmp_path / "model.safetensors").write_bytes(safetensors_bytes)
    torch.save(
        {"state_dict": {"weight": torch.tensor([1.0, 2.0])}, "config": MINIMAL_CONFIG, "metadata": {"total_parameters": 2}},
        tmp_path / "model.pth",
    )
    download_manifest = {
        "schema_version": 1,
        "files": {
            name: {
                "url": f"fixture://{name}",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "remote_content_length": len(data),
                "remote_etag": "fixture",
            }
            for name, data in (("config.json", config_bytes), ("model.safetensors", safetensors_bytes))
        },
    }
    (tmp_path / "download_manifest.json").write_text(json.dumps(download_manifest), encoding="utf-8")

    first = ensure_converted_checkpoint(tmp_path)
    second = ensure_converted_checkpoint(tmp_path)

    assert first.action == "bootstrapped"
    assert second.action == "reused"
    assert (tmp_path / "conversion_manifest.json").is_file()


def test_converting_model_pth_removes_files_derived_from_the_old_one(tmp_path: Path) -> None:
    """
    Review finding: a replaced model.pth of the same size left the old float16
    file current. A conversion now removes every derived file; reusing an
    unchanged model.pth keeps them.
    """

    from src.checkpoint.derived import HALF_ENCODER_FILENAME, HALF_ENCODER_MANIFEST

    config_bytes = json.dumps(MINIMAL_CONFIG).encode("utf-8")
    (tmp_path / "config.json").write_bytes(config_bytes)
    write_safetensors(tmp_path / "model.safetensors", REQUIRED_TENSORS)
    safetensors_bytes = (tmp_path / "model.safetensors").read_bytes()
    download_manifest = {
        "schema_version": 1,
        "files": {
            name: {
                "url": f"fixture://{name}",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "remote_content_length": len(data),
                "remote_etag": "fixture",
            }
            for name, data in (("config.json", config_bytes), ("model.safetensors", safetensors_bytes))
        },
    }
    (tmp_path / "download_manifest.json").write_text(json.dumps(download_manifest), encoding="utf-8")
    for name in (HALF_ENCODER_FILENAME, HALF_ENCODER_MANIFEST):
        (tmp_path / name).write_text("built from an older model.pth", encoding="utf-8")

    converted = ensure_converted_checkpoint(tmp_path)

    assert converted.action == "converted"
    assert not (tmp_path / HALF_ENCODER_FILENAME).exists()
    assert not (tmp_path / HALF_ENCODER_MANIFEST).exists()

    for name in (HALF_ENCODER_FILENAME, HALF_ENCODER_MANIFEST):
        (tmp_path / name).write_text("built from the current model.pth", encoding="utf-8")
    reused = ensure_converted_checkpoint(tmp_path)

    assert reused.action == "reused"
    assert (tmp_path / HALF_ENCODER_FILENAME).exists()
