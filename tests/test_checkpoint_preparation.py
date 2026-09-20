"""Regression tests for per-file checkpoint acquisition and conversion freshness."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from src.checkpoint import ensure_converted_checkpoint
from src.conversion import convert_checkpoint
from src.utils.download import DownloadSpec, ensure_checkpoint_files


class _QuietRequestHandler(SimpleHTTPRequestHandler):
    """Serve fixture files without writing request logs to the test terminal."""

    def log_message(self, format: str, *args: object) -> None:
        return


class CheckpointPreparationTests(unittest.TestCase):
    """Verify idempotent per-file download and conversion decisions."""

    def test_only_mismatching_file_is_redownloaded(self) -> None:
        """A corrupted file is fetched again while matching neighbors are reused."""

        with tempfile.TemporaryDirectory() as server_directory:
            with tempfile.TemporaryDirectory() as checkpoint_directory:
                server_root = Path(server_directory)
                checkpoint_root = Path(checkpoint_directory)
                (server_root / "first.json").write_bytes(b'{"first": 1}\n')
                (server_root / "second.json").write_bytes(b'{"second": 2}\n')

                handler = lambda *args, **kwargs: _QuietRequestHandler(
                    *args,
                    directory=server_root,
                    **kwargs,
                )
                server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()

                try:
                    base_url = f"http://127.0.0.1:{server.server_port}"
                    first_payload = (server_root / "first.json").read_bytes()
                    second_payload = (server_root / "second.json").read_bytes()
                    specifications = (
                        DownloadSpec(
                            "first.json",
                            f"{base_url}/first.json",
                            len(first_payload),
                            expected_sha256=hashlib.sha256(first_payload).hexdigest(),
                        ),
                        DownloadSpec(
                            "second.json",
                            f"{base_url}/second.json",
                            len(second_payload),
                            expected_sha256=hashlib.sha256(second_payload).hexdigest(),
                        ),
                    )

                    first_run = ensure_checkpoint_files(
                        checkpoint_root,
                        specifications=specifications,
                    )
                    self.assertEqual(
                        [result.action for result in first_run],
                        ["downloaded", "downloaded"],
                    )

                    second_run = ensure_checkpoint_files(
                        checkpoint_root,
                        specifications=specifications,
                    )
                    self.assertEqual(
                        [result.action for result in second_run],
                        ["reused", "reused"],
                    )

                    (checkpoint_root / "second.json").write_bytes(b"corrupted")
                    third_run = ensure_checkpoint_files(
                        checkpoint_root,
                        specifications=specifications,
                    )
                    self.assertEqual(
                        [result.action for result in third_run],
                        ["reused", "downloaded"],
                    )
                    self.assertEqual(
                        (checkpoint_root / "second.json").read_bytes(),
                        (server_root / "second.json").read_bytes(),
                    )
                finally:
                    server.shutdown()
                    server.server_close()
                    server_thread.join()

    def test_existing_model_pth_bootstraps_conversion_manifest(self) -> None:
        """A valid legacy bundle is retained and gains conversion evidence."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            config = {
                "architectures": ["ParakeetForTDT"],
                "model_type": "parakeet_tdt",
                "vocab_size": 2,
                "durations": [0, 1],
                "encoder_config": {"num_hidden_layers": 0},
            }
            config_bytes = json.dumps(config).encode("utf-8")
            safetensors_bytes = b"synthetic-safetensors-source"
            (checkpoint_dir / "config.json").write_bytes(config_bytes)
            (checkpoint_dir / "model.safetensors").write_bytes(safetensors_bytes)

            state_dict = {"weight": torch.tensor([1.0, 2.0])}
            torch.save(
                {
                    "state_dict": state_dict,
                    "config": config,
                    "metadata": {"total_parameters": 2},
                },
                checkpoint_dir / "model.pth",
            )

            download_manifest = {
                "schema_version": 1,
                "files": {
                    "config.json": {
                        "url": "fixture://config.json",
                        "size_bytes": len(config_bytes),
                        "sha256": hashlib.sha256(config_bytes).hexdigest(),
                        "remote_content_length": len(config_bytes),
                        "remote_etag": "fixture-config",
                    },
                    "model.safetensors": {
                        "url": "fixture://model.safetensors",
                        "size_bytes": len(safetensors_bytes),
                        "sha256": hashlib.sha256(safetensors_bytes).hexdigest(),
                        "remote_content_length": len(safetensors_bytes),
                        "remote_etag": "fixture-weights",
                    },
                },
            }
            (checkpoint_dir / "download_manifest.json").write_text(
                json.dumps(download_manifest),
                encoding="utf-8",
            )

            result = ensure_converted_checkpoint(
                checkpoint_dir,
            )

            self.assertEqual(result.action, "bootstrapped")
            self.assertTrue((checkpoint_dir / "conversion_manifest.json").is_file())
            self.assertTrue((checkpoint_dir / "model.pth").is_file())

    def test_conversion_round_trips_synthetic_safetensors(self) -> None:
        """The internal converter preserves every synthetic tensor value."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            config = {
                "architectures": ["ParakeetForTDT"],
                "model_type": "parakeet_tdt",
                "vocab_size": 2,
                "durations": [0, 1],
                "encoder_config": {"num_hidden_layers": 0},
            }
            (checkpoint_dir / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            tensors = {
                "encoder.subsampling.layers.0.weight": torch.ones(1),
                "encoder.subsampling.linear.weight": torch.ones(1),
                "encoder_projector.weight": torch.ones(1),
                "decoder.embedding.weight": torch.ones(2, 1),
                "decoder.lstm.weight_ih_l0": torch.ones(1),
                "decoder.lstm.weight_hh_l0": torch.ones(1),
                "decoder.decoder_projector.weight": torch.ones(1),
                "joint.head.weight": torch.ones(4, 1),
            }
            header: dict[str, object] = {}
            chunks: list[bytes] = []
            offset = 0
            for name, tensor in tensors.items():
                payload = tensor.numpy().tobytes()
                header[name] = {
                    "dtype": "F32",
                    "shape": list(tensor.shape),
                    "data_offsets": [offset, offset + len(payload)],
                }
                chunks.append(payload)
                offset += len(payload)

            header_bytes = json.dumps(header).encode("utf-8")
            with (checkpoint_dir / "model.safetensors").open("wb") as output_file:
                output_file.write(len(header_bytes).to_bytes(8, "little"))
                output_file.write(header_bytes)
                for chunk in chunks:
                    output_file.write(chunk)

            result = convert_checkpoint(
                checkpoint_dir,
                checkpoint_dir / "model.pth",
            )
            bundle = torch.load(
                checkpoint_dir / "model.pth",
                weights_only=False,
            )

            self.assertEqual(result.tensor_count, len(tensors))
            self.assertEqual(set(bundle["state_dict"]), set(tensors))
            for name, expected in tensors.items():
                self.assertTrue(torch.equal(bundle["state_dict"][name], expected))


if __name__ == "__main__":
    unittest.main()
