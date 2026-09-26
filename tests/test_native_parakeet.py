"""Regression tests for the standalone native Parakeet runtime."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from src.audio.reader import read_wav
from src.audio.features import build_mel_filter_bank
from src.configuration.config import DEFAULT_WEIGHTS_DIR, ParakeetConfig, load_config
from src.models.attention import Attention
from src.models.parakeet import select_device
from src.text.tokenization import BpeTokenizer


class ParakeetRuntimeTests(unittest.TestCase):
    """Verify standalone runtime boundaries without loading the large checkpoint."""

    def test_config_reads_values_from_json(self) -> None:
        """Runtime fields come from the checkpoint configuration instead of literals."""

        configuration = load_config(DEFAULT_WEIGHTS_DIR)

        self.assertEqual(configuration.blank_token_id, 8192)
        self.assertEqual(configuration.vocab_size, 8193)
        self.assertEqual(configuration.encoder["num_hidden_layers"], 24)
        self.assertEqual(configuration.processor["feature_extractor"]["feature_size"], 128)

    def test_read_float32_wav(self) -> None:
        """Native RIFF parsing returns the original float waveform and sample rate."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "float.wav"
            waveform = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)
            payload = struct.pack("<3f", *waveform.tolist())
            fmt_chunk = struct.pack("<HHIIHH", 3, 1, 16000, 16000 * 4, 4, 32)
            chunks = [
                b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk,
                b"data" + struct.pack("<I", len(payload)) + payload,
            ]
            body = b"WAVE" + b"".join(chunks)
            path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)

            loaded_waveform, sample_rate = read_wav(path)

            self.assertEqual(sample_rate, 16000)
            self.assertTrue(torch.allclose(loaded_waveform, waveform))

    def test_tokenizer_decodes_special_tokens_and_metaspace(self) -> None:
        """Decoder removes special IDs and converts Metaspace markers to spaces."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            tokenizer_path = Path(temporary_directory) / "tokenizer.json"
            tokenizer_path.write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "BPE",
                            "vocab": {
                                "<pad>": 2,
                                "<blank>": 3,
                                "hello": 4,
                                "world": 5,
                            },
                            "merges": [],
                        },
                        "added_tokens": [
                            {"id": 2, "content": "<pad>", "special": True},
                            {"id": 3, "content": "<blank>", "special": True},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            tokenizer = BpeTokenizer(tokenizer_path, pad_token_id=2, blank_token_id=3)
            tokenizer.id_to_token[4] = "▁hello"
            tokenizer.id_to_token[5] = "▁world"

            self.assertEqual(tokenizer.decode([2, 4, 5, 3]), "hello world")

    def test_slaney_filter_bank_has_expected_shape_and_finite_values(self) -> None:
        """Native mel filters have the processor's dimensions and no numerical anomalies."""

        filters = build_mel_filter_bank(16000, 512, 128, 0.0, 8000.0)

        self.assertEqual(tuple(filters.shape), (128, 257))
        self.assertTrue(torch.isfinite(filters).all())
        self.assertGreater(float(filters.sum()), 0.0)

    def test_select_device_returns_supported_torch_device(self) -> None:
        """Device selection is centralized and returns an actual torch device."""

        device = select_device()

        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, {"cpu", "cuda", "mps"})

    def test_attention_masks_fully_padded_queries_without_nan(self) -> None:
        """Mixed-duration padding must not create all-negative-infinity softmax rows."""

        configuration = ParakeetConfig(
            model={
                "encoder_config": {
                    "hidden_size": 8,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "attention_bias": False,
                }
            },
            processor={},
            generation={},
            weights_dir=DEFAULT_WEIGHTS_DIR,
        )
        attention = Attention(configuration, layer_index=0).eval()
        hidden_states = torch.randn(2, 4, 8)
        position_embeddings = torch.randn(2, 7, 8)
        valid_lengths = torch.tensor([2, 4])
        output_mask = torch.arange(4)[None, :] < valid_lengths[:, None]
        attention_mask = output_mask[:, None, :, None] & output_mask[:, None, None, :]

        output = attention(hidden_states, position_embeddings, attention_mask)

        self.assertEqual(tuple(output.shape), (2, 4, 8))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()