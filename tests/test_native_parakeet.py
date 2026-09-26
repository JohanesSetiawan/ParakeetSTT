"""Regression tests for the native model, features, config, and tokenizer."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.audio.features import ParakeetFeatureExtractor, build_mel_filter_bank
from src.configuration.config import ParakeetConfig, load_config
from src.models.attention import Attention
from src.models.parakeet import ParakeetTDT, load_model
from src.runtime.device import describe_runtime, select_device
from src.text.tokenization import BpeTokenizer
from support import TINY_BLANK_ID, TINY_VOCAB_SIZE, write_tiny_checkpoint


def _tiny_model(directory: Path, durations: list[int] | None = None) -> ParakeetTDT:
    torch.manual_seed(0)
    configuration = load_config(write_tiny_checkpoint(directory, durations))
    return ParakeetTDT(configuration).eval()


def _scripted_joint(token_id: int, duration_class: int, vocab_size: int):
    """Replace joint logits so every step picks one token and duration class."""

    def forward(decoder_hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = decoder_hidden_states.shape[0]
        duration_count = 5
        logits = torch.full(
            (batch_size, 1, vocab_size + duration_count),
            -1e4,
            device=decoder_hidden_states.device,
        )
        logits[..., token_id] = 1e4
        logits[..., vocab_size + duration_class] = 1e4
        return logits

    return forward


class ConfigurationTests(unittest.TestCase):
    """Checkpoint JSON is validated before any model allocation."""

    def test_tiny_checkpoint_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            configuration = load_config(write_tiny_checkpoint(Path(temporary_directory)))

        self.assertEqual(configuration.blank_token_id, TINY_BLANK_ID)
        self.assertEqual(configuration.durations, (0, 1, 2, 3, 4))

    def test_unimplemented_math_is_rejected(self) -> None:
        """Activations and input scaling the runtime hard-wires must match."""

        mutations = [
            ("hidden_act", None, "gelu"),
            ("hidden_act", "encoder_config", "relu"),
            ("scale_input", "encoder_config", True),
        ]
        for key, section, value in mutations:
            with self.subTest(key=key, section=section):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    directory = write_tiny_checkpoint(Path(temporary_directory))
                    config_path = directory / "config.json"
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    target = config[section] if section else config
                    target[key] = value
                    config_path.write_text(json.dumps(config), encoding="utf-8")

                    with self.assertRaises(ValueError):
                        load_config(directory)


class FeatureExtractionTests(unittest.TestCase):
    """Features stay finite and match the reference normalization."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.configuration = load_config(write_tiny_checkpoint(Path(self._directory.name)))
        self.extractor = ParakeetFeatureExtractor(self.configuration)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_tiny_and_empty_inputs_produce_finite_features(self) -> None:
        """
        Regression: fewer than two valid frames divided by n - 1 = 0 and fed
        NaN into the encoder; 217 samples then hung the decoder.
        """

        lengths = (0, 1, 159, 217, 320, 1600)
        waveforms = [torch.full((length,), 0.1) for length in lengths]
        features, mask = self.extractor(waveforms, [16000] * len(lengths), torch.device("cpu"))

        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(mask.sum(dim=1).tolist(), [length // 160 for length in lengths])

        empty_features, empty_mask = self.extractor([torch.zeros(0)], [16000], torch.device("cpu"))
        self.assertTrue(torch.isfinite(empty_features).all())
        self.assertEqual(int(empty_mask.sum()), 0)

    def test_normalization_matches_unbiased_statistics(self) -> None:
        """For n >= 2 valid frames the clamp changes nothing."""

        generator = torch.Generator().manual_seed(3)
        waveform = torch.randn(16000, generator=generator) * 0.1
        features, mask = self.extractor([waveform], [16000], torch.device("cpu"))

        valid = features[0, mask[0]]
        self.assertTrue(torch.allclose(valid.mean(dim=0), torch.zeros(valid.shape[1]), atol=1e-4))
        self.assertTrue(torch.allclose(valid.std(dim=0, unbiased=True), torch.ones(valid.shape[1]), atol=1e-3))

    def test_slaney_filter_bank_has_expected_shape_and_finite_values(self) -> None:
        filters = build_mel_filter_bank(16000, 512, 128, 0.0, 8000.0)

        self.assertEqual(tuple(filters.shape), (128, 257))
        self.assertTrue(torch.isfinite(filters).all())
        self.assertGreater(float(filters.sum()), 0.0)


class GenerationTests(unittest.TestCase):
    """Greedy TDT decoding terminates and follows the duration table."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _features(self, model: ParakeetTDT, samples: tuple[int, ...]):
        extractor = ParakeetFeatureExtractor(model.configuration)
        waveforms = [torch.randn(count) * 0.1 for count in samples]
        return extractor(waveforms, [16000] * len(samples), torch.device("cpu"))

    def test_stuck_decoder_is_forced_forward_and_terminates(self) -> None:
        """
        Regression: a joint that always emits a real token with duration 0
        used to run until the step bound and raise, killing the whole batch.
        """

        model = _tiny_model(Path(self._directory.name))
        features, mask = self._features(model, (16000, 8000))
        with patch.object(model.joint, "forward", _scripted_joint(5, 0, TINY_VOCAB_SIZE)):
            result = model.generate(features, mask)

        lengths = result.encoder_lengths.tolist()
        max_symbols = model.max_symbols_per_step
        self.assertEqual(result.forced_advances.tolist(), lengths)
        for row, length in enumerate(lengths):
            emitted = [token for token in result.sequences[row, 1:].tolist() if token != 2]
            self.assertEqual(len(emitted), length * max_symbols)

    def test_duration_classes_map_through_duration_values(self) -> None:
        """Class index 1 of durations [0, 2, 4] advances two frames, not one."""

        model = _tiny_model(Path(self._directory.name), durations=[0, 2, 4])
        features, mask = self._features(model, (16000,))
        with patch.object(model.joint, "forward", _scripted_joint(TINY_BLANK_ID, 1, TINY_VOCAB_SIZE)):
            result = model.generate(features, mask)

        length = int(result.encoder_lengths[0])
        steps = (length + 1) // 2
        self.assertEqual(result.durations[0, 1 : 1 + steps].tolist(), [2] * steps)
        self.assertEqual(result.frame_starts[0, 1 : 1 + steps].tolist(), list(range(0, 2 * steps, 2)))
        self.assertEqual(int(result.forced_advances[0]), 0)

    def test_non_finite_encoder_row_is_skipped(self) -> None:
        model = _tiny_model(Path(self._directory.name))
        features, mask = self._features(model, (16000, 16000))
        original = model.encoder_projector.forward

        def poisoned(hidden_states: torch.Tensor) -> torch.Tensor:
            output = original(hidden_states).clone()
            output[1, 0, 0] = float("nan")
            return output

        with patch.object(model.encoder_projector, "forward", poisoned):
            result = model.generate(features, mask)

        self.assertEqual(result.encoder_finite.tolist(), [True, False])
        self.assertTrue((result.sequences[1, 1:] == 2).all())


class CheckpointLoadingTests(unittest.TestCase):
    """model.pth loading is strict, safe, and fully materialized."""

    def test_round_trip_through_model_pth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = _tiny_model(directory)
            torch.save(
                {"state_dict": source.state_dict(), "config": {}, "metadata": {"note": "tiny"}},
                directory / "model.pth",
            )

            loaded, _configuration, metadata = load_model(directory, device=torch.device("cpu"))

        self.assertEqual(metadata, {"note": "tiny"})
        for name, tensor in source.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[name], tensor), name)
        for name, tensor in list(loaded.named_parameters()) + list(loaded.named_buffers()):
            self.assertFalse(tensor.is_meta, name)
        self.assertEqual(loaded.duration_values.tolist(), [0, 1, 2, 3, 4])
        self.assertTrue(torch.equal(
            loaded.encoder.encode_positions.inv_freq,
            source.encoder.encode_positions.inv_freq,
        ))

    def test_arbitrary_pickled_objects_are_refused(self) -> None:
        """weights_only=True: a tampered model.pth cannot run code on load."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = write_tiny_checkpoint(Path(temporary_directory))
            torch.save({"state_dict": {}, "payload": _Unsafe()}, directory / "model.pth")

            with self.assertRaises(pickle.UnpicklingError):
                load_model(directory, device=torch.device("cpu"))


class _Unsafe:
    """Stand-in for an attacker-controlled pickled object."""


class TokenizerAndRuntimeTests(unittest.TestCase):
    """Small boundary checks that need no checkpoint."""

    def test_tokenizer_decodes_special_tokens_and_metaspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tokenizer_path = Path(temporary_directory) / "tokenizer.json"
            tokenizer_path.write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "BPE",
                            "vocab": {"<pad>": 2, "<blank>": 3, "▁hello": 4, "▁world": 5},
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

            self.assertEqual(tokenizer.decode([2, 4, 5, 3]), "hello world")

    def test_select_device_and_runtime_report(self) -> None:
        device = select_device()
        report = describe_runtime(device, torch.float32)

        self.assertIn(device.type, {"cpu", "cuda", "mps"})
        self.assertEqual(report.precision, "float32")
        self.assertEqual(report.torch_version, torch.__version__)
        self.assertTrue(all(line.isascii() for line in report.lines()))

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
            weights_dir=Path("."),
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
