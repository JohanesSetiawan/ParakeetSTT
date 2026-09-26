"""
Shared fixtures for tests that must not depend on the 2.5 GB checkpoint.

``write_tiny_checkpoint`` produces a complete, schema-valid checkpoint
directory (config, processor, generation, tokenizer JSON) for a model small
enough to build and run on CPU in milliseconds. Its structure mirrors the real
Parakeet config.json key for key; only the sizes differ.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

from src.configuration.settings import CheckpointSettings, InferenceSettings

TINY_VOCAB_SIZE = 12
TINY_BLANK_ID = TINY_VOCAB_SIZE - 1
TINY_PAD_ID = 2
TINY_SAMPLE_RATE = 16000


def tiny_model_config(durations: list[int] | None = None) -> dict[str, object]:
    """Return a Parakeet config.json with tiny dimensions."""

    return {
        "architectures": ["ParakeetForTDT"],
        "blank_token_id": TINY_BLANK_ID,
        "decoder_hidden_size": 8,
        "durations": durations if durations is not None else [0, 1, 2, 3, 4],
        "encoder_config": {
            "activation_dropout": 0.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "conv_kernel_size": 3,
            "convolution_bias": False,
            "dropout": 0.0,
            "dropout_positions": 0.0,
            "hidden_act": "silu",
            "hidden_size": 16,
            "initializer_range": 0.02,
            "intermediate_size": 32,
            "layerdrop": 0.0,
            "max_position_embeddings": 5000,
            "model_type": "parakeet_encoder",
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
            "num_mel_bins": 8,
            "scale_input": False,
            "subsampling_conv_channels": 4,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "subsampling_factor": 8,
        },
        "hidden_act": "relu",
        "initializer_range": 0.02,
        "is_encoder_decoder": True,
        "max_symbols_per_step": 3,
        "model_type": "parakeet_tdt",
        "num_decoder_layers": 1,
        "pad_token_id": TINY_PAD_ID,
        "vocab_size": TINY_VOCAB_SIZE,
    }


def write_tiny_checkpoint(directory: Path, durations: list[int] | None = None) -> Path:
    """Write every JSON artifact ``load_config`` and the tokenizer need."""

    model_config = tiny_model_config(durations)
    duration_count = len(model_config["durations"])  # type: ignore[arg-type]
    processor = {
        "blank_token": "<blank>",
        "feature_extractor": {
            "feature_extractor_type": "ParakeetFeatureExtractor",
            "feature_size": 8,
            "hop_length": 160,
            "n_fft": 512,
            "padding_side": "right",
            "padding_value": 0.0,
            "preemphasis": 0.97,
            "return_attention_mask": True,
            "sampling_rate": TINY_SAMPLE_RATE,
            "win_length": 400,
        },
        "processor_class": "ParakeetProcessor",
    }
    generation = {
        "decoder_start_token_id": TINY_BLANK_ID,
        "pad_token_id": TINY_PAD_ID,
        "suppress_tokens": list(range(TINY_VOCAB_SIZE, TINY_VOCAB_SIZE + duration_count)),
    }
    vocabulary = {"<unk>": 0, "<eos>": 1, "<pad>": TINY_PAD_ID}
    pieces = ["▁a", "▁b", "▁c", "d", "e", "▁f", "g", "h"]
    for offset, piece in enumerate(pieces):
        vocabulary[piece] = 3 + offset
    tokenizer = {
        "model": {"type": "BPE", "vocab": vocabulary, "merges": []},
        "added_tokens": [
            {"id": TINY_PAD_ID, "content": "<pad>", "special": True},
            {"id": TINY_BLANK_ID, "content": "<blank>", "special": True},
        ],
    }

    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("config.json", model_config),
        ("processor_config.json", processor),
        ("generation_config.json", generation),
        ("tokenizer.json", tokenizer),
    ):
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")
    return directory


def write_float_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Write a mono IEEE-float WAV using only standard-library bytes."""

    payload = struct.pack(f"<{waveform.numel()}f", *waveform.tolist())
    fmt = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def checkpoint_settings() -> CheckpointSettings:
    """Fast network policy for local-server download tests."""

    return CheckpointSettings(
        request_timeout_seconds=5.0,
        download_attempts=2,
        retry_backoff_seconds=0.0,
        stream_block_bytes=4,
    )


def inference_settings(**overrides: object) -> InferenceSettings:
    """Small but valid inference budgets; override any field by keyword."""

    values: dict[str, object] = {
        "batch_size": 2,
        "recursive": True,
        "output_filename": "transcriptions.csv",
        "audio_extensions": (),
        "max_chunk_feature_frames": 100,
        "overlap_feature_frames": 5,
        "max_batch_feature_frames": 200,
        "max_padding_fraction": 0.5,
        "progress_interval_seconds": 0.0,
    }
    values.update(overrides)
    return InferenceSettings(**values)  # type: ignore[arg-type]
