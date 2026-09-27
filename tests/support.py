"""
Shared test helpers.

Tiny checkpoint: ``write_tiny_checkpoint`` produces a complete, schema-valid
checkpoint directory (config, processor, generation, tokenizer JSON) for a
model small enough to build and run on CPU in milliseconds. Its structure
mirrors the real Parakeet config.json key for key; only the sizes differ.

Real speech: ``load_speech_clips`` reads the committed LibriSpeech clips and
their manifest (reference text and expected transcripts) for the full tier.

Scoring: ``word_error_rate`` is the standard Levenshtein WER over normalized
words, implemented here so the tests need no scoring library.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from src.configuration.settings import CheckpointSettings, InferenceSettings
from src.models.parakeet import GenerationResult

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
        "merge_tolerance_feature_frames": 10,
        "untranscribed_gap_seconds": 4.0,
        "gap_silence_rms": 0.001,
        "recovery_start_offsets_feature_frames": (),
        "progress_interval_seconds": 0.0,
    }
    values.update(overrides)
    return InferenceSettings(**values)  # type: ignore[arg-type]


# =============================================================================
# Real speech clips (full tier)
# =============================================================================

SPEECH_DIR = Path(__file__).resolve().parent / "data" / "librispeech"


@dataclass(frozen=True)
class SpeechClip:
    """One committed LibriSpeech utterance and its expected outputs."""

    clip_id: str
    path: Path
    sha256: str
    duration_seconds: float
    chunks: int
    reference_text: str
    expected_transcript: str
    expected_status: str
    full_context_transcript: str


def load_speech_clips() -> tuple[SpeechClip, ...]:
    """
    Read manifest.json and verify every audio file against its SHA-256.

    Raises:
        AssertionError: If a clip is missing or its bytes changed, because
            every expected transcript is only valid for those exact bytes.
    """

    manifest = json.loads((SPEECH_DIR / "manifest.json").read_text(encoding="utf-8"))
    clips = []
    for entry in manifest["clips"]:
        path = SPEECH_DIR / entry["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"{path.name} does not match its manifest hash"
        clips.append(
            SpeechClip(
                clip_id=entry["id"],
                path=path,
                sha256=entry["sha256"],
                duration_seconds=entry["duration_seconds"],
                chunks=entry["chunks"],
                reference_text=entry["reference_text"],
                expected_transcript=entry["expected_transcript"],
                expected_status=entry["expected_status"],
                full_context_transcript=entry["full_context_transcript"],
            )
        )
    return tuple(clips)


# =============================================================================
# Word error rate
# =============================================================================


# LibriSpeech spells titles out ("MISTER"); the model writes either "mister"
# or "Mr". Mapping both to one form keeps a spelling choice from counting as
# a recognition error. This mirrors the title handling of common English ASR
# normalizers and is limited to titles LibriSpeech actually spells out.
SPELLED_TITLES = {"mr": "mister", "mrs": "missus", "dr": "doctor"}


def normalize_words(text: str) -> list[str]:
    """
    Lowercase, drop punctuation except apostrophes, spell out titles, split.

    LibriSpeech references are uppercase without punctuation, while the model
    writes cased, punctuated text; both must meet in the same form.
    """

    lowered = text.lower().replace("-", " ")
    letters_only = re.sub(r"[^a-z0-9' ]+", " ", lowered)
    return [SPELLED_TITLES.get(word, word) for word in letters_only.split()]


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """Return substitutions + deletions + insertions (Levenshtein on words)."""

    previous_row = list(range(len(hypothesis) + 1))
    for reference_index, reference_word in enumerate(reference, start=1):
        current_row = [reference_index] + [0] * len(hypothesis)
        for hypothesis_index, hypothesis_word in enumerate(hypothesis, start=1):
            substitution = previous_row[hypothesis_index - 1] + (reference_word != hypothesis_word)
            deletion = previous_row[hypothesis_index] + 1
            insertion = current_row[hypothesis_index - 1] + 1
            current_row[hypothesis_index] = min(substitution, deletion, insertion)
        previous_row = current_row
    return previous_row[-1]


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Corpus WER: total word errors divided by total reference words."""

    total_errors = 0
    total_words = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        reference_words = normalize_words(reference)
        total_errors += word_errors(reference_words, normalize_words(hypothesis))
        total_words += len(reference_words)
    return total_errors / total_words


# =============================================================================
# Scripted model (orchestration tests without real inference)
# =============================================================================


class ScriptedModel(nn.Module):
    """
    Emit one fixed token per row; optionally fail the encoder.

    ``at_last_frame`` emits on the row's final encoder frame with duration 4,
    so the token midpoint lies past the end of the audio.
    """

    def __init__(self, token_id: int, encoder_finite: bool = True, at_last_frame: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.token_id = token_id
        self.encoder_finite = encoder_finite
        self.at_last_frame = at_last_frame

    def generate(self, features: torch.Tensor, mask: torch.Tensor) -> GenerationResult:
        batch_size = features.shape[0]
        encoder_lengths = (mask.sum(dim=1) + 7) // 8
        if self.at_last_frame:
            starts = (encoder_lengths - 1).clamp(min=0)[:, None]
            durations = torch.full((batch_size, 1), 4, dtype=torch.long)
        else:
            starts = torch.zeros((batch_size, 1), dtype=torch.long)
            durations = torch.ones((batch_size, 1), dtype=torch.long)
        return GenerationResult(
            sequences=torch.full((batch_size, 1), self.token_id, dtype=torch.long),
            durations=durations,
            frame_starts=starts,
            frame_ends=starts + durations,
            encoder_lengths=encoder_lengths,
            forced_advances=torch.zeros(batch_size, dtype=torch.long),
            encoder_finite=torch.full((batch_size,), self.encoder_finite),
        )


class SteadySpeechModel(nn.Module):
    """
    Emit one word every ``every`` encoder frames, like steady speech; on the
    generate calls listed in ``collapse_calls`` emit nothing at all, like the
    real model's decoder collapse. Calls are counted from 0.
    """

    def __init__(self, token_id: int, every: int = 4, collapse_calls: frozenset[int] = frozenset()) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.token_id = token_id
        self.every = every
        self.collapse_calls = collapse_calls
        self.calls = 0

    def generate(self, features: torch.Tensor, mask: torch.Tensor) -> GenerationResult:
        call = self.calls
        self.calls += 1
        batch_size = features.shape[0]
        encoder_lengths = (mask.sum(dim=1) + 7) // 8
        longest = int(encoder_lengths.max())
        starts = torch.arange(0, max(longest, 1), self.every, dtype=torch.long)
        steps = starts.numel()
        sequences = torch.full((batch_size, steps), self.token_id, dtype=torch.long)
        frame_starts = starts.repeat(batch_size, 1)
        durations = torch.full((batch_size, steps), self.every, dtype=torch.long)
        for row in range(batch_size):
            beyond = frame_starts[row] >= encoder_lengths[row]
            sequences[row, beyond] = 2
            durations[row, beyond] = 0
            if call in self.collapse_calls:
                sequences[row] = TINY_BLANK_ID
        return GenerationResult(
            sequences=sequences,
            durations=durations,
            frame_starts=frame_starts,
            frame_ends=frame_starts + durations,
            encoder_lengths=encoder_lengths,
            forced_advances=torch.zeros(batch_size, dtype=torch.long),
            encoder_finite=torch.ones(batch_size, dtype=torch.bool),
        )
