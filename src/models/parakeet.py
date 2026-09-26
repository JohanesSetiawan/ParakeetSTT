"""
Top-level Parakeet TDT model and checkpoint loading.

This module composes the acoustic encoder, LSTM prediction network, and TDT
joint network. It owns greedy token-duration control flow and strict loading of
``model.pth``. Audio parsing, feature extraction, tokenizer decoding, command-line
reporting, and artifact persistence remain outside this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..configuration.config import DEFAULT_WEIGHTS_DIR, ParakeetConfig, load_config
from .decoder import Decoder, DecoderCache
from .encoder import Encoder
from .joint import JointNetwork


@dataclass(frozen=True)
class GenerationResult:
    """Token IDs and per-step frame durations produced by greedy TDT decoding."""

    sequences: torch.LongTensor
    durations: torch.LongTensor
    frame_starts: torch.LongTensor | None = None
    frame_ends: torch.LongTensor | None = None
    encoder_lengths: torch.LongTensor | None = None


# =============================================================================
# Parakeet TDT composition
# =============================================================================
# State-dict compatibility depends on four top-level attribute names:
# encoder, encoder_projector, decoder, and joint. Their internal attributes also
# mirror checkpoint paths. File/module boundaries do not affect state-dict names,
# but renaming these attributes does.
# =============================================================================


class ParakeetTDT(nn.Module):
    """Standalone Parakeet Token-and-Duration Transducer model."""

    def __init__(self, configuration: ParakeetConfig) -> None:
        """Construct all checkpoint-compatible model components from JSON."""

        super().__init__()
        self.configuration = configuration
        self.encoder = Encoder(configuration)
        self.encoder_projector = nn.Linear(
            configuration.encoder["hidden_size"],
            configuration.model["decoder_hidden_size"],
        )
        self.decoder = Decoder(configuration)
        self.joint = JointNetwork(configuration)
        self.max_symbols_per_step = configuration.model["max_symbols_per_step"]

    def encode(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode acoustic features and project them into the joint dimension.

        Args:
            input_features: Padded log-mel batch ``(B, T_in, M)``.
            attention_mask: Valid input frames ``(B, T_in)``.

        Returns:
            Projected encoder states ``(B, T_out, H_decoder)`` and valid encoder
            mask ``(B, T_out)``.
        """

        hidden_states, output_mask = self.encoder(
            input_features,
            attention_mask,
        )
        if output_mask is None:
            raise RuntimeError("TDT generation requires an attention mask")

        return self.encoder_projector(hidden_states), output_mask

    @torch.inference_mode()
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> GenerationResult:
        """
        Run batched greedy TDT decoding until every encoder stream is exhausted.

        At each iteration, the model selects one token class and one duration
        class. The selected duration advances the encoder frame pointer. Blank
        tokens are forced to advance by at least one frame so decoding cannot
        remain forever on a frame with ``duration=0``.

        Args:
            input_features: Padded normalized features ``(B, T_in, M)``.
            attention_mask: Valid feature-frame mask ``(B, T_in)``.

        Returns:
            Padded token and duration sequences with shape ``(B, U)``.
        """

        encoder_states, encoder_mask = self.encode(input_features, attention_mask)
        batch_size, encoder_length, _ = encoder_states.shape
        valid_lengths = encoder_mask.sum(dim=-1)

        frame_indices = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=encoder_states.device,
        )
        finished = valid_lengths <= 0
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.configuration.blank_token_id,
            dtype=torch.long,
            device=encoder_states.device,
        )
        decoder_cache = DecoderCache(self.configuration)
        # This upper bound mirrors the reference generation buffer policy. The
        # normal stopping condition is encoder exhaustion; reaching this bound
        # indicates every frame emitted max_symbols_per_step zero-duration tokens.
        maximum_steps = self.max_symbols_per_step * max(1, encoder_length)
        output_capacity = maximum_steps + 1
        sequence_buffer = torch.full(
            (batch_size, output_capacity),
            self.configuration.pad_token_id,
            dtype=torch.long,
            device=encoder_states.device,
        )
        duration_buffer = torch.zeros_like(sequence_buffer)
        frame_start_buffer = torch.zeros_like(sequence_buffer)
        frame_end_buffer = torch.zeros_like(sequence_buffer)
        sequence_buffer[:, 0] = decoder_input_ids[:, 0]
        output_length = 1
        batch_indices = torch.arange(
            batch_size,
            device=encoder_states.device,
        )
        blank_input_ids = torch.full_like(
            decoder_input_ids,
            self.configuration.blank_token_id,
        )
        pad_token_ids = torch.full_like(
            decoder_input_ids,
            self.configuration.pad_token_id,
        )
        zero_durations = torch.zeros_like(decoder_input_ids)

        for _step in range(maximum_steps):
            decoder_hidden_states = self.decoder(
                decoder_input_ids,
                decoder_cache,
            )

            safe_frame_indices = frame_indices.clamp(max=encoder_length - 1)
            current_encoder_states = encoder_states[
                batch_indices,
                safe_frame_indices,
                None,
                :,
            ]
            logits = self.joint(
                decoder_hidden_states,
                current_encoder_states,
            ).squeeze(1)

            token_ids = logits[:, : self.configuration.vocab_size].argmax(dim=-1)
            duration_ids = logits[:, self.configuration.vocab_size :].argmax(dim=-1)
            active = ~finished
            blank_mask = token_ids == self.configuration.blank_token_id

            duration_ids = torch.where(
                active & blank_mask & (duration_ids == 0),
                torch.ones_like(duration_ids),
                duration_ids,
            )
            emitted_token_ids = torch.where(
                active,
                token_ids,
                pad_token_ids[:, 0],
            )
            emitted_durations = torch.where(
                active,
                duration_ids,
                zero_durations[:, 0],
            )
            emitted_frame_starts = torch.where(
                active,
                frame_indices,
                zero_durations[:, 0],
            )
            emitted_frame_ends = torch.where(
                active,
                frame_indices + emitted_durations,
                zero_durations[:, 0],
            )
            sequence_buffer[:, output_length] = emitted_token_ids
            duration_buffer[:, output_length] = emitted_durations
            frame_start_buffer[:, output_length] = emitted_frame_starts
            frame_end_buffer[:, output_length] = emitted_frame_ends
            output_length += 1

            frame_indices = torch.where(
                active,
                frame_indices + emitted_durations,
                frame_indices,
            )
            finished = finished | (frame_indices >= valid_lengths)


            # Completed rows receive blank decoder input to preserve cache state;
            # emitted output for those rows is already padded above.
            decoder_input_ids = torch.where(
                finished,
                blank_input_ids[:, 0],
                emitted_token_ids,
            )[:, None]

            if bool(finished.all()):
                break
        else:
            raise RuntimeError(
                "TDT decoding reached the maximum step bound before encoder exhaustion"
            )

        return GenerationResult(
            sequences=sequence_buffer[:, :output_length],
            durations=duration_buffer[:, :output_length],
            frame_starts=frame_start_buffer[:, :output_length],
            frame_ends=frame_end_buffer[:, :output_length],
            encoder_lengths=valid_lengths,
        )


# =============================================================================
# Device and checkpoint loading
# =============================================================================
# Model construction always starts on CPU. The state dict is loaded strictly
# before moving the model to the selected accelerator, which avoids allocating
# an incompatible model on scarce GPU memory and provides complete key errors.
# =============================================================================


def select_device() -> torch.device:
    """Select CUDA, then MPS, then CPU from actual PyTorch availability."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    device: torch.device | None = None,
) -> tuple[ParakeetTDT, ParakeetConfig, dict[str, Any]]:
    """
    Strict-load ``model.pth`` into the standalone architecture.

    Args:
        weights_dir: Directory containing JSON artifacts and ``model.pth``.
        device: Optional target device. When omitted, centralized automatic
            selection chooses CUDA, MPS, or CPU.

    Returns:
        Loaded eval-mode model, validated configuration, and checkpoint metadata.

    Raises:
        ValueError: If the checkpoint bundle lacks a state dictionary.
        RuntimeError: If strict state loading reports missing or unexpected keys.
    """

    configuration = load_config(weights_dir)
    model = ParakeetTDT(configuration)
    checkpoint = torch.load(
        configuration.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("model.pth must contain a dictionary with state_dict")

    incompatibility = model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(
            f"Strict state-dict load reported incompatibility: {incompatibility}"
        )

    resolved_device = device or select_device()
    model = model.to(resolved_device)
    model.eval()

    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("model.pth metadata must be a dictionary")

    return model, configuration, metadata
