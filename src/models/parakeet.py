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

from ..configuration.config import ParakeetConfig, load_config
from ..runtime.device import select_device
from .decoder import Decoder, DecoderCache
from .encoder import Encoder
from .joint import JointNetwork


# Decoding steps between two host-side completion checks. Each check costs a
# GPU-to-host synchronization; at most this many minus one extra steps of
# padding run after the batch is done. It only trades latency for syncs and
# never changes the output, so it is an implementation constant, not a setting.
FINISHED_CHECK_INTERVAL = 8


@dataclass(frozen=True)
class GenerationResult:
    """
    Token IDs and per-step frame bookkeeping produced by greedy TDT decoding.

    Every tensor is ``(B, U)`` except the per-row summaries. Rows that finish
    early are right-padded with ``pad_token_id`` and zero durations.

    Attributes:
        sequences: Emitted token per step; step 0 is the decoder start token.
        durations: Encoder frames advanced after each step.
        frame_starts: Encoder frame each step was emitted on.
        frame_ends: ``frame_starts + durations``.
        encoder_lengths: Valid encoder frames per row ``(B,)``.
        forced_advances: Times the per-frame symbol guard forced progress
            ``(B,)``; zero on well-behaved audio.
        encoder_finite: Whether every valid encoder state was finite ``(B,)``.
            Non-finite rows are not decoded and emit no tokens.
    """

    sequences: torch.LongTensor
    durations: torch.LongTensor
    frame_starts: torch.LongTensor
    frame_ends: torch.LongTensor
    encoder_lengths: torch.LongTensor
    forced_advances: torch.LongTensor
    encoder_finite: torch.BoolTensor


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

        # Duration classes map a joint-output index to a frame count. The
        # buffer is not persistent, so the checkpoint key set is unchanged, and
        # it is created on CPU so it stays real under meta-device construction.
        self.register_buffer(
            "duration_values",
            torch.tensor(configuration.durations, dtype=torch.long, device="cpu"),
            persistent=False,
        )

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

        At each step the joint network picks one token class and one duration
        class; the duration's frame count advances that row's encoder pointer.
        Two rules guarantee progress:

        * a blank token with duration 0 advances one frame (reference behavior);
        * after ``max_symbols_per_step`` consecutive non-blank tokens on the same
          frame, the last one is forced to advance one frame. This mirrors the
          NeMo greedy transducer guard. It never fires on normal speech, and it
          turns degenerate input into bounded output instead of a hang.

        Args:
            input_features: Padded normalized features ``(B, T_in, M)``.
            attention_mask: Valid feature-frame mask ``(B, T_in)``.

        Returns:
            Padded token, duration, and frame bookkeeping; see
            :class:`GenerationResult`.
        """

        encoder_states, encoder_mask = self.encode(input_features, attention_mask)
        batch_size, encoder_length, _ = encoder_states.shape
        device = encoder_states.device
        valid_lengths = encoder_mask.sum(dim=-1)

        # One reduction per batch (not per step): padded frames are excluded
        # because their values are never read by the decoder.
        finite_or_padding = torch.isfinite(encoder_states) | ~encoder_mask[:, :, None]
        encoder_finite = finite_or_padding.all(dim=2).all(dim=1)

        blank_token_id = self.configuration.blank_token_id
        pad_token_id = self.configuration.pad_token_id
        vocab_size = self.configuration.vocab_size

        frame_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        symbols_on_frame = torch.zeros_like(frame_indices)
        forced_advances = torch.zeros_like(frame_indices)
        finished = (valid_lengths <= 0) | ~encoder_finite

        decoder_input_ids = torch.full(
            (batch_size, 1),
            blank_token_id,
            dtype=torch.long,
            device=device,
        )
        decoder_cache = DecoderCache(self.configuration)

        # With the per-frame guard each frame consumes at most
        # max_symbols_per_step steps, so this bound is a safety net that only a
        # logic error can reach.
        maximum_steps = self.max_symbols_per_step * max(1, encoder_length)
        output_capacity = maximum_steps + 1
        sequence_buffer = torch.full(
            (batch_size, output_capacity),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        duration_buffer = torch.zeros_like(sequence_buffer)
        frame_start_buffer = torch.zeros_like(sequence_buffer)
        sequence_buffer[:, 0] = blank_token_id
        output_length = 1

        batch_indices = torch.arange(batch_size, device=device)
        blank_ids = torch.full((batch_size,), blank_token_id, dtype=torch.long, device=device)
        pad_ids = torch.full_like(blank_ids, pad_token_id)
        # Frame-start marker of a row that no longer decodes. Real frame
        # indices are never negative, so the marker identifies the steps that
        # ran only after every row had finished.
        inactive_frame = torch.full_like(blank_ids, -1)
        max_symbols = self.max_symbols_per_step

        # This loop runs once per emitted symbol, so each elementwise op below
        # is a kernel launch paid thousands of times per batch. An identity
        # duration table (Parakeet's 0..4) therefore skips the lookup.
        duration_classes_are_frames = self.configuration.durations == tuple(
            range(len(self.configuration.durations))
        )

        for step in range(maximum_steps):
            # Checking for completion copies a flag to the host and stalls the
            # GPU queue, so it runs only every few steps. Steps taken after
            # every row finished emit padding only and are trimmed below.
            if step % FINISHED_CHECK_INTERVAL == 0 and bool(finished.all()):
                break

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
            ]  # (B, 1, H_decoder)
            logits = self.joint(
                decoder_hidden_states,
                current_encoder_states,
            ).squeeze(1)  # (B, vocab + durations)

            token_ids = logits[:, :vocab_size].argmax(dim=-1)
            frame_advance = logits[:, vocab_size:].argmax(dim=-1)
            if not duration_classes_are_frames:
                frame_advance = self.duration_values[frame_advance]

            active = ~finished
            blank_mask = token_ids == blank_token_id
            zero_advance = (frame_advance == 0) & active

            # Count consecutive non-blank zero-duration emissions on the current
            # frame. Counting modulo max_symbols restarts the count at 1 right
            # after a forced advance, so no separate reset op is needed.
            stays_on_frame = zero_advance & ~blank_mask
            symbols_on_frame = (symbols_on_frame % max_symbols + 1) * stays_on_frame
            guard_fires = symbols_on_frame == max_symbols
            forced_advances += guard_fires

            # A blank with duration 0 (reference rule) and a guarded token both
            # advance exactly one frame.
            frame_advance = frame_advance + (zero_advance & (blank_mask | guard_fires))

            emitted_token_ids = torch.where(active, token_ids, pad_ids)
            emitted_durations = frame_advance * active
            sequence_buffer[:, output_length] = emitted_token_ids
            duration_buffer[:, output_length] = emitted_durations
            frame_start_buffer[:, output_length] = torch.where(active, frame_indices, inactive_frame)
            output_length += 1

            frame_indices = frame_indices + emitted_durations
            finished = finished | (frame_indices >= valid_lengths)

            # Completed rows receive blank decoder input so their cache row is
            # left untouched; their emitted output is already padded above.
            decoder_input_ids = torch.where(finished, blank_ids, emitted_token_ids)[:, None]
        else:
            if not bool(finished.all()):
                raise RuntimeError(
                    "TDT decoding reached the maximum step bound before encoder "
                    "exhaustion despite the per-frame symbol guard"
                )

        # Keep exactly the steps in which some row was still decoding. Rows only
        # finish, never restart, so those steps are a prefix; the steps run
        # between completion and the next check come after it and are
        # dropped, giving the same result as stopping at the exact step. The
        # pad token itself is a valid joint output, so it cannot mark them.
        step_frame_starts = frame_start_buffer[:, 1:output_length]  # (B, steps)
        step_was_active = (step_frame_starts >= 0).any(dim=0)  # (steps,)
        output_length = 1 + int(step_was_active.sum())

        durations = duration_buffer[:, :output_length]
        # Rows that had finished report frame 0, as before the marker existed.
        frame_starts = frame_start_buffer[:, :output_length].clamp(min=0)
        return GenerationResult(
            sequences=sequence_buffer[:, :output_length],
            durations=durations,
            frame_starts=frame_starts,
            frame_ends=frame_starts + durations,
            encoder_lengths=valid_lengths,
            forced_advances=forced_advances,
            encoder_finite=encoder_finite,
        )


# =============================================================================
# Checkpoint loading
# =============================================================================
# Model construction always starts on CPU. The state dict is loaded strictly
# before moving the model to the selected accelerator, which avoids allocating
# an incompatible model on scarce GPU memory and provides complete key errors.
# =============================================================================


def load_model(
    weights_dir: Path,
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

    # The meta device builds the module graph without allocating or randomly
    # initializing 600M parameters that the checkpoint overwrites anyway;
    # load_state_dict(assign=True) then adopts the loaded CPU tensors directly.
    with torch.device("meta"):
        model = ParakeetTDT(configuration)

    # weights_only=True refuses arbitrary pickled objects, so a replaced or
    # tampered model.pth cannot execute code during load.
    #
    # mmap=True maps the file instead of copying 2.4 GB into private memory.
    # Copied tensors stayed resident after the move to the GPU (3.1 GB
    # working set, 8.2 GB commit charge measured); mapped pages belong to the
    # file and are released with the CPU tensors (0.7 GB working set, 4.1 GB
    # commit), at the same load time.
    checkpoint = torch.load(
        configuration.checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("model.pth must contain a dictionary with state_dict")

    incompatibility = model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
        assign=True,
    )
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(
            f"Strict state-dict load reported incompatibility: {incompatibility}"
        )

    _require_materialized(model)

    resolved_device = device or select_device()
    model = model.to(resolved_device)
    model.eval()

    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("model.pth metadata must be a dictionary")

    return model, configuration, metadata


def _require_materialized(model: nn.Module) -> None:
    """
    Fail loudly if any tensor is still on the meta device after loading.

    Parameters and persistent buffers come from the checkpoint; non-persistent
    buffers must be created on CPU in their module constructors. A meta tensor
    here means a new buffer was added without following that rule.
    """

    leftovers = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.is_meta
    ]
    if leftovers:
        raise RuntimeError(
            f"Tensors were not materialized from the checkpoint: {leftovers[:10]}"
        )
