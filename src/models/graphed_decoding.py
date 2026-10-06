"""
Greedy TDT decoding with each step captured once as a CUDA Graph.

The eager loop in ``ParakeetTDT._decode_eager`` launches about 46 small
kernels per emitted symbol. On a small GPU that is launch-bound: measured
1.2 to 2.0 ms per step against 0.4 ms of actual GPU work. A CUDA Graph replays
the whole step with a single launch, so a step costs about what the GPU
computes (0.38 ms measured), and the output is bit-identical to the eager
loop because the same operations run on the same values.

A captured graph replays on fixed tensors at fixed addresses, so every step
works in place on static buffers. One graph is sized for the largest batch
(``max_batch_rows`` x ``max_encoder_frames``) and serves every smaller one:

* missing rows are marked finished from the start, so they emit nothing;
* frames past a row's valid length are never read by an active row (a row
  finishes as soon as its frame pointer reaches its valid length), so the
  encoder states can sit at the start of a larger zeroed buffer.

Per-step GPU time is dominated by reading the decoder and joint weights and
barely depends on the row count (0.38 ms for 1 row, 0.41 ms for 8, measured),
so decoding small batches at full width costs almost nothing and avoids a
separate capture, and its memory, for every batch shape.
"""

from __future__ import annotations

import torch

from .parakeet import FINISHED_CHECK_INTERVAL, DecodedSteps, ParakeetTDT


class GraphedGreedyDecoder:
    """
    Static-buffer greedy TDT decoder, replayed through one CUDA Graph.

    Args:
        model: The model whose decoder, joint, and configuration are used.
        max_batch_rows: Rows of the static buffers (largest batch).
        max_encoder_frames: Encoder frames of the static buffers (longest row).
        use_cuda_graph: Capture and replay the step on CUDA. False runs the
            same static-buffer step eagerly, which tests use on CPU to check
            the step logic against the eager loop.
    """

    # Eager steps run on a side stream before capture, so CUDA libraries
    # allocate their workspaces outside the graph (PyTorch's documented
    # capture recipe).
    WARMUP_STEPS = 3

    def __init__(
        self,
        model: ParakeetTDT,
        max_batch_rows: int,
        max_encoder_frames: int,
        use_cuda_graph: bool = True,
    ) -> None:
        if max_batch_rows < 1 or max_encoder_frames < 1:
            raise ValueError("Graph decoding needs at least one row and one encoder frame")

        configuration = model.configuration
        self.model = model
        self.rows = max_batch_rows
        self.frames = max_encoder_frames
        self.use_cuda_graph = use_cuda_graph
        self.device = next(model.parameters()).device
        self.blank_token_id = configuration.blank_token_id
        self.pad_token_id = configuration.pad_token_id
        self.vocab_size = configuration.vocab_size
        self.max_symbols = model.max_symbols_per_step
        self.duration_classes_are_frames = model.durations_are_frame_counts
        self.graph: torch.cuda.CUDAGraph | None = None

        hidden_size = configuration.model["decoder_hidden_size"]
        layer_count = configuration.model["num_decoder_layers"]
        capacity = self.max_symbols * max_encoder_frames + 1
        decoder_dtype = model.decoder.decoder_projector.weight.dtype
        long_options = {"dtype": torch.long, "device": self.device}

        # Inputs, copied in for every batch.
        self.encoder_states = torch.zeros(
            max_batch_rows,
            max_encoder_frames,
            hidden_size,
            dtype=torch.float32,
            device=self.device,
        )  # (B, T, H_decoder)
        self.valid_lengths = torch.zeros(max_batch_rows, **long_options)

        # Constants.
        self.batch_indices = torch.arange(max_batch_rows, device=self.device)
        self.blank_ids = torch.full((max_batch_rows,), self.blank_token_id, **long_options)
        self.pad_ids = torch.full((max_batch_rows,), self.pad_token_id, **long_options)
        self.inactive_frame = torch.full((max_batch_rows,), -1, **long_options)

        # Decoding state, updated in place by every step.
        self.frame_indices = torch.zeros(max_batch_rows, **long_options)
        self.symbols_on_frame = torch.zeros(max_batch_rows, **long_options)
        self.forced_advances = torch.zeros(max_batch_rows, **long_options)
        self.finished = torch.ones(max_batch_rows, dtype=torch.bool, device=self.device)
        self.decoder_input_ids = torch.full((max_batch_rows, 1), self.blank_token_id, **long_options)
        self.first_step = torch.ones(1, dtype=torch.bool, device=self.device)
        self.hidden_state = torch.zeros(
            layer_count,
            max_batch_rows,
            hidden_size,
            dtype=decoder_dtype,
            device=self.device,
        )  # (L, B, H)
        self.cell_state = torch.zeros_like(self.hidden_state)
        self.decoder_output = torch.zeros(
            max_batch_rows,
            1,
            hidden_size,
            dtype=decoder_dtype,
            device=self.device,
        )  # (B, 1, H)

        # Outputs, one column per step; written at a device-side position so
        # the captured step does not depend on a Python counter.
        self.sequence_buffer = torch.full((max_batch_rows, capacity), self.pad_token_id, **long_options)
        self.duration_buffer = torch.zeros_like(self.sequence_buffer)
        self.frame_start_buffer = torch.zeros_like(self.sequence_buffer)
        self.write_position = torch.ones(1, **long_options)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def accepts(self, encoder_states: torch.Tensor) -> bool:
        """Whether a batch fits the static buffers."""

        batch_size, encoder_length, hidden_size = encoder_states.shape
        return (
            encoder_states.device == self.device
            and encoder_states.dtype == torch.float32
            and batch_size <= self.rows
            and encoder_length <= self.frames
            and hidden_size == self.encoder_states.shape[2]
        )

    def decode(
        self,
        encoder_states: torch.Tensor,
        valid_lengths: torch.Tensor,
        encoder_finite: torch.Tensor,
    ) -> DecodedSteps:
        """
        Decode one batch; same contract as ``ParakeetTDT._decode_eager``.

        Args:
            encoder_states: Projected encoder states ``(B, T, H)`` in float32.
            valid_lengths: Valid encoder frames per row ``(B,)``.
            encoder_finite: Whether each row's states are finite ``(B,)``.

        Returns:
            Per-step buffers for the batch's rows (copies, since the static
            buffers are reused by the next batch).
        """

        if not self.accepts(encoder_states):
            raise ValueError("Batch does not fit the graph decoder's static buffers")

        batch_size, encoder_length, _ = encoder_states.shape
        self._load(encoder_states, valid_lengths, encoder_finite)
        if self.use_cuda_graph and self.graph is None:
            self._capture(encoder_states, valid_lengths, encoder_finite)

        maximum_steps = self.max_symbols * max(1, encoder_length)
        for step in range(maximum_steps):
            # Same completion check and cadence as the eager loop.
            if step % FINISHED_CHECK_INTERVAL == 0 and bool(self.finished.all()):
                break
            if self.graph is not None:
                self.graph.replay()
            else:
                self._step()
        else:
            if not bool(self.finished.all()):
                raise RuntimeError(
                    "TDT decoding reached the maximum step bound before encoder "
                    "exhaustion despite the per-frame symbol guard"
                )

        written_columns = int(self.write_position.item())
        return DecodedSteps(
            sequences=self.sequence_buffer[:batch_size, :written_columns].clone(),
            durations=self.duration_buffer[:batch_size, :written_columns].clone(),
            frame_starts=self.frame_start_buffer[:batch_size, :written_columns].clone(),
            written_columns=written_columns,
            forced_advances=self.forced_advances[:batch_size].clone(),
        )

    # -------------------------------------------------------------------------
    # Static state
    # -------------------------------------------------------------------------

    def _load(
        self,
        encoder_states: torch.Tensor,
        valid_lengths: torch.Tensor,
        encoder_finite: torch.Tensor,
    ) -> None:
        """Copy one batch into the static inputs and reset the decoding state."""

        batch_size, encoder_length, _ = encoder_states.shape
        self.encoder_states.zero_()
        self.encoder_states[:batch_size, :encoder_length].copy_(encoder_states)
        self.valid_lengths.zero_()
        self.valid_lengths[:batch_size].copy_(valid_lengths)
        # Rows beyond the batch are finished from the start and emit nothing.
        self.finished.fill_(True)
        self.finished[:batch_size].copy_(ParakeetTDT.initial_finished(valid_lengths, encoder_finite))

        self.frame_indices.zero_()
        self.symbols_on_frame.zero_()
        self.forced_advances.zero_()
        self.decoder_input_ids.fill_(self.blank_token_id)
        self.first_step.fill_(True)
        self.hidden_state.zero_()
        self.cell_state.zero_()
        self.decoder_output.zero_()

        self.sequence_buffer.fill_(self.pad_token_id)
        self.sequence_buffer[:, 0] = self.blank_token_id
        self.duration_buffer.zero_()
        self.frame_start_buffer.zero_()
        self.write_position.fill_(1)

    def _capture(
        self,
        encoder_states: torch.Tensor,
        valid_lengths: torch.Tensor,
        encoder_finite: torch.Tensor,
    ) -> None:
        """Warm up on a side stream, restore the state, and capture one step."""

        side_stream = torch.cuda.Stream(self.device)
        side_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side_stream):
            for _ in range(self.WARMUP_STEPS):
                self._step()
        torch.cuda.current_stream(self.device).wait_stream(side_stream)

        # The warm-up advanced the state; capturing does not run the step.
        self._load(encoder_states, valid_lengths, encoder_finite)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step()
        self.graph = graph

    # -------------------------------------------------------------------------
    # One decoding step (captured)
    # -------------------------------------------------------------------------

    def _step(self) -> None:
        """
        One greedy TDT step on the static buffers, in place.

        Mirrors ``ParakeetTDT._decode_eager`` operation for operation; see the
        comments there for the decoding rules.
        """

        decoder = self.model.decoder

        # Decoder: blank input keeps a row's cache, except on the first step,
        # when the eager cache is initialized from the start token for all rows.
        commit = (self.decoder_input_ids[:, -1] != self.blank_token_id) | self.first_step
        embeddings = decoder.embedding(self.decoder_input_ids)
        lstm_output, (new_hidden, new_cell) = decoder.lstm(
            embeddings,
            (self.hidden_state, self.cell_state),
        )
        new_output = decoder.decoder_projector(lstm_output)  # (B, 1, H)
        self.decoder_output.copy_(torch.where(commit[:, None, None], new_output, self.decoder_output))
        self.hidden_state.copy_(torch.where(commit[None, :, None], new_hidden, self.hidden_state))
        self.cell_state.copy_(torch.where(commit[None, :, None], new_cell, self.cell_state))
        self.first_step.fill_(False)

        # Joint on the current frame of each row.
        safe_frame_indices = self.frame_indices.clamp(max=self.frames - 1)
        current_encoder_states = self.encoder_states[
            self.batch_indices,
            safe_frame_indices,
            None,
            :,
        ]  # (B, 1, H)
        logits = self.model.joint(self.decoder_output, current_encoder_states).squeeze(1)
        token_ids = logits[:, : self.vocab_size].argmax(dim=-1)
        frame_advance = logits[:, self.vocab_size :].argmax(dim=-1)
        if not self.duration_classes_are_frames:
            frame_advance = self.model.duration_values[frame_advance]

        # Per-frame symbol guard and forced progress.
        active = ~self.finished
        blank_mask = token_ids == self.blank_token_id
        zero_advance = (frame_advance == 0) & active
        stays_on_frame = zero_advance & ~blank_mask
        self.symbols_on_frame.copy_((self.symbols_on_frame % self.max_symbols + 1) * stays_on_frame)
        guard_fires = self.symbols_on_frame == self.max_symbols
        self.forced_advances.add_(guard_fires)
        frame_advance = frame_advance + (zero_advance & (blank_mask | guard_fires))

        # Outputs at the device-side write position.
        emitted_token_ids = torch.where(active, token_ids, self.pad_ids)
        emitted_durations = frame_advance * active
        emitted_frame_starts = torch.where(active, self.frame_indices, self.inactive_frame)
        self.sequence_buffer.index_copy_(1, self.write_position, emitted_token_ids[:, None])
        self.duration_buffer.index_copy_(1, self.write_position, emitted_durations[:, None])
        self.frame_start_buffer.index_copy_(1, self.write_position, emitted_frame_starts[:, None])
        self.write_position.add_(1)

        # Advance.
        self.frame_indices.add_(emitted_durations)
        self.finished.copy_(self.finished | (self.frame_indices >= self.valid_lengths))
        next_input_ids = torch.where(self.finished, self.blank_ids, emitted_token_ids)
        self.decoder_input_ids.copy_(next_input_ids[:, None])
