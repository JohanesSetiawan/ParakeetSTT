"""
Audio input boundary for the standalone Parakeet runtime.

This module owns RIFF/WAV parsing and waveform normalization. It deliberately
uses only Python's standard library and PyTorch, so the model runtime does not
depend on SoundFile, Librosa, NumPy, or an audio framework.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch


# =============================================================================
# RIFF/WAV parsing
# =============================================================================
# WAV files are RIFF containers made of little-endian chunks. The parser keeps
# the format boundary here so feature extraction receives one predictable
# mono-float waveform and never needs to know whether the source was PCM or
# IEEE float audio.
# =============================================================================


def read_wav(path: Path) -> tuple[torch.Tensor, int]:
    """
    Read PCM or IEEE-float RIFF/WAV audio into a contiguous mono tensor.

    Supported payloads are IEEE float32, IEEE float64, signed PCM16, and signed
    PCM32. Multi-channel samples are averaged after decoding. The returned
    waveform has shape ``(frames,)`` and values in the normalized float range
    expected by the Parakeet feature extractor.

    Args:
        path: WAV file path.

    Returns:
        A tuple containing the mono float32 waveform and its sample rate.

    Raises:
        ValueError: If the container, chunks, encoding, or sample alignment is
            unsupported or malformed.
    """

    raw_bytes = path.read_bytes()
    if raw_bytes[:4] != b"RIFF" or raw_bytes[8:12] != b"WAVE":
        raise ValueError(f"Unsupported WAV container in {path}")

    audio_format: int | None = None
    channel_count: int | None = None
    sample_rate: int | None = None
    bits_per_sample: int | None = None
    audio_payload: bytes | None = None
    cursor = 12

    # Skip unknown RIFF chunks while honoring the mandatory even-byte chunk
    # alignment. This lets metadata chunks coexist with fmt/data chunks.
    while cursor + 8 <= len(raw_bytes):
        chunk_id = raw_bytes[cursor : cursor + 4]
        chunk_size = struct.unpack_from("<I", raw_bytes, cursor + 4)[0]
        chunk_start = cursor + 8
        chunk_end = chunk_start + chunk_size

        if chunk_end > len(raw_bytes):
            raise ValueError(f"WAV chunk {chunk_id!r} exceeds file size in {path}")

        chunk_data = raw_bytes[chunk_start:chunk_end]
        if chunk_id == b"fmt ":
            if len(chunk_data) < 16:
                raise ValueError(f"WAV fmt chunk is truncated in {path}")
            (
                audio_format,
                channel_count,
                sample_rate,
                _byte_rate,
                _block_align,
                bits_per_sample,
            ) = struct.unpack_from("<HHIIHH", chunk_data)
        elif chunk_id == b"data":
            audio_payload = chunk_data

        cursor = chunk_end + (chunk_size & 1)

    if None in {audio_format, channel_count, sample_rate, bits_per_sample}:
        raise ValueError(f"WAV fmt chunk is missing in {path}")
    if audio_payload is None:
        raise ValueError(f"WAV data chunk is missing in {path}")
    if channel_count <= 0:
        raise ValueError(f"WAV channel count must be positive in {path}")

    # Decode the sample container first. Scaling is applied only for integer
    # PCM because IEEE float WAV already stores normalized floating values.
    if audio_format == 3 and bits_per_sample == 32:
        samples = torch.frombuffer(bytearray(audio_payload), dtype=torch.float32).clone()
    elif audio_format == 3 and bits_per_sample == 64:
        samples = torch.frombuffer(bytearray(audio_payload), dtype=torch.float64).clone().float()
    elif audio_format == 1 and bits_per_sample == 16:
        samples = torch.frombuffer(bytearray(audio_payload), dtype=torch.int16).clone().float()
        samples = samples / 32768.0
    elif audio_format == 1 and bits_per_sample == 32:
        samples = torch.frombuffer(bytearray(audio_payload), dtype=torch.int32).clone().float()
        samples = samples / 2147483648.0
    else:
        raise ValueError(
            f"Unsupported WAV encoding format={audio_format}, bits_per_sample={bits_per_sample}"
        )

    if samples.numel() % channel_count != 0:
        raise ValueError(f"WAV payload does not align to {channel_count} channels in {path}")

    samples = samples.reshape(-1, channel_count)
    if channel_count > 1:
        # Channel averaging is the explicit mono conversion policy used before
        # feature extraction; it avoids an external resampling/audio dependency.
        samples = samples.mean(dim=1)
    else:
        samples = samples[:, 0]

    return samples.contiguous(), sample_rate
