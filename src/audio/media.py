"""
Bounded media inspection and segment decoding for offline inference.

The native model path still receives only PyTorch tensors. This boundary uses
the already-installed ``soundfile``/libsndfile backend lazily for container
decoding, because Python's standard library and PyTorch do not decode MP3.
Source segments are sought and read in bounded blocks, downmixed to mono, and
resampled to the checkpoint rate before feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch
from torch.nn import functional as torch_functional

from ..inference.planning import AudioMetadata


@dataclass(frozen=True)
class DecodedSegment:
    """A bounded target-rate waveform and measured source quality values."""

    waveform: torch.Tensor
    sample_rate: int
    source_start_frame: int
    source_end_frame: int
    source_sample_rate: int
    source_channels: int
    rms: float
    peak: float
    clipping_ratio: float
    finite: bool


def _soundfile_module():
    """Load the optional codec gateway only when media decoding is requested."""

    try:
        return importlib.import_module("soundfile")
    except ImportError as error:
        raise RuntimeError(
            "Media decoding requires the existing soundfile/libsndfile gateway; "
            "installing or changing dependencies was not attempted."
        ) from error


def _codec_binary(environment_name: str, executable_name: str) -> str | None:
    """Resolve a fallback codec executable without embedding machine paths."""

    configured_path = os.environ.get(environment_name)
    if configured_path:
        configured = Path(configured_path).expanduser()
        if configured.is_file():
            return str(configured)
        raise RuntimeError(
            f"{environment_name} points to a missing executable: {configured}"
        )
    return shutil.which(executable_name)


def _run_ffprobe(path: Path) -> dict[str, object]:
    """Probe the first audio stream through an externally resolved FFprobe."""

    executable = _codec_binary("FFPROBE_BINARY", "ffprobe")
    if executable is None:
        raise RuntimeError(
            "The media format is not supported by soundfile/libsndfile and "
            "ffprobe was not found in PATH. Set FFPROBE_BINARY externally."
        )
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration,nb_frames,codec_name",
        "-show_entries",
        "format=format_name,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"FFprobe could not inspect media file {path}: {error}") from error
    document = json.loads(completed.stdout)
    streams = document.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise ValueError(f"No usable audio stream found in {path}")
    return {"stream": streams[0], "format": document.get("format", {})}


def _probe_number(value: object, name: str) -> float:
    """Parse a finite numeric FFprobe field with an actionable error."""

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"FFprobe returned invalid {name}: {value!r}") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"FFprobe returned invalid {name}: {value!r}")
    return number


def inspect_media(path: Path) -> AudioMetadata:
    """Read container metadata without allocating decoded sample storage."""

    try:
        soundfile = _soundfile_module()
        information = soundfile.info(str(path))
        return AudioMetadata(
            path=path,
            sample_rate=int(information.samplerate),
            channels=int(information.channels),
            frame_count=int(information.frames),
            format_name=str(information.format),
        )
    except Exception as soundfile_error:
        probe = _run_ffprobe(path)
        stream = probe["stream"]
        if not isinstance(stream, dict):
            raise ValueError(f"FFprobe returned an invalid audio stream for {path}")
        sample_rate = int(_probe_number(stream.get("sample_rate"), "sample_rate"))
        channels = int(_probe_number(stream.get("channels"), "channels"))
        duration = stream.get("duration")
        if duration in {None, "N/A"}:
            format_data = probe.get("format")
            duration = format_data.get("duration") if isinstance(format_data, dict) else None
        frame_count_value = stream.get("nb_frames")
        if frame_count_value in {None, "N/A"}:
            frame_count = max(1, round(_probe_number(duration, "duration") * sample_rate))
        else:
            frame_count = max(1, round(_probe_number(frame_count_value, "nb_frames")))
        return AudioMetadata(
            path=path,
            sample_rate=sample_rate,
            channels=channels,
            frame_count=frame_count,
            format_name=str(stream.get("codec_name") or "FFmpeg audio"),
        )


def _resample_waveform(
    waveform: torch.Tensor,
    source_rate: int,
    target_rate: int,
    target_frames: int,
) -> torch.Tensor:
    """Resample a bounded mono waveform with deterministic torch interpolation."""

    if target_frames <= 0:
        return torch.empty(0, dtype=torch.float32)
    if source_rate == target_rate and waveform.numel() == target_frames:
        return waveform.to(dtype=torch.float32).contiguous()
    if waveform.numel() == 0:
        return torch.zeros(target_frames, dtype=torch.float32)

    # Interpolation is performed on a (N, C, T) tensor and then trimmed to the
    # exact target interval. The planner supplies overlap, so a chunk boundary
    # never depends on an absent neighboring sample.
    resampled = torch_functional.interpolate(
        waveform.reshape(1, 1, -1),
        size=target_frames,
        mode="linear",
        align_corners=False,
    )
    return resampled.reshape(-1).to(dtype=torch.float32).contiguous()


def read_media_segment(
    path: Path,
    target_start_frame: int,
    target_end_frame: int,
    target_sample_rate: int,
) -> DecodedSegment:
    """Seek, decode, downmix, and resample one bounded target-rate segment."""

    if target_start_frame < 0 or target_end_frame < target_start_frame:
        raise ValueError("Media segment frame bounds are invalid")
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")

    try:
        soundfile = _soundfile_module()
        with soundfile.SoundFile(str(path), mode="r") as source:
            source_rate = int(source.samplerate)
            source_channels = int(source.channels)
            source_start = math.floor(target_start_frame * source_rate / target_sample_rate)
            source_end = math.ceil(target_end_frame * source_rate / target_sample_rate)
            source.seek(source_start)
            decoded = source.read(
                frames=max(0, source_end - source_start),
                dtype="float32",
                always_2d=True,
            )
    except Exception:
        return _read_ffmpeg_segment(
            path,
            target_start_frame,
            target_end_frame,
            target_sample_rate,
        )

    waveform = torch.from_numpy(decoded.copy())
    if waveform.ndim != 2 or waveform.shape[1] != source_channels:
        raise ValueError(f"Codec gateway returned unexpected shape for {path}: {waveform.shape}")
    if source_channels > 1:
        waveform = waveform.mean(dim=1)
    else:
        waveform = waveform[:, 0]

    target_frames = max(0, target_end_frame - target_start_frame)
    waveform = _resample_waveform(
        waveform,
        source_rate,
        target_sample_rate,
        target_frames,
    )
    finite_mask = torch.isfinite(waveform)
    finite = bool(finite_mask.all())
    if not finite:
        waveform = torch.nan_to_num(waveform)
    if waveform.numel():
        rms = float(torch.sqrt(torch.mean(waveform.square())).item())
        peak = float(waveform.abs().max().item())
        clipping_ratio = float((waveform.abs() >= 0.999).float().mean().item())
    else:
        rms = 0.0
        peak = 0.0
        clipping_ratio = 0.0

    return DecodedSegment(
        waveform=waveform,
        sample_rate=target_sample_rate,
        source_start_frame=target_start_frame,
        source_end_frame=target_end_frame,
        source_sample_rate=source_rate,
        source_channels=source_channels,
        rms=rms,
        peak=peak,
        clipping_ratio=clipping_ratio,
        finite=finite,
    )


def _read_ffmpeg_segment(
    path: Path,
    target_start_frame: int,
    target_end_frame: int,
    target_sample_rate: int,
) -> DecodedSegment:
    """Decode one bounded mono segment through the FFmpeg fallback gateway."""

    executable = _codec_binary("FFMPEG_BINARY", "ffmpeg")
    if executable is None:
        raise RuntimeError(
            "The media format is not supported by soundfile/libsndfile and "
            "ffmpeg was not found in PATH. Set FFMPEG_BINARY externally."
        )
    target_frames = target_end_frame - target_start_frame
    duration_seconds = target_frames / target_sample_rate
    start_seconds = target_start_frame / target_sample_rate
    command = [
        executable,
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{start_seconds:.9f}",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-t",
        f"{duration_seconds:.9f}",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"FFmpeg could not decode media segment from {path}: {error}") from error

    waveform = torch.frombuffer(bytearray(completed.stdout), dtype=torch.float32).clone()
    if waveform.numel() > target_frames:
        waveform = waveform[:target_frames]
    elif waveform.numel() < target_frames:
        waveform = torch.nn.functional.pad(waveform, (0, target_frames - waveform.numel()))
    return _build_decoded_segment(
        waveform,
        target_start_frame,
        target_end_frame,
        target_sample_rate,
        source_sample_rate=target_sample_rate,
        source_channels=1,
    )


def _build_decoded_segment(
    waveform: torch.Tensor,
    target_start_frame: int,
    target_end_frame: int,
    target_sample_rate: int,
    *,
    source_sample_rate: int,
    source_channels: int,
) -> DecodedSegment:
    """Compute common bounded-segment quality metadata."""

    finite_mask = torch.isfinite(waveform)
    finite = bool(finite_mask.all())
    if not finite:
        waveform = torch.nan_to_num(waveform)
    if waveform.numel():
        rms = float(torch.sqrt(torch.mean(waveform.square())).item())
        peak = float(waveform.abs().max().item())
        clipping_ratio = float((waveform.abs() >= 0.999).float().mean().item())
    else:
        rms = peak = clipping_ratio = 0.0
    return DecodedSegment(
        waveform=waveform.contiguous(),
        sample_rate=target_sample_rate,
        source_start_frame=target_start_frame,
        source_end_frame=target_end_frame,
        source_sample_rate=source_sample_rate,
        source_channels=source_channels,
        rms=rms,
        peak=peak,
        clipping_ratio=clipping_ratio,
        finite=finite,
    )


class MediaSession:
    """Keep one primary decoder handle open while a file's chunks are processed."""

    def __init__(self, path: Path) -> None:
        """Open a soundfile handle when the primary backend supports the input."""

        self.path = path
        self._source = None
        try:
            soundfile = _soundfile_module()
            self._source = soundfile.SoundFile(str(path), mode="r")
        except Exception:
            self._source = None

    def __enter__(self) -> "MediaSession":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the persistent primary decoder handle if one is open."""

        if self._source is not None:
            self._source.close()
            self._source = None

    def read_segment(
        self,
        target_start_frame: int,
        target_end_frame: int,
        target_sample_rate: int,
    ) -> DecodedSegment:
        """Read one bounded segment while reusing this file's decoder handle."""

        if self._source is None:
            return _read_ffmpeg_segment(
                self.path,
                target_start_frame,
                target_end_frame,
                target_sample_rate,
            )

        source_rate = int(self._source.samplerate)
        source_channels = int(self._source.channels)
        source_start = math.floor(target_start_frame * source_rate / target_sample_rate)
        source_end = math.ceil(target_end_frame * source_rate / target_sample_rate)
        self._source.seek(source_start)
        decoded = self._source.read(
            frames=max(0, source_end - source_start),
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(decoded.copy())
        if waveform.ndim != 2 or waveform.shape[1] != source_channels:
            raise ValueError(
                f"Codec gateway returned unexpected shape for {self.path}: {waveform.shape}"
            )
        waveform = waveform.mean(dim=1) if source_channels > 1 else waveform[:, 0]
        waveform = _resample_waveform(
            waveform,
            source_rate,
            target_sample_rate,
            target_end_frame - target_start_frame,
        )
        return _build_decoded_segment(
            waveform,
            target_start_frame,
            target_end_frame,
            target_sample_rate,
            source_sample_rate=source_rate,
            source_channels=source_channels,
        )

    def read_sequential_segment(
        self,
        source_start_frame: int,
        source_end_frame: int,
        target_sample_rate: int,
        previous_source_end_frame: int | None = None,
        previous_waveform: torch.Tensor | None = None,
    ) -> tuple[DecodedSegment, torch.Tensor]:
        """Read a source interval while reusing the previous decoded overlap tail."""

        if self._source is None:
            segment = self.read_segment(source_start_frame, source_end_frame, target_sample_rate)
            return segment, segment.waveform

        source_rate = int(self._source.samplerate)
        source_channels = int(self._source.channels)
        decode_start_frame = (
            source_start_frame
            if previous_source_end_frame is None
            else max(source_start_frame, previous_source_end_frame)
        )
        source_start = math.floor(decode_start_frame * source_rate / target_sample_rate)
        source_end = math.ceil(source_end_frame * source_rate / target_sample_rate)
        self._source.seek(source_start)
        decoded = self._source.read(
            frames=max(0, source_end - source_start),
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(decoded.copy())
        waveform = waveform.mean(dim=1) if source_channels > 1 else waveform[:, 0]
        new_waveform = _resample_waveform(
            waveform,
            source_rate,
            target_sample_rate,
            source_end_frame - decode_start_frame,
        )
        if previous_waveform is not None and previous_source_end_frame is not None:
            overlap_frames = max(0, previous_source_end_frame - source_start_frame)
            waveform = torch.cat((previous_waveform[-overlap_frames:], new_waveform))
        else:
            waveform = new_waveform
        segment = _build_decoded_segment(
            waveform,
            source_start_frame,
            source_end_frame,
            target_sample_rate,
            source_sample_rate=source_rate,
            source_channels=source_channels,
        )
        return segment, waveform


def open_media_session(path: Path) -> MediaSession:
    """Open one reusable decoder session for sequential bounded processing."""

    return MediaSession(path)