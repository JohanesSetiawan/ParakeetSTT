"""
Bounded media inspection and segment decoding for offline inference.

The native model path still receives only PyTorch tensors. This boundary uses
soundfile/libsndfile for container decoding, because neither the Python
standard library nor PyTorch decodes MP3. Formats libsndfile cannot open fall
back to FFmpeg, resolved from ``FFMPEG_BINARY``/``FFPROBE_BINARY`` or ``PATH``.
Source segments are sought and read in bounded blocks, downmixed to mono, and
resampled to the checkpoint rate before feature extraction.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import soundfile
import torch
from torch.nn import functional as torch_functional

from ..inference.planning import AudioMetadata


# soundfile raises LibsndfileError (a RuntimeError) for unsupported or corrupt
# containers, and OSError for unreadable paths. Anything else is a real bug and
# must propagate instead of silently triggering the FFmpeg fallback.
SOUNDFILE_OPEN_ERRORS = (RuntimeError, OSError)


class CodecUnavailableError(RuntimeError):
    """
    FFprobe/FFmpeg is not installed, so a non-libsndfile format cannot be read.

    This is a per-file limitation (the file is skipped as unreadable). A codec
    path that is configured but wrong raises a plain RuntimeError instead,
    because that is a setup mistake that must stop the run.
    """


@dataclass(frozen=True)
class DecodedSegment:
    """
    A bounded mono target-rate waveform plus input health facts.

    Attributes:
        waveform: Float32 samples ``(T,)``; non-finite input is replaced by 0.
        sample_rate: Rate of ``waveform`` (the checkpoint rate).
        source_start_frame: First target-rate frame covered.
        source_end_frame: One past the last target-rate frame covered.
        source_sample_rate: Container sample rate before resampling.
        source_channels: Container channel count before downmixing.
        rms: Root mean square of ``waveform``; exactly 0.0 means digital silence.
        finite: False when the decoded input contained NaN or Inf.
    """

    waveform: torch.Tensor
    sample_rate: int
    source_start_frame: int
    source_end_frame: int
    source_sample_rate: int
    source_channels: int
    rms: float
    finite: bool


# =============================================================================
# External codec resolution
# =============================================================================


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
        raise CodecUnavailableError(
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
        "stream=sample_rate,channels,duration,codec_name",
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
    """Parse a finite positive FFprobe field with an actionable error."""

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"FFprobe returned invalid {name}: {value!r}") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"FFprobe returned invalid {name}: {value!r}")
    return number


# =============================================================================
# Metadata
# =============================================================================


def _require_openable(path: Path) -> None:
    """
    Re-raise OS-level open failures that libsndfile reports as format errors.

    libsndfile turns "file missing", "access denied", and "too many open
    files" into the same error as "unknown format". Only the last one may send
    a file to the FFmpeg fallback; the others must surface as they are.
    """

    with path.open("rb"):
        pass


def _metadata_from_probe(path: Path, probe: dict[str, object]) -> AudioMetadata:
    """
    Convert FFprobe output into planner metadata.

    The sample count is ``duration * sample_rate``. FFprobe's ``nb_frames`` is
    deliberately ignored: for audio streams it counts codec packets (1024
    samples each for AAC), not samples, and using it shrank a 13.8 s M4A file
    to 0.014 s.
    """

    stream = probe["stream"]
    if not isinstance(stream, dict):
        raise ValueError(f"FFprobe returned an invalid audio stream for {path}")

    sample_rate = int(_probe_number(stream.get("sample_rate"), "sample_rate"))
    channels = int(_probe_number(stream.get("channels"), "channels"))

    duration = stream.get("duration")
    if duration in {None, "N/A"}:
        format_data = probe.get("format")
        duration = format_data.get("duration") if isinstance(format_data, dict) else None
    duration_seconds = _probe_number(duration, "duration")

    return AudioMetadata(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        frame_count=max(1, round(duration_seconds * sample_rate)),
        format_name=str(stream.get("codec_name") or "FFmpeg audio"),
    )


def inspect_media(path: Path) -> AudioMetadata:
    """
    Read container metadata without decoding samples.

    Raises:
        ValueError: If neither libsndfile nor FFprobe can read an audio stream,
            including when FFprobe is not installed. The message carries both
            backend errors.
        OSError: If the file itself cannot be opened (missing, permission,
            too many open files); that is not a format problem.
        RuntimeError: If FFPROBE_BINARY is set but points nowhere.
    """

    try:
        information = soundfile.info(str(path))
    except SOUNDFILE_OPEN_ERRORS as soundfile_error:
        _require_openable(path)
        try:
            return _metadata_from_probe(path, _run_ffprobe(path))
        except (CodecUnavailableError, ValueError) as probe_error:
            raise ValueError(
                f"Unsupported or unreadable media {path}: "
                f"soundfile: {soundfile_error}; ffprobe: {probe_error}"
            ) from probe_error

    return AudioMetadata(
        path=path,
        sample_rate=int(information.samplerate),
        channels=int(information.channels),
        frame_count=int(information.frames),
        format_name=str(information.format),
    )


# =============================================================================
# Waveform helpers
# =============================================================================


def _resample_waveform(
    waveform: torch.Tensor,
    source_rate: int,
    target_rate: int,
    target_frames: int,
) -> torch.Tensor:
    """Resample a bounded mono waveform with deterministic linear interpolation."""

    if target_frames <= 0:
        return torch.empty(0, dtype=torch.float32)
    if source_rate == target_rate and waveform.numel() == target_frames:
        return waveform.to(dtype=torch.float32).contiguous()
    if waveform.numel() == 0:
        return torch.zeros(target_frames, dtype=torch.float32)

    resampled = torch_functional.interpolate(
        waveform.reshape(1, 1, -1),
        size=target_frames,
        mode="linear",
        align_corners=False,
    )  # (1, 1, target_frames)
    return resampled.reshape(-1).to(dtype=torch.float32).contiguous()


def _build_decoded_segment(
    waveform: torch.Tensor,
    target_start_frame: int,
    target_end_frame: int,
    target_sample_rate: int,
    *,
    source_sample_rate: int,
    source_channels: int,
) -> DecodedSegment:
    """Sanitize non-finite samples and record input health facts."""

    finite = bool(torch.isfinite(waveform).all())
    if not finite:
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

    if waveform.numel():
        rms = float(torch.sqrt(torch.mean(waveform.square())))
    else:
        rms = 0.0

    return DecodedSegment(
        waveform=waveform.contiguous(),
        sample_rate=target_sample_rate,
        source_start_frame=target_start_frame,
        source_end_frame=target_end_frame,
        source_sample_rate=source_sample_rate,
        source_channels=source_channels,
        rms=rms,
        finite=finite,
    )


# =============================================================================
# FFmpeg fallback decoding
# =============================================================================


def _read_ffmpeg_segment(
    path: Path,
    target_start_frame: int,
    target_end_frame: int,
    target_sample_rate: int,
) -> DecodedSegment:
    """Decode one bounded mono segment through the FFmpeg fallback gateway."""

    executable = _codec_binary("FFMPEG_BINARY", "ffmpeg")
    if executable is None:
        raise CodecUnavailableError(
            "The media format is not supported by soundfile/libsndfile and "
            "ffmpeg was not found in PATH. Set FFMPEG_BINARY externally."
        )

    target_frames = target_end_frame - target_start_frame
    command = [
        executable,
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{target_start_frame / target_sample_rate:.9f}",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-t",
        f"{target_frames / target_sample_rate:.9f}",
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
    # FFmpeg may return a few samples more or fewer than requested around codec
    # frame boundaries; the planner's sample budget is the contract.
    if waveform.numel() > target_frames:
        waveform = waveform[:target_frames]
    elif waveform.numel() < target_frames:
        waveform = torch_functional.pad(waveform, (0, target_frames - waveform.numel()))

    return _build_decoded_segment(
        waveform,
        target_start_frame,
        target_end_frame,
        target_sample_rate,
        source_sample_rate=target_sample_rate,
        source_channels=1,
    )


# =============================================================================
# Persistent per-file decoder session
# =============================================================================


class MediaSession:
    """
    Keep one decoder handle open while a file's chunks are processed.

    Chunks of one file must be read in order through
    :meth:`read_sequential_segment` so overlap samples are reused instead of
    decoded twice.
    """

    def __init__(self, path: Path) -> None:
        """Open a soundfile handle when libsndfile supports the input."""

        self.path = path
        self._source: soundfile.SoundFile | None
        try:
            self._source = soundfile.SoundFile(str(path), mode="r")
        except SOUNDFILE_OPEN_ERRORS:
            _require_openable(path)
            # A format libsndfile cannot decode: FFmpeg fallback, with
            # per-segment subprocesses spawned on demand.
            self._source = None

    def __enter__(self) -> "MediaSession":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the persistent decoder handle if one is open."""

        if self._source is not None:
            self._source.close()
            self._source = None

    def _decode_target_range(
        self,
        target_start_frame: int,
        target_end_frame: int,
        target_sample_rate: int,
    ) -> torch.Tensor:
        """
        Decode ``[start, end)`` target-rate frames as a mono float32 tensor.

        The source range is widened to whole source samples (floor/ceil). A read
        that ends early because the range rounds past end-of-file is zero-padded
        to the requested source length, so resampling never stretches the tail.
        """

        if self._source is None:
            raise RuntimeError("Decoder handle is closed or unavailable")

        source_rate = int(self._source.samplerate)
        source_channels = int(self._source.channels)
        source_start = math.floor(target_start_frame * source_rate / target_sample_rate)
        source_end = math.ceil(target_end_frame * source_rate / target_sample_rate)
        requested_frames = max(0, source_end - source_start)

        self._source.seek(source_start)
        decoded = self._source.read(
            frames=requested_frames,
            dtype="float32",
            always_2d=True,
        )  # (frames, channels)
        waveform = torch.from_numpy(decoded.copy())
        if waveform.ndim != 2 or waveform.shape[1] != source_channels:
            raise ValueError(
                f"Codec gateway returned unexpected shape for {self.path}: {tuple(waveform.shape)}"
            )

        if source_channels > 1:
            mono = waveform.mean(dim=1)
        else:
            mono = waveform[:, 0]

        missing_frames = requested_frames - mono.numel()
        if missing_frames > 0:
            mono = torch_functional.pad(mono, (0, missing_frames))

        return _resample_waveform(
            mono,
            source_rate,
            target_sample_rate,
            target_end_frame - target_start_frame,
        )

    def read_segment(
        self,
        target_start_frame: int,
        target_end_frame: int,
        target_sample_rate: int,
    ) -> DecodedSegment:
        """Read one bounded segment while reusing this file's decoder handle."""

        if target_start_frame < 0 or target_end_frame < target_start_frame:
            raise ValueError(
                f"Invalid segment bounds [{target_start_frame}, {target_end_frame}) for {self.path}"
            )

        if self._source is None:
            return _read_ffmpeg_segment(
                self.path,
                target_start_frame,
                target_end_frame,
                target_sample_rate,
            )

        waveform = self._decode_target_range(
            target_start_frame,
            target_end_frame,
            target_sample_rate,
        )
        return _build_decoded_segment(
            waveform,
            target_start_frame,
            target_end_frame,
            target_sample_rate,
            source_sample_rate=int(self._source.samplerate),
            source_channels=int(self._source.channels),
        )

    def read_sequential_segment(
        self,
        source_start_frame: int,
        source_end_frame: int,
        target_sample_rate: int,
        previous_source_end_frame: int | None = None,
        previous_waveform: torch.Tensor | None = None,
    ) -> tuple[DecodedSegment, torch.Tensor]:
        """
        Read the next chunk, reusing the overlap tail of the previous chunk.

        Only ``previous_source_end_frame - source_start_frame`` samples are
        reused. When that is zero or negative (no overlap, or a gap), the chunk
        is decoded fresh; reusing "the last 0 samples" must never be expressed
        as ``tensor[-0:]``, which in Python selects the whole tensor.

        Args:
            source_start_frame: First target-rate frame of this chunk.
            source_end_frame: One past its last target-rate frame.
            target_sample_rate: Checkpoint sample rate.
            previous_source_end_frame: End frame of the previous chunk of the
                same file, or None for the first chunk.
            previous_waveform: Waveform returned for the previous chunk.

        Returns:
            The decoded segment and the waveform to pass as
            ``previous_waveform`` for the next chunk.

        Raises:
            ValueError: If the claimed overlap is longer than the previous
                waveform, which means chunks were read out of order.
        """

        reusable_frames = 0
        if previous_waveform is not None and previous_source_end_frame is not None:
            reusable_frames = previous_source_end_frame - source_start_frame
            if reusable_frames > previous_waveform.numel():
                raise ValueError(
                    f"Overlap of {reusable_frames} frames exceeds the previous chunk "
                    f"({previous_waveform.numel()} frames) for {self.path}; "
                    "chunks must be read in order"
                )

        if self._source is None or reusable_frames <= 0:
            segment = self.read_segment(
                source_start_frame,
                source_end_frame,
                target_sample_rate,
            )
            return segment, segment.waveform

        # previous_source_end_frame is not None here because reusable_frames > 0.
        new_start_frame = min(previous_source_end_frame, source_end_frame)
        new_waveform = self._decode_target_range(
            new_start_frame,
            source_end_frame,
            target_sample_rate,
        )
        overlap_start = previous_waveform.numel() - reusable_frames
        overlap_tail = previous_waveform[overlap_start:]
        waveform = torch.cat((overlap_tail, new_waveform))

        segment = _build_decoded_segment(
            waveform,
            source_start_frame,
            source_end_frame,
            target_sample_rate,
            source_sample_rate=int(self._source.samplerate),
            source_channels=int(self._source.channels),
        )
        return segment, segment.waveform


def open_media_session(path: Path) -> MediaSession:
    """Open one reusable decoder session for sequential bounded processing."""

    return MediaSession(path)
