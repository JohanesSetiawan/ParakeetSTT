"""
Bounded media inspection and segment decoding for offline inference.

The native model path still receives only PyTorch tensors. This boundary uses
soundfile/libsndfile for container decoding, because neither the Python
standard library nor PyTorch decodes MP3. Formats libsndfile cannot open fall
back to FFmpeg, resolved from ``FFMPEG_BINARY``/``FFPROBE_BINARY`` or ``PATH``.
Source segments are sought and read in bounded blocks, downmixed to mono, and
resampled to the checkpoint rate with an anti-aliased windowed-sinc filter
before feature extraction.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy
import soundfile
import torch
from torch.nn import functional as torch_functional

from ..inference.planning import AudioMetadata
from .resampling import plan_block, resample_block


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


def _downmix(interleaved: numpy.ndarray) -> numpy.ndarray:
    """
    Average ``(frames, channels)`` float32 samples into one contiguous channel.

    Adding whole channel columns is about 2.7 times faster than
    ``torch.mean(dim=1)`` over the interleaved samples (0.41 s against 1.1 s
    for a 74-minute stereo file) and gives the same values for mono and
    stereo: ``(a + b) / 2`` is exact either way.
    """

    channels = interleaved.shape[1]
    if channels == 1:
        return numpy.ascontiguousarray(interleaved[:, 0])
    mono = interleaved[:, 0] + interleaved[:, 1]
    for channel in range(2, channels):
        mono += interleaved[:, channel]
    numpy.divide(mono, channels, out=mono)
    return mono


def slice_segment(segment: DecodedSegment, start_frame: int, end_frame: int) -> DecodedSegment:
    """
    The part ``[start_frame, end_frame)`` of an already decoded segment.

    Lets several overlapping windows share one decode. The slice keeps the
    parent's ``finite`` flag: non-finite samples were zeroed in the parent,
    so a slice cannot tell whether it contained any, and reporting them for
    the whole read is the conservative choice.
    """

    if not segment.source_start_frame <= start_frame <= end_frame <= segment.source_end_frame:
        raise ValueError(
            f"Slice [{start_frame}, {end_frame}) lies outside the segment "
            f"[{segment.source_start_frame}, {segment.source_end_frame})"
        )
    offset = segment.source_start_frame
    waveform = segment.waveform[start_frame - offset : end_frame - offset]
    # The parent already replaced non-finite samples, so only the RMS is new.
    rms = float(torch.sqrt(torch.mean(waveform.square()))) if waveform.numel() else 0.0
    return DecodedSegment(
        waveform=waveform.contiguous(),
        sample_rate=segment.sample_rate,
        source_start_frame=start_frame,
        source_end_frame=end_frame,
        source_sample_rate=segment.source_sample_rate,
        source_channels=segment.source_channels,
        rms=rms,
        finite=segment.finite,
    )


# =============================================================================
# FFmpeg fallback decoding
# =============================================================================


def _require_ffmpeg() -> str:
    """Resolve the FFmpeg executable or explain how to provide one."""

    executable = _codec_binary("FFMPEG_BINARY", "ffmpeg")
    if executable is None:
        raise CodecUnavailableError(
            "The media format is not supported by soundfile/libsndfile and "
            "ffmpeg was not found in PATH. Set FFMPEG_BINARY externally."
        )
    return executable


def _ffmpeg_decode_command(
    executable: str,
    path: Path,
    target_sample_rate: int,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> list[str]:
    """
    FFmpeg arguments that decode the first audio stream to mono float32.

    The one-off segment decode and the persistent stream share this command,
    so both paths always downmix, resample, and encode the same way.
    """

    command = [executable, "-v", "error", "-nostdin"]
    if start_seconds is not None:
        command += ["-ss", f"{start_seconds:.9f}"]
    command += ["-i", str(path), "-map", "0:a:0", "-vn", "-sn", "-dn"]
    if duration_seconds is not None:
        command += ["-t", f"{duration_seconds:.9f}"]
    command += ["-ac", "1", "-ar", str(target_sample_rate), "-f", "f32le", "pipe:1"]
    return command


def _float32_samples(payload: bytes) -> torch.Tensor:
    """
    Interpret raw little-endian float32 bytes from FFmpeg as a 1-D tensor.

    An empty payload (the stream already ended) is a valid zero-length
    result, which torch.frombuffer would reject.
    """

    if not payload:
        return torch.zeros(0, dtype=torch.float32)
    return torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()


def _read_ffmpeg_segment(
    path: Path,
    target_start_frame: int,
    target_end_frame: int,
    target_sample_rate: int,
) -> DecodedSegment:
    """Decode one bounded mono segment through the FFmpeg fallback gateway."""

    target_frames = target_end_frame - target_start_frame
    command = _ffmpeg_decode_command(
        _require_ffmpeg(),
        path,
        target_sample_rate,
        start_seconds=target_start_frame / target_sample_rate,
        duration_seconds=target_frames / target_sample_rate,
    )
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"FFmpeg could not decode media segment from {path}: {error}") from error

    waveform = _float32_samples(completed.stdout)
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
# Persistent FFmpeg stream (sequential fallback decoding)
# =============================================================================


class _FfmpegStream:
    """
    One FFmpeg process that decodes a file to mono target-rate float32.

    Chunks of a decode stream are read in order, so each new piece continues
    exactly where the previous one stopped. One process per stream replaces
    one process (plus a seek) per chunk, and FFmpeg's resampler runs over the
    stream continuously instead of restarting at every chunk. A stream that
    covers a later region of the file starts there with FFmpeg's own seek,
    instead of decoding and discarding everything before it.
    """

    STDERR_LINES_KEPT = 20

    def __init__(self, path: Path, target_sample_rate: int, start_frame: int = 0) -> None:
        start_seconds = start_frame / target_sample_rate if start_frame > 0 else None
        command = _ffmpeg_decode_command(
            _require_ffmpeg(),
            path,
            target_sample_rate,
            start_seconds=start_seconds,
        )
        self.path = path
        self.position = start_frame
        self.exhausted = False
        self._stderr_tail: deque[str] = deque(maxlen=self.STDERR_LINES_KEPT)
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            # Same contract as the one-off decode: a per-file decode failure.
            raise ValueError(f"FFmpeg could not start decoding {path}: {error}") from error
        # Drain stderr on a thread: a full stderr pipe would block FFmpeg and,
        # through the stdout reads below, this process too.
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())

    def read(self, start: int, end: int) -> torch.Tensor:
        """
        Return target samples ``[start, end)``; ``start`` must not be behind
        the stream. Samples between the current position and ``start`` are
        skipped. Samples past the end of the audio are zeros.

        Raises:
            ValueError: If FFmpeg exits with an error.
        """

        if start < self.position:
            raise ValueError(f"FFmpeg stream for {self.path} is already past sample {start}")
        skipped = self._read_bytes((start - self.position) * 4)
        self.position += len(skipped) // 4
        payload = self._read_bytes((end - start) * 4)
        self.position = start + len(payload) // 4

        waveform = _float32_samples(payload)
        missing = (end - start) - waveform.numel()
        if missing > 0:
            waveform = torch_functional.pad(waveform, (0, missing))
            self.position = end
        return waveform

    def _read_bytes(self, count: int) -> bytes:
        assert self._process.stdout is not None
        chunks: list[bytes] = []
        remaining = count
        while remaining > 0 and not self.exhausted:
            data = self._process.stdout.read(remaining)
            if not data:
                self._finish()
                break
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def _finish(self) -> None:
        """Mark end of stream and fail loudly if FFmpeg reported an error."""

        self.exhausted = True
        return_code = self._process.wait()
        # The process has exited, so stderr is at EOF and the drain thread
        # ends promptly; joining here makes the error tail below complete.
        self._stderr_thread.join()
        if return_code != 0:
            detail = " | ".join(self._stderr_tail) or "no error output"
            raise ValueError(f"FFmpeg failed while decoding {self.path} (exit {return_code}): {detail}")

    def close(self) -> None:
        """Stop FFmpeg and release its pipes and the stderr thread."""

        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        # Closing stdout first lets a killed FFmpeg's pipes drain; the stderr
        # thread is joined before its pipe is closed so it never reads from a
        # closed file.
        if self._process.stdout is not None:
            self._process.stdout.close()
        self._stderr_thread.join()
        if self._process.stderr is not None:
            self._process.stderr.close()


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
        # FFmpeg fallback stream, started on the first sequential read.
        self._stream: _FfmpegStream | None = None
        try:
            self._source = soundfile.SoundFile(str(path), mode="r")
        except SOUNDFILE_OPEN_ERRORS:
            _require_openable(path)
            # A format libsndfile cannot decode: FFmpeg fallback.
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
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _read_mono(self, source_start: int, source_end: int) -> torch.Tensor:
        """
        Read ``[source_start, source_end)`` source samples as mono float32.

        Positions before the file start or past its end are returned as zeros,
        which is what the resampling filter expects outside the signal.
        """

        if self._source is None:
            raise RuntimeError("Decoder handle is closed or unavailable")

        source_channels = int(self._source.channels)
        file_frames = int(self._source.frames)
        read_start = min(max(0, source_start), file_frames)
        read_end = min(max(read_start, source_end), file_frames)

        self._source.seek(read_start)
        decoded = self._source.read(
            frames=read_end - read_start,
            dtype="float32",
            always_2d=True,
        )  # (frames, channels)
        if decoded.ndim != 2 or decoded.shape[1] != source_channels:
            raise ValueError(
                f"Codec gateway returned unexpected shape for {self.path}: {tuple(decoded.shape)}"
            )
        mono = torch.from_numpy(_downmix(decoded))

        # A short read at end-of-file and positions outside the file become
        # zeros, so the returned length always matches the request.
        leading_zeros = read_start - source_start
        trailing_zeros = (source_end - source_start) - leading_zeros - mono.numel()
        return torch_functional.pad(mono, (leading_zeros, max(0, trailing_zeros)))

    def _decode_target_range(
        self,
        target_start_frame: int,
        target_end_frame: int,
        target_sample_rate: int,
    ) -> torch.Tensor:
        """
        Decode ``[start, end)`` target-rate frames as a mono float32 tensor.

        At the target rate the samples are read directly. Otherwise a block
        aligned to the resampling period, with filter context on both sides,
        is read and resampled (see ``resampling``). The result is the same as
        resampling the whole file and slicing it, so chunk pieces agree at
        every seam. Around each piece the filter context re-reads a few dozen
        source samples; the overlap itself is never decoded twice.
        """

        if self._source is None:
            raise RuntimeError("Decoder handle is closed or unavailable")

        source_rate = int(self._source.samplerate)
        if source_rate == target_sample_rate:
            return self._read_mono(target_start_frame, target_end_frame)

        plan = plan_block(target_start_frame, target_end_frame, source_rate, target_sample_rate)
        block = self._read_mono(plan.block_start, plan.block_end)
        return resample_block(block, plan, source_rate, target_sample_rate)

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

        if self._source is None:
            return self._read_sequential_through_ffmpeg(
                source_start_frame,
                source_end_frame,
                target_sample_rate,
                previous_waveform,
                reusable_frames,
            )

        if reusable_frames <= 0:
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


    def _read_sequential_through_ffmpeg(
        self,
        source_start_frame: int,
        source_end_frame: int,
        target_sample_rate: int,
        previous_waveform: torch.Tensor | None,
        reusable_frames: int,
    ) -> tuple[DecodedSegment, torch.Tensor]:
        """
        Continue the file's FFmpeg stream; fall back to a one-off decode only
        when the request lies behind the stream (never for in-order chunks).
        """

        # Same clamp as the soundfile path: an overlap reaching past this
        # chunk's end leaves nothing new to read.
        new_start_frame = min(source_start_frame + max(0, reusable_frames), source_end_frame)
        # A stream skips ahead by decoding and discarding. When the jump is
        # longer than the piece to read (a decode stream moving on to its
        # next block), starting FFmpeg again at the new position is cheaper.
        if self._stream is not None:
            skip = new_start_frame - self._stream.position
            if skip > source_end_frame - new_start_frame:
                self._stream.close()
                self._stream = None
        if self._stream is None:
            self._stream = _FfmpegStream(self.path, target_sample_rate, new_start_frame)

        if new_start_frame < self._stream.position:
            segment = _read_ffmpeg_segment(
                self.path,
                source_start_frame,
                source_end_frame,
                target_sample_rate,
            )
            return segment, segment.waveform

        new_waveform = self._stream.read(new_start_frame, source_end_frame)
        if reusable_frames > 0 and previous_waveform is not None:
            overlap_tail = previous_waveform[previous_waveform.numel() - reusable_frames :]
            waveform = torch.cat((overlap_tail, new_waveform))
        else:
            waveform = new_waveform

        segment = _build_decoded_segment(
            waveform,
            source_start_frame,
            source_end_frame,
            target_sample_rate,
            source_sample_rate=target_sample_rate,
            source_channels=1,
        )
        return segment, segment.waveform


def open_media_session(path: Path) -> MediaSession:
    """Open one reusable decoder session for sequential bounded processing."""

    return MediaSession(path)
