"""Anomalous but realistic inputs through the real model: every file gets an honest status."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile

from src.inference.offline import FileStatus, OfflineTranscriber

SAMPLE_RATE = 16000


@pytest.fixture(scope="module")
def transcriber(real_model, real_settings) -> OfflineTranscriber:
    model, configuration = real_model
    return OfflineTranscriber(model, configuration, real_settings.inference)


def transcribe_array(transcriber: OfflineTranscriber, tmp_path: Path, name: str, audio: np.ndarray):
    path = tmp_path / f"{name}.wav"
    soundfile.write(str(path), audio.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
    return transcriber.transcribe([path]).files[0]


def test_digital_silence_is_no_speech(transcriber, tmp_path: Path) -> None:
    file_result = transcribe_array(transcriber, tmp_path, "silence", np.zeros(3 * SAMPLE_RATE))

    assert file_result.status is FileStatus.NO_SPEECH
    assert file_result.transcript == ""


@pytest.mark.parametrize("amplitude", [0.01, 0.05])
def test_background_noise_is_an_empty_transcript_not_invented_words(transcriber, tmp_path: Path, amplitude: float) -> None:
    noise = np.random.default_rng(0).standard_normal(3 * SAMPLE_RATE) * amplitude

    file_result = transcribe_array(transcriber, tmp_path, f"noise_{amplitude}", noise)

    assert file_result.status is FileStatus.EMPTY_TRANSCRIPT
    assert file_result.transcript == ""


def test_corrupted_samples_are_flagged_and_speech_still_transcribed(transcriber, speech_clips, tmp_path: Path) -> None:
    clip = next(clip for clip in speech_clips if clip.chunks == 1)
    audio, _rate = soundfile.read(str(clip.path), dtype="float32")
    audio[1000:1010] = np.nan

    file_result = transcribe_array(transcriber, tmp_path, "nan_clip", audio)

    assert file_result.status is FileStatus.INPUT_NONFINITE
    assert file_result.transcript == clip.expected_transcript


def test_resampled_stereo_input_is_transcribed(transcriber, speech_clips, tmp_path: Path) -> None:
    """A 48 kHz stereo copy of a 16 kHz clip goes through downmix and resampling."""

    clip = next(clip for clip in speech_clips if clip.chunks == 1)
    audio, _rate = soundfile.read(str(clip.path), dtype="float32")
    upsampled = np.repeat(audio, 3)
    stereo = np.stack([upsampled, upsampled], axis=1)
    path = tmp_path / "stereo48k.wav"
    soundfile.write(str(path), stereo, 48000, subtype="FLOAT")

    file_result = transcriber.transcribe([path]).files[0]

    assert file_result.status is FileStatus.OK
    assert file_result.duration_seconds == pytest.approx(clip.duration_seconds, abs=0.01)
    assert len(file_result.transcript.split()) >= len(clip.reference_text.split()) - 2
