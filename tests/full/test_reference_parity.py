"""
Parity with the reference implementation, Hugging Face `ParakeetForTDT`.

`transformers` is a verification tool only; it is never imported by the
runtime. The reference runs on CPU so it does not compete with the session
model for GPU memory on small cards.
"""

from __future__ import annotations

import pytest
import soundfile
import torch

transformers = pytest.importorskip("transformers", reason="transformers not installed (verification-only dependency)")


@pytest.fixture(scope="module")
def reference(real_settings):
    processor = transformers.AutoProcessor.from_pretrained(str(real_settings.paths.weights_dir))
    model = transformers.ParakeetForTDT.from_pretrained(
        str(real_settings.paths.weights_dir),
        dtype=torch.float32,
    ).eval()
    return processor, model


# transformers warns that generate() falls back to its default max_length;
# the TDT loop stops on encoder exhaustion long before that bound.
@pytest.mark.filterwarnings("ignore:Using the model-agnostic default `max_length`:UserWarning")
def test_single_chunk_transcripts_match_the_reference(reference, speech_clips) -> None:
    processor, model = reference
    for clip in speech_clips:
        if clip.chunks != 1:
            continue
        audio, rate = soundfile.read(str(clip.path), dtype="float32")
        inputs = processor(audio, sampling_rate=rate, return_tensors="pt")

        with torch.inference_mode():
            output = model.generate(**inputs, return_dict_in_generate=True)
        reference_text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()

        assert reference_text == clip.expected_transcript, clip.clip_id
