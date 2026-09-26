# Test audio source

The five `.flac` files in this directory are unmodified utterances from the
LibriSpeech corpus (dev-clean, speaker 1272), taken from the
`hf-internal-testing/librispeech_asr_dummy` subset on Hugging Face.

- Corpus: LibriSpeech ASR corpus, https://www.openslr.org/12/
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Authors: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur.
  "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015.
- The underlying recordings are public-domain LibriVox audiobooks.

| File | Duration (s) | Note |
|---|---|---|
| 1272-128104-0000.flac | 5.86 | single chunk |
| 1272-128104-0001.flac | 4.82 | single chunk |
| 1272-128104-0004.flac | 29.40 | three chunks with default settings; exercises chunk boundaries |
| 1272-135031-0009.flac | 1.91 | short utterance |
| 1272-141231-0000.flac | 4.65 | single chunk |

`manifest.json` stores each file's SHA-256, the LibriSpeech reference text, and
the transcripts the tests compare against.
