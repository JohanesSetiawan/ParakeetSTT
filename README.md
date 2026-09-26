# ParakeetSTT

A standalone PyTorch runtime for NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) speech-to-text model.

The model is reimplemented module by module in plain PyTorch. At runtime it does not use NeMo, Hugging Face Transformers, torchaudio, or torchcodec. You give it one audio file or a folder of files, of any length. It downloads and verifies the checkpoint on first use, splits long recordings into memory-bounded chunks, batches work across files, and writes a transcript per file with an explicit status.

Output parity with the reference implementation was verified: token-for-token and text-for-text against Hugging Face `ParakeetForTDT` on 15 test clips, and byte-identical long-form transcripts across every change to this runtime.

## Contents

- [What the model is](#what-the-model-is)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Output](#output)
- [Configuration](#configuration)
- [Supported input](#supported-input)
- [How long audio is processed](#how-long-audio-is-processed)
- [Measured performance](#measured-performance)
- [Checkpoint preparation](#checkpoint-preparation)
- [Logs](#logs)
- [Running the tests](#running-the-tests)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)
- [License and attribution](#license-and-attribution)

## What the model is

| Property | Value |
|---|---|
| Checkpoint | `nvidia/parakeet-tdt-0.6b-v3` |
| Parameters | about 627 million |
| Encoder | FastConformer, 24 layers, hidden size 1024, 8 attention heads, relative positional attention |
| Front end | 128-bin log-mel features at 16 kHz (25 ms window, 10 ms hop), 8x convolutional subsampling (one encoder frame = 80 ms) |
| Decoder | 2-layer LSTM prediction network, hidden size 640 |
| Output | Token-and-Duration Transducer (TDT): each step predicts a token and a duration of 0 to 4 encoder frames |
| Vocabulary | 8192 SentencePiece-style BPE pieces plus one blank token; output includes punctuation and capitalization |
| Languages | 25 European languages: bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk (per the model card) |
| Weights license | CC-BY-4.0 (NVIDIA) |

Decoding is greedy. Beam search and language-model rescoring are not implemented.

## Requirements

- **Python 3.11 or newer.** The code uses `tomllib` and `enum.StrEnum`. It was developed and tested with Python 3.13.
- **PyTorch.** Any build for your platform: CUDA, Apple MPS, or CPU. The device is chosen automatically. Development used PyTorch 2.14 with CUDA 13.2.
- **soundfile** (libsndfile) for decoding WAV, FLAC, OGG, MP3 and the other formats libsndfile supports. It installs NumPy with it.
- **FFmpeg (optional).** Only needed for formats libsndfile cannot read, such as M4A/AAC, Opus in MP4/WebM, or audio inside video files. `ffmpeg` and `ffprobe` are found through `PATH`, or through the `FFMPEG_BINARY` and `FFPROBE_BINARY` environment variables.
- **Disk:** about 5 GB for the checkpoint directory: the 2.5 GB `model.safetensors` download plus the 2.5 GB converted `model.pth`.
- **GPU memory:** the float32 weights take about 2.4 GB on the device. Transcription peaked at about 2.6 GB with the default settings on a 4 GB laptop GPU; see [Measured performance](#measured-performance).

## Installation

1. Clone the repository and create a virtual environment:

   ```powershell
   git clone https://github.com/JohanesSetiawan/ParakeetSTT.git
   cd ParakeetSTT
   python -m venv venv
   ```

2. Install PyTorch for your platform using the selector at [pytorch.org](https://pytorch.org/get-started/locally/). Keep the index URL it gives you and add `--no-cache-dir`.

3. Install the audio dependency:

   ```powershell
   venv\Scripts\python.exe -m pip install soundfile numpy --no-cache-dir
   ```

4. Optionally install FFmpeg and make sure `ffmpeg` and `ffprobe` are on `PATH`.

On Linux or macOS, use `venv/bin/python` instead of `venv\Scripts\python.exe`.

## Quick start

Transcribe one file. The transcript is printed to the terminal:

```powershell
venv\Scripts\python.exe inference.py --transcribe path\to\audio.mp3
```

Transcribe every audio file in a folder. The results are written to `transcriptions.csv` inside that folder:

```powershell
venv\Scripts\python.exe inference.py --transcribe path\to\folder
```

`--transcribe` is the only argument. Everything else is set in `config.toml`.

The first run downloads the checkpoint from Hugging Face, verifies every file against pinned hashes, converts it to `model.pth`, and writes a readiness marker. Later runs skip all of that.

A folder run prints plain-text progress. This example is from a two-file run on an RTX 3050 Ti Laptop GPU:

```text
Run id: 6e8d715c7ad0
Log file: ...\logs\log_2026-09-26.txt
Weights: ready
Device: cuda:0
Accelerator: CUDA
Device count: 1
Device name: NVIDIA GeForce RTX 3050 Ti Laptop GPU
Precision: float32
PyTorch: 2.14.0+cu132
CUDA runtime: 13.2
Python: 3.13.13
Model load seconds: 1.863
Transcribing 2 file(s)
Batch: 1 / 1, Progress: 100.00 percent, Elapsed: 0.6 s, ETA: 0.0 s
CSV: ...\transcriptions.csv
Files: 2
File statuses: ok=2
Total audio seconds: 17.220
Wall-clock seconds: 0.636
Media decode seconds: 0.002
Feature extraction seconds: 0.298
Model generation seconds: 0.332
Real-time factor: 0.036912
Throughput audio seconds per second: 27.091
Work items: 2, batches: 1
Peak accelerator memory allocated: 2511.6 MiB
```

## Output

### Single file

The terminal shows `Status:`, `Transcript:`, the audio duration, and the same timing summary as a folder run.

### Folder

`transcriptions.csv` is UTF-8 with this header:

```text
path_audio,filename_audio,duration_audio,status,transcription
```

- There is one row per file, sorted by path. With `recursive = true`, subfolders are included.
- `duration_audio` is in seconds and is empty for unreadable files.
- An existing `transcriptions.csv` in the folder is never treated as input.
- The CSV is written atomically: the new file replaces the old one only once it is complete.
- If the CSV cannot be written (for example, it is open in Excel on Windows), the rows are saved to `logs/transcriptions_<run id>.csv` instead. The command then reports this and exits with code 1.

### Status values

Every file gets exactly one status. When a file has several conditions, the most severe one wins, in the order of this table.

| Status | Meaning | What to do |
|---|---|---|
| `numerical_failure` | Features or encoder states were NaN/Inf for at least one chunk. The tokens from that chunk were discarded. | Report it; this should not happen with valid audio. |
| `input_nonfinite` | The decoded audio contained NaN/Inf samples. They were replaced with zeros before inference. | Check the source file. |
| `decoder_forced_advance` | The decoder emitted too many tokens on one frame and a safety guard forced it forward. Part of the transcript comes from a degenerate decoding loop. | Listen to the audio; the transcript may contain junk. |
| `no_speech` | Every chunk was digital silence (all-zero samples). | Nothing. |
| `empty_transcript` | The audio was not silent, but no token was produced (noise, music, or speech too short). | Check whether the file contains speech. |
| `unreadable` | Neither libsndfile nor FFmpeg could read the file. No transcript was attempted. | Install FFmpeg, or convert the file. |
| `ok` | A normal transcript. | |

The exit code is 0 when the run completes, including files with review statuses, which are summarized in a `Warning:` line. It is 1 when the run fails or the CSV had to go to the fallback location.

## Configuration

All settings live in `config.toml` at the repository root. Every key is required. Invalid values are rejected at startup, before the model loads, with the offending `section.key` in the error message. Relative paths are resolved against the repository root, not the current directory.

### `[paths]`

| Key | Default | Meaning |
|---|---|---|
| `weights_dir` | `"weights/parakeet-tdt-0.6b-v3"` | Checkpoint directory: downloads, `model.pth`, manifests, readiness marker. |
| `log_dir` | `"logs"` | Directory for dated run logs and fallback CSVs. |

### `[logging]`

| Key | Default | Meaning |
|---|---|---|
| `level` | `"INFO"` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. `DEBUG` adds a line per batch with its items, frame counts, and stage timings. |

### `[checkpoint]`

| Key | Default | Meaning |
|---|---|---|
| `request_timeout_seconds` | `120.0` | Timeout per HTTP request. Must be greater than 0. |
| `download_attempts` | `3` | Attempts per file for network failures. At least 1. |
| `retry_backoff_seconds` | `2.0` | The wait before attempt N is this value times (N - 1). |
| `stream_block_bytes` | `1048576` | Read size while streaming and hashing downloads. |

### `[inference]`

| Key | Default | Meaning |
|---|---|---|
| `batch_size` | `8` | Maximum chunks per micro-batch. |
| `recursive` | `true` | Include subfolders in folder mode. |
| `output_filename` | `"transcriptions.csv"` | CSV name inside the input folder. Must be a plain file name. |
| `audio_extensions` | `[]` | Optional suffix allow-list such as `[".wav", ".mp3"]`. Empty means every file is probed by content. |
| `max_chunk_feature_frames` | `1500` | Hard limit on one chunk's feature frames, including overlap. At a 10 ms hop that is 15 s of audio. |
| `overlap_feature_frames` | `50` | Context added on each side of a chunk (0.5 s). May be 0. `max_chunk_feature_frames` must be greater than twice this value. |
| `max_batch_feature_frames` | `3000` | Limit on a batch's padded size: rows times the longest row. Must be at least `max_chunk_feature_frames`. |
| `max_padding_fraction` | `0.25` | The largest share of a batch that may be padding, between 0 and 1. |
| `progress_interval_seconds` | `2.0` | Minimum time between two progress lines. The final line is always printed. |

`max_batch_feature_frames` and `max_chunk_feature_frames` are the memory controls. If a run stops with an out-of-memory error, lower them. The error message names both keys.

## Supported input

libsndfile handles WAV (PCM and float), FLAC, OGG/Vorbis, MP3 and the other formats it supports. It is always tried first. Any other file is probed with `ffprobe` and decoded with `ffmpeg`.

- **Channels:** multi-channel audio is averaged to mono.
- **Sample rate:** any rate is resampled to 16 kHz.
- **Non-finite samples:** NaN/Inf samples are replaced with zero and reported as `input_nonfinite`.
- **Missing FFmpeg:** if `ffprobe` is not installed, formats libsndfile cannot read are reported as `unreadable` and the rest of the folder is still processed.
- **Misconfigured FFmpeg path:** if `FFMPEG_BINARY` or `FFPROBE_BINARY` is set but points to a missing file, the run stops. That is treated as a setup error, not a property of one file.

## How long audio is processed

There is no duration limit, and memory use does not grow with the length of the recording.

1. **Planning.** Each file is cut into chunks. A chunk has a core that it owns, plus up to `overlap_feature_frames` of context on each side. The planner sizes chunks by the exact STFT frame count, `samples // 160 + 1`, so no chunk exceeds `max_chunk_feature_frames`.
2. **Scheduling.** Chunks from all files are interleaved round-robin, so one long file cannot delay every short file behind it. They are then packed into micro-batches bounded by `batch_size`, by the padded size `max_batch_feature_frames`, and by `max_padding_fraction`.
3. **Decoding.** Each file keeps one decoder handle open from its first chunk to its last. The overlap samples a chunk shares with the previous one are reused rather than decoded twice. Only the current chunk and one overlap tail per in-progress file are held in memory.
4. **Inference.** Each batch goes through feature extraction, the encoder, and greedy TDT decoding on the selected device.
5. **Merging.** A token is kept only by the chunk whose core contains the token's midpoint in time, so every position is transcribed by exactly one chunk. Repeated words are kept as spoken; nothing is deduplicated or corrected against a word list.

If the accelerator runs out of memory, the run stops with a diagnostic. It does not retry, shrink the batch, fall back to CPU, or write a partial CSV.

## Measured performance

These numbers come from one machine: an NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4 GB, PyTorch 2.14 + CUDA 13.2, Windows 11, default settings, and float32. They show what this hardware achieved, not a guarantee for other hardware. A laptop GPU's clock also varies with temperature by several percent between runs.

| Workload | Audio | Wall-clock | Real-time factor | Peak VRAM allocated |
|---|---|---|---|---|
| One MP3, 48 kHz stereo | 36 min (2163 s) | 27 to 31 s | about 0.013 | 2591 MiB |
| One MP3, 48 kHz stereo | 74 min (4420 s) | 57 to 62 s | about 0.013 | 2591 MiB |
| Folder: 15 short WAV + both MP3s | 112 min | about 88 s | about 0.013 | 2591 MiB |

- **Where the time goes:** in the 36-minute run, about 85% of the time is greedy decoding, about 12% is media decoding, and about 2% is feature extraction.
- **Model load:** 2 to 4 seconds once the checkpoint is prepared and the file is in the OS cache.
- **Memory does not grow with length:** peak VRAM and RAM growth were the same for the 36- and 74-minute files.

CPU and Apple MPS execution is supported by the code but has not been benchmarked.

## Checkpoint preparation

This runs automatically on the first transcription. You can also run it on its own:

```powershell
venv\Scripts\python.exe -m src.commands.prepare_checkpoint
```

It prepares `weights_dir` as follows:

1. **Download.** It downloads `config.json`, `generation_config.json`, `processor_config.json`, `tokenizer.json`, `tokenizer_config.json`, and `model.safetensors` from Hugging Face. Each file streams to a temporary file and replaces the target only after it passes verification.
2. **Verify.** Each file is checked against a size and hash pinned in `src/checkpoint/download.py`: SHA-256 for the weights, Git blob SHA-1 for the JSON files. A file that doesn't match is rejected. Files that already match are reused, and only missing or corrupted files are downloaded again.
3. **Convert.** It converts `model.safetensors` into `model.pth`, reading tensor by tensor with bounded memory. It checks that all required keys are present, that no weight is NaN/Inf, and that the vocabulary and output dimensions match.
4. **Record.** It writes `download_manifest.json` and `conversion_manifest.json`, so unchanged artifacts are never converted twice.
5. **Mark ready.** The inference command strict-loads the model. Only when that succeeds does it write `.ready`, which records the size of `model.pth`.

On every later run the fast path checks only that `.ready` exists and that `model.pth` still has the recorded size. A deleted, truncated, or replaced checkpoint triggers a full preparation again. To force one, delete `weights_dir/.ready`.

`model.pth` is loaded with `torch.load(..., weights_only=True)`, which refuses arbitrary pickled objects.

## Logs

Every run appends to `logs/log_<YYYY-MM-DD>.txt`. Each line carries the run id printed at startup:

```text
<timestamp> <LEVEL> run=<run id> <module>: <message>
```

A run logs:

- the input and mode
- the checkpoint action
- the device report
- the model load time
- the plan (files, work items, batches)
- one line per file (status, duration, chunks, word count)
- unreadable files
- full tracebacks on failure

Transcripts themselves are not written to the log.

## Running the tests

The tests use the standard library `unittest`:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Most tests build a tiny schema-valid checkpoint (`tests/support.py`) and need neither the real weights nor a GPU. `tests/test_smoke_real_weights.py` runs the real command on two sample WAVs. It is skipped automatically when the prepared checkpoint or the local sample audio (`docs/*.wav`, not committed) is missing.

## Project layout

```text
inference.py                  Command launcher (python inference.py --transcribe ...)
config.toml                   All runtime settings
src/
  audio/                      Media probing and decoding (media.py), log-mel features (features.py)
  checkpoint/                 Download and verification, safetensors -> model.pth, readiness marker
  commands/                   CLI entry points and terminal progress reporting
  configuration/              Checkpoint JSON validation (config.py), config.toml settings (settings.py)
  inference/                  Chunk planning (planning.py), offline orchestration and merging (offline.py)
  models/                     Subsampling, attention, Conformer blocks, encoder, LSTM decoder, joint, TDT loop
  runtime/                    Device selection and report, dated logging, atomic file writes
  text/                       Tokenizer decoding from tokenizer.json
tests/                        Unit, integration, and smoke tests
```

[AGENTS.md](AGENTS.md) documents the architecture, invariants, and development rules in depth.

## Known limitations

- **Resampling** is linear interpolation without a low-pass filter. Audio above 16 kHz sample rate with strong content above 8 kHz is aliased slightly, and the FFmpeg path resamples differently from the libsndfile path.
- **The FFmpeg fallback** starts one `ffmpeg` process per chunk and decodes overlap regions twice. libsndfile formats are not affected.
- **The greedy decoding loop** synchronizes with the GPU once per step. It is the dominant cost of a run.
- **Precision:** only float32 inference is implemented.
- **Scope:** the runtime does offline transcription only. There is no streaming, speaker diarization, word-level timestamps in the output, beam search, or language-model rescoring.
- **Language:** the model has no language selection. It transcribes whatever it recognizes among its 25 languages.

## License and attribution

The model weights are published by NVIDIA as `nvidia/parakeet-tdt-0.6b-v3` under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Using or redistributing them requires attribution to NVIDIA. This repository does not include the weights; they are downloaded from Hugging Face on first use.

This repository does not yet include a license file for its source code.
