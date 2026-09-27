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
- [Benchmarking](#benchmarking)
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

3. Install the runtime dependencies (and, to run the tests, the development ones):

   ```powershell
   venv\Scripts\python.exe -m pip install -r requirements.txt --no-cache-dir
   venv\Scripts\python.exe -m pip install -r requirements-dev.txt --no-cache-dir
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

A folder run prints plain-text progress. This example is the five test clips in `tests/data/librispeech/` on an RTX 3050 Ti Laptop GPU:

```text
Run id: 07d4afc4f690
Log file: ...\logs\log_2026-09-27.txt
Weights: ready
Device: cuda:0
Accelerator: CUDA
Device count: 1
Device name: NVIDIA GeForce RTX 3050 Ti Laptop GPU
Precision: float32
PyTorch: 2.14.0+cu132
CUDA runtime: 13.2
Python: 3.13.13
Model load seconds: 3.481
Transcribing 5 file(s)
Batch: 1 / 6, Progress: 16.67 percent, Elapsed: 0.9 s, ETA: 4.7 s
Batch: 6 / 6, Progress: 100.00 percent, Elapsed: 1.8 s, ETA: 0.0 s
CSV: ...\transcriptions.csv
Files: 5
File statuses: ok=5
Total audio seconds: 46.630
Wall-clock seconds: 1.801
Media decode seconds: 0.018
Feature extraction seconds: 0.535
Model generation seconds: 1.239
Real-time factor: 0.038623
Throughput audio seconds per second: 25.892
Work items: 7, batches: 6
Peak accelerator memory allocated: 2496.2 MiB
Peak process memory: 3856.9 MiB
```

`Peak process memory` is the largest resident memory of the whole process so far (peak working set on Windows, peak RSS elsewhere). Most of it is the checkpoint, which is read into CPU memory before it moves to the GPU.

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
| `untranscribed_gap` | A stretch of more than `untranscribed_gap_seconds` of non-silent audio produced no words, even after re-decoding with shifted windows (see [How long audio is processed](#how-long-audio-is-processed)). The rest of the transcript is normal. | Listen to that stretch; it is often laughter, music, or noise, but it may be missed speech. The log gives its position. |
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
| `metrics_dir` | `"metrics"` | Directory for benchmark results (see [Benchmarking](#benchmarking)). |

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
| `merge_tolerance_feature_frames` | `100` | How far apart (1.0 s) two neighboring chunks may place the same word and still be recognized as one word at the seam. `0` disables seam alignment. |
| `untranscribed_gap_seconds` | `4.0` | A stretch of a chunk this long with no word, and not silent, is treated as a possible decoder collapse and re-decoded. |
| `gap_silence_rms` | `0.001` | RMS below which such a stretch counts as silence (about -60 dBFS) and is left alone. |
| `recovery_start_offsets_feature_frames` | `[-25, 25, -50, 50]` | Window-start shifts tried, in order, when re-decoding a collapsed chunk. Each must be at most `overlap_feature_frames` in size. An empty list disables recovery. |
| `progress_interval_seconds` | `2.0` | Minimum time between two progress lines. The final line is always printed. |

`max_batch_feature_frames` and `max_chunk_feature_frames` are the memory controls. If a run stops with an out-of-memory error, lower them. The error message names both keys.

On Windows, the NVIDIA driver can place allocations that do not fit in VRAM into shared system memory instead of raising an out-of-memory error. The run then continues, but several times slower. If a run is suddenly much slower than usual and `Peak accelerator memory allocated` is close to the card's size, lower the same two keys.

### `[benchmark]`

Used only by the development benchmark command, not by transcription.

| Key | Default | Meaning |
|---|---|---|
| `warmup_rounds` | `1` | Rounds run and discarded before measuring (CUDA kernel selection, file cache). At least 0. |
| `measured_rounds` | `3` | Rounds that are timed and compared. At least 1. |

## Supported input

libsndfile handles WAV (PCM and float), FLAC, OGG/Vorbis, MP3 and the other formats it supports. It is always tried first. Any other file is probed with `ffprobe` and decoded with `ffmpeg`.

- **Channels:** multi-channel audio is averaged to mono.
- **Sample rate:** any rate is resampled to 16 kHz with an anti-aliased windowed-sinc filter, so content above 8 kHz is removed instead of folding into the speech band. With 9 to 15 kHz hiss mixed into 48 kHz speech, word error rate stayed at 3.7% (the clean value); the old linear interpolation reached 15.6%.
- **Non-finite samples:** NaN/Inf samples are replaced with zero and reported as `input_nonfinite`.
- **FFmpeg decoding:** one `ffmpeg` process per file decodes the whole stream to 16 kHz mono in order, and each chunk continues exactly where the previous one stopped. Overlap regions are reused, not decoded again. A non-zero `ffmpeg` exit stops that file with the last lines of its error output.
- **Missing FFmpeg:** if `ffprobe` is not installed, formats libsndfile cannot read are reported as `unreadable` and the rest of the folder is still processed.
- **Misconfigured FFmpeg path:** if `FFMPEG_BINARY` or `FFPROBE_BINARY` is set but points to a missing file, the run stops. That is treated as a setup error, not a property of one file.

## How long audio is processed

There is no duration limit, and memory use does not grow with the length of the recording.

1. **Planning.** Each file is cut into chunks. A chunk has a core that it owns, plus up to `overlap_feature_frames` of context on each side. The planner sizes chunks by the exact STFT frame count, `samples // 160 + 1`, so no chunk exceeds `max_chunk_feature_frames`.
2. **Scheduling.** Chunks from all files are interleaved round-robin, so one long file cannot delay every short file behind it. They are then packed into micro-batches bounded by `batch_size`, by the padded size `max_batch_feature_frames`, and by `max_padding_fraction`.
3. **Decoding.** Each file keeps one decoder handle open from its first chunk to its last. The overlap samples a chunk shares with the previous one are reused rather than decoded twice. Only the current chunk and one overlap tail per in-progress file are held in memory.
4. **Inference.** Each batch goes through feature extraction, the encoder, and greedy TDT decoding on the selected device.
5. **Collapse recovery.** On rare windows the model skips seconds of clear speech and emits nothing. The Hugging Face reference does the same on the same samples, so it is a property of the model. Moving the window start by a few hundred milliseconds usually fixes it. A chunk with a long non-silent stretch and no words is re-decoded with the shifted windows from `recovery_start_offsets_feature_frames`, and the first result without the gap is used. If none works, the file gets the status `untranscribed_gap`.
6. **Merging.** Tokens are grouped into words and a word is never split between chunks. In the overlap, the words both chunks agree on are taken once, from the chunk with more context at that point. Neighboring chunks can place the same word up to about half a second apart, so agreement is checked within `merge_tolerance_feature_frames`. Without agreement, each whole word goes to the chunk whose core holds its start. Repeated words are kept as spoken; nothing is corrected against a word list.

If the accelerator runs out of memory, the run stops with a diagnostic. It does not retry, shrink the batch, fall back to CPU, or write a partial CSV.

## Measured performance

These numbers come from one machine: an NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4 GB, PyTorch 2.14 + CUDA 13.2, Windows 11, default settings, and float32. They show what this hardware achieved, not a guarantee for other hardware. A laptop GPU's clock also varies with temperature by several percent between runs.

| Workload | Audio | Wall-clock | Real-time factor | Peak VRAM allocated |
|---|---|---|---|---|
| One MP3, 48 kHz stereo | 36 min (2163 s) | 27 to 31 s | about 0.013 | 2591 MiB |
| One MP3, 48 kHz stereo | 74 min (4420 s) | 57 to 62 s | about 0.013 | 2591 MiB |
| Folder: 15 short WAV + both MP3s | 112 min | about 88 s | about 0.013 | 2591 MiB |
| The 36-minute recording as M4A (AAC, through FFmpeg) | 36 min (2163 s) | 30.5 s (one run) | about 0.014 | not recorded |

- **Where the time goes:** in the 36-minute MP3 run, about 85% of the time is greedy decoding, about 12% is media decoding, and about 2% is feature extraction. For the M4A, media decoding takes 1.7 s of the total.
- **Model load:** 2 to 4 seconds once the checkpoint is prepared and the file is in the OS cache.
- **Memory does not grow with length:** peak VRAM and RAM growth were the same for the 36- and 74-minute files.
- **Process memory:** about 3.9 GB peak, most of it the checkpoint staged on CPU during loading.

CPU and Apple MPS execution is supported by the code but has not been benchmarked.

## Benchmarking

For development, a separate command times the real pipeline repeatably:

```powershell
venv\Scripts\python.exe -m src.commands.benchmark --input path\to\audio_or_folder
```

It loads the model once and reports that cold start on its own. It then runs `[benchmark] warmup_rounds` transcriptions that are discarded, and `measured_rounds` that are timed. Nothing is written next to the audio. Each run appends one JSON line to `<metrics_dir>/benchmark_<YYYY-MM-DD>.jsonl` with:

- the git commit and whether the working tree had uncommitted changes;
- the device report and the `[inference]` and `[benchmark]` settings;
- the input (files, audio seconds, unreadable files);
- the cold start (weights action, bootstrap and load seconds);
- every measured round: wall, media decode, feature, generation, and recovery seconds, real-time factor, work items, batches, peak accelerator memory, and per-file status, chunks, recovered chunks, processing seconds, and real-time factor;
- mean, min, max, and standard deviation of wall and generation seconds;
- whether every round produced the same transcripts;
- the peak process memory.

To compare two commits, run both on the same input and compare the summaries. A laptop GPU's clock varies with temperature, so a difference smaller than the spread between rounds is noise.

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
- one line per file (status, duration, chunks, word count, processing seconds, real-time factor)
- unreadable files
- full tracebacks on failure

Transcripts themselves are not written to the log.

A file's processing seconds are its own media decoding plus a share of each batch's feature and generation time, split by feature frames, plus any collapse re-decodes. Batches mix files, so this is an attribution, not a separately timed run.

## Running the tests

The suite uses pytest (`requirements-dev.txt`) and has three tiers, selected by directory:

| Tier | Directory | What it needs | What it checks |
|---|---|---|---|
| `unit` | `tests/unit/` | nothing (tiny fixture checkpoint, CPU) | planner, features, TDT loop, media decoding, merge, statuses, settings, checkpoint download and conversion against a local HTTP server, CLI |
| `regression` | `tests/regression/` | nothing | one test per defect that was found and fixed, each naming the fixing commit |
| `full` | `tests/full/` | the prepared checkpoint | real speech end to end: exact transcripts, word error rate, batching and determinism, long-form memory bound, anomalies, the real `inference.py` command, and parity with Hugging Face `ParakeetForTDT` (when `transformers` is installed) |

```powershell
venv\Scripts\python.exe -m pytest                     # everything
venv\Scripts\python.exe -m pytest -m "not full"       # fast suite, no checkpoint needed
venv\Scripts\python.exe -m pytest -m full             # real checkpoint and speech only
```

The `full` tier uses five LibriSpeech clips committed under `tests/data/librispeech/` (CC BY 4.0; see `SOURCE.md` there). If the checkpoint has not been prepared, the whole tier is skipped with a message rather than downloading it. To also run your own long recordings through the long-form checks, list them in `PARAKEET_TEST_LONG_AUDIO`, separated by `;` on Windows or `:` elsewhere.

On the development machine the full suite takes about 70 seconds.

## Project layout

```text
inference.py                  Command launcher (python inference.py --transcribe ...)
config.toml                   All runtime settings
src/
  audio/                      Media probing and decoding (media.py), resampling (resampling.py), log-mel features (features.py)
  checkpoint/                 Download and verification, safetensors -> model.pth, readiness marker
  commands/                   CLI entry points (inference.py, prepare_checkpoint.py, benchmark.py) and progress reporting
  configuration/              Checkpoint JSON validation (config.py), config.toml settings (settings.py)
  inference/                  Chunk planning (planning.py), orchestration (offline.py), seam merging (merging.py), collapse recovery (recovery.py)
  models/                     Subsampling, attention, Conformer blocks, encoder, LSTM decoder, joint, TDT loop
  runtime/                    Device selection and report, dated logging, atomic file writes, process memory
  text/                       Tokenizer decoding from tokenizer.json
tests/
  unit/                       Fast isolated tests (tiny fixture checkpoint)
  regression/                 One test per fixed defect
  full/                       Real checkpoint and real speech, end to end
  data/librispeech/           Five CC BY 4.0 speech clips with expected transcripts
```

[AGENTS.md](AGENTS.md) documents the architecture, invariants, and development rules in depth.

## Known limitations

- **Chunking still costs a little accuracy.** On the test clips, word error rate is 3.7% when each clip is transcribed alone and 4.6% when the same speech is one 196-second recording cut into 15 chunks. Each chunk sees less context than the whole recording.
- **Decoder collapse cannot always be recovered.** The first chunk of a file has no earlier audio to shift into, and some windows stay collapsed at every tried shift. Such files are marked `untranscribed_gap` rather than passed off as complete.
- **The greedy decoding loop** is the dominant cost of a run. Each step launches dozens of small GPU kernels, so on a small GPU the loop is limited by launch overhead rather than arithmetic.
- **Precision:** only float32 inference is implemented.
- **Scope:** the runtime does offline transcription only. There is no streaming, speaker diarization, word-level timestamps in the output, beam search, or language-model rescoring.
- **Language:** the model has no language selection. It transcribes whatever it recognizes among its 25 languages.

## License and attribution

The source code in this repository is licensed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/), the same license NVIDIA uses for the model. The full legal text is in [LICENSE](LICENSE). Copyright (c) 2026 Johanes Setiawan.

The model weights are a separate work: NVIDIA publishes them as [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) under CC-BY-4.0. Using or redistributing the weights requires attribution to NVIDIA. This repository does not include the weights; they are downloaded from Hugging Face on first use.
