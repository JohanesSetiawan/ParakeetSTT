# AGENTS.md

Guidance for coding agents and contributors working on this repository. [README.md](README.md) explains how to use the runtime. This file explains how it is built, which properties must never break, and how to prove a change did not break them.

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Architecture](#2-architecture)
3. [Module map and dependency rules](#3-module-map-and-dependency-rules)
4. [Invariants that must hold](#4-invariants-that-must-hold)
5. [Configuration discipline](#5-configuration-discipline)
6. [Error handling, statuses, and logging](#6-error-handling-statuses-and-logging)
7. [Coding standards](#7-coding-standards)
8. [Testing strategy](#8-testing-strategy)
9. [Verification protocol for inference changes](#9-verification-protocol-for-inference-changes)
10. [Performance work](#10-performance-work)
11. [Dependencies and environment](#11-dependencies-and-environment)
12. [Git and change management](#12-git-and-change-management)
13. [What agents may and must not modify](#13-what-agents-may-and-must-not-modify)
14. [Troubleshooting](#14-troubleshooting)
15. [Decision records](#15-decision-records)
16. [Open optimization items](#16-open-optimization-items)

## 1. Purpose and scope

This repository is a standalone PyTorch inference runtime for `nvidia/parakeet-tdt-0.6b-v3`. At runtime it uses only PyTorch, the Python standard library, and soundfile (plus FFmpeg as an optional external fallback). It must:

- produce the same tokens as the reference implementation (Hugging Face `ParakeetForTDT`);
- transcribe audio of any length with memory that does not grow with duration;
- process folders of mixed short and long files with bounded, deterministic batching;
- report every anomaly explicitly instead of producing a normal-looking transcript;
- run on a 4 GB GPU with the full float32 weights.

This repository only runs inference with the published checkpoint. Changing or replacing the model weights is out of scope.

## 2. Architecture

### Data flow of one command

```text
inference.py --transcribe <file|folder>
  commands/inference.py
    configuration/settings.py   load and validate config.toml
    runtime/logging_setup.py    open logs/log_<date>.txt, assign run id
    discover_audio_files        content-probe inputs (audio/media.inspect_media)
    checkpoint/bootstrap.py     readiness gate, then a single strict model load
      checkpoint/orchestration.py  download + verify + convert when not ready
    runtime/device.py           resolved device and version report
    inference/offline.py        OfflineTranscriber.transcribe
      inference/planning.py       chunks, round-robin scheduling, micro-batches
      per batch:
        audio/media.py            MediaSession.read_sequential_segment (decode, mono, 16 kHz;
                                  libsndfile, or one persistent ffmpeg stream per file)
        audio/features.py         log-mel features + per-recording normalization
        models/parakeet.py        ParakeetTDT.generate (encoder + greedy TDT loop)
      inference/recovery.py       re-decode chunks with a long non-silent wordless stretch
      inference/merging.py        word-level seam merge, then tokenizer decode and status
    CSV (folder) or terminal (single file)

python -m src.commands.benchmark --input <file|folder>
  same load and OfflineTranscriber, warm-up + measured rounds,
  one JSON line appended to <metrics_dir>/benchmark_<date>.jsonl
```

### Model

| Stage | Module | Shape (B = batch, T = feature frames, T' = T/8) |
|---|---|---|
| Log-mel front end | `audio/features.py` | waveform `(B, samples)` to `(B, T, 128)` plus mask `(B, T)` |
| Subsampling | `models/subsampling.py` | Conv2d stack, 3 stride-2 stages: `(B, T, 128)` to `(B, T', 1024)` |
| Positional encoding | `models/attention.py` `RelativePositionalEncoding` | `(B, 2T'-1, 1024)` sinusoidal relative positions |
| Conformer blocks x24 | `models/conformer.py` `EncoderBlock` | half FFN, relative multi-head attention with bias_u/bias_v, depthwise conv module (GLU, BatchNorm, SiLU), half FFN, LayerNorm |
| Encoder projection | `ParakeetTDT.encoder_projector` | `(B, T', 1024)` to `(B, T', 640)` |
| Prediction network | `models/decoder.py` | embedding + 2-layer LSTM + projection, blank-aware cache |
| Joint | `models/joint.py` | ReLU(enc + dec), then linear to 8193 token logits + 5 duration logits |
| Greedy TDT loop | `ParakeetTDT.generate` | per step: argmax token, argmax duration, advance frame pointer |

Greedy TDT rules (`generate`):

- A blank token with duration 0 advances one frame. This is the reference rule.
- A non-blank token updates the decoder state. Its duration may be 0, in which case the decoder stays on the same frame.
- After `max_symbols_per_step` consecutive non-blank zero-duration tokens on one frame, the last one is forced to advance one frame. It is counted in `GenerationResult.forced_advances`. This is the NeMo-style guard; see D3 in [Decision records](#15-decision-records).
- Rows whose encoder output is non-finite are marked finished before decoding and emit no tokens.
- `maximum_steps = max_symbols_per_step * encoder_length` is a safety net that the guard makes unreachable.

## 3. Module map and dependency rules

| Package | Responsibility | May import |
|---|---|---|
| `configuration/config.py` | Load and validate checkpoint JSON (`config.json`, processor, generation) | standard library only |
| `configuration/settings.py` | Load and validate `config.toml` | standard library, `config.py` |
| `runtime/` | Device selection and report, dated run logging, atomic writes, peak process memory | standard library, torch |
| `text/` | Tokenizer decoding from `tokenizer.json` | standard library, torch |
| `models/` | Network modules and the TDT loop, strict checkpoint loading | torch, `configuration`, `runtime.device` |
| `audio/features.py` | Feature extraction | torch, `configuration` |
| `audio/resampling.py` | Windowed-sinc resampling on a global grid | torch |
| `audio/media.py` | Probe and decode media, sequential chunk reading, persistent FFmpeg stream | torch, soundfile, subprocess, threading, `audio/resampling.py`, `inference.planning` (for `AudioMetadata`) |
| `inference/planning.py` | Pure planning math: chunks, schedule, batches | standard library only |
| `inference/merging.py` | Word grouping and seam alignment | standard library only |
| `inference/recovery.py` | Gap detection and recovery windows | standard library only |
| `inference/offline.py` | Orchestration, merge, statuses | everything above except `commands` and `checkpoint` |
| `checkpoint/` | Download, verify, convert, readiness marker | standard library, torch, `configuration.settings`, `runtime` |
| `commands/` | CLI parsing, terminal output, wiring, the benchmark command | anything |

Rules:

- `models/` never imports `audio/`, `inference/`, `checkpoint/`, or `commands/`.
- `checkpoint/bootstrap.py` does not import the model. The caller passes a `loader`, so the checkpoint layer stays independent of the architecture.
- Terminal printing happens only in `commands/`. Library code logs through `logging.getLogger(__name__)` and never prints.
- Command modules use literal logger names (`"src.commands.inference"`), not `__name__`. Under `python -m`, `__name__` is `"__main__"`, which falls outside the `"src"` logger that owns the log file.
- New public names go into the package `__init__.py` and its `__all__`.

## 4. Invariants that must hold

A change that breaks any of these is a regression, whatever else it improves.

### Model and checkpoint

1. **State-dict names are fixed.** The top-level attributes `encoder`, `encoder_projector`, `decoder`, `joint` and every nested attribute name mirror checkpoint keys. Renaming one breaks strict loading. File and class names may change; attribute names may not.
2. **Loading is strict.** `load_state_dict(strict=True, assign=True)`. Missing or unexpected keys raise.
3. **Loading is safe.** `torch.load(..., weights_only=True)` everywhere. Never switch to `weights_only=False`.
4. **Meta-device construction.** `load_model` builds the model under `torch.device("meta")` to skip random initialization. Any new non-persistent buffer must be created with an explicit `device="cpu"` in its constructor, as `inv_freq` and `duration_values` are. `_require_materialized` fails loudly otherwise.
5. **Pinned artifacts.** URLs, sizes, SHA-256, and Git blob SHA-1 values in `checkpoint/download.py` identify the verified upstream revision. They are an integrity invariant, not configuration. Update them only together with a full parity re-verification.
6. **Unimplemented config values are rejected.** `validate_config` requires `hidden_act == "relu"` (joint), `encoder_config.hidden_act == "silu"`, and `scale_input == false`, because the modules hard-wire that math. Add support in the modules before relaxing a check.

### Numerics and parity

7. **Feature extraction matches the reference.** Constants: preemphasis 0.97, periodic-false Hann window of 400 samples, `n_fft` 512, hop 160, constant STFT padding, Slaney mel filters, `LOG_GUARD = 2**-24`, `NORMALIZATION_EPSILON = 1e-5`. Normalization is per recording with divisor n for the mean and n - 1 for the variance. Both divisors are clamped to at least 1; that changes nothing for n >= 2 and keeps inputs with 0 or 1 valid frames finite.
8. **Parity targets.** Same tokens and durations single-file vs mixed-batch. Same content tokens and text as HF `ParakeetForTDT`. Long-form transcripts byte-identical to the previous commit, unless a change intends to alter them, in which case say so and measure it.

### Planning, decoding, merging

9. **Chunk budget.** Every work item satisfies `feature_frames = source_samples // 160 + 1 <= max_chunk_feature_frames`. The core is at most `(max_chunk - 2 * overlap) * 160 - 1` samples, and the one-sample reserve prevents the 1500 vs 1501 off-by-one.
10. **Batch budget is padded.** `len(batch) * max(item.feature_frames) <= max_batch_feature_frames`. Do not revert to summing unpadded frames.
11. **Cores tile the file.** Cores are contiguous, non-overlapping, and cover `[0, total_samples)`. Sources are the core plus at most `overlap` on each side, clamped at 0 and at the end of the file.
12. **Sequential decoding.** Chunks of one file are read in order through `read_sequential_segment`. Only `previous_end - current_start` samples are reused, and never through `tensor[-k:]` with `k == 0`, which selects the whole tensor. A non-positive overlap decodes fresh. The result must match a fresh decode of the same interval: bit-identical at the native 16 kHz, and within 1e-6 after resampling (float32 summation order; a real grid mismatch shows up as differences of order 1e-2).
    - **FFmpeg path.** Formats libsndfile cannot read are decoded by one `ffmpeg` process per file (`_FfmpegStream`), which resamples the whole stream continuously. Chunks read in order must equal the same slices of one continuous decode. A request behind the stream position (collapse recovery) uses a one-off decode of that range. The process is closed with its session, and a non-zero exit raises instead of passing as a short read.
12a. **Resampling on a global grid** (`audio/resampling.py`). Windowed-sinc (Hann, 6 zero crossings, 0.99 roll-off) polyphase resampling. Every source block starts on a multiple of the reduced source period, so any target interval equals the same slice of the whole file resampled in one piece. Never resample a piece independently from its own start, because for ratios like 44.1 kHz to 16 kHz that shifts the grid at every seam.
13. **Session lifetime.** A decoder session opens at a file's first chunk and closes after its last. Never open all files up front.
14. **Word-level ownership** (`inference/merging.py`). Tokens are grouped into words (a word starts at a piece beginning with the SentencePiece marker), and a word is never split between chunks. In the overlap, the longest run of words both chunks agree on is found: same letters and digits, start times within `merge_tolerance_feature_frames`. Each agreed word is taken once, from the chunk whose core holds its start. Without agreement, whole words go to the chunk whose core holds their start. Neighboring chunks disagree on word times by up to about 0.5 s, so never go back to token-level or midpoint ownership.
    - **Collapse recovery** (`inference/recovery.py`, `OfflineTranscriber._recover_chunk`). A chunk whose core has a stretch longer than `untranscribed_gap_seconds` with no word start and RMS at least `gap_silence_rms` is re-decoded. The window start is shifted by each configured offset in turn. Offsets are at most the overlap, so the core is always covered and the budget kept. The first result without the gap wins. Otherwise the smallest gap is kept and the file is `untranscribed_gap`.
15. **No text rewriting.** No deduplication of repeated words, no dictionary, word list, phrase, or name correction.

### Resource policy

16. **Memory does not scale with duration.** Never hold a full-file waveform, feature tensor, encoder output, or logits for long audio.
17. **OOM policy.** On accelerator OOM, stop with a diagnostic. No retry, no silent batch reduction, no CPU fallback, no partial CSV.
18. **Device agnostic.** Device resolution happens once in `runtime/device.select_device()`. Components use the model's device. Do not assume CUDA.

## 5. Configuration discipline

- Every user-facing value lives in `config.toml` and is parsed in `configuration/settings.py`. That covers paths, log level, network policy, budgets, and reporting interval.
- Every key is required. Validation reports `section.key` and the rule it broke. There are no silent defaults.
- Cross-field rules enforced today:
  - `max_chunk_feature_frames > 2 * overlap_feature_frames`
  - `max_batch_feature_frames >= max_chunk_feature_frames`
  - `0 <= max_padding_fraction <= 1`
  - `request_timeout_seconds > 0`
  - `download_attempts >= 1`
  - `untranscribed_gap_seconds > 0`
  - every `recovery_start_offsets_feature_frames` entry is non-zero and at most `overlap_feature_frames` in size
  - `warmup_rounds >= 0`, `measured_rounds >= 1`
- Constants that belong to the checkpoint's input contract (feature constants, pinned hashes) stay in code with a comment explaining why they are not settings.
- Adding a setting means:
  1. add a field to the dataclass;
  2. add the parser and its validation;
  3. add the key to `config.toml` with a comment;
  4. add a test in `tests/unit/test_runtime_services.py`;
  5. add a row to the README configuration table.

## 6. Error handling, statuses, and logging

### File statuses

`inference/offline.py` `FileStatus`, most severe first:

| Status | Assigned when |
|---|---|
| `numerical_failure` | features or encoder states non-finite for any chunk; that chunk's tokens are dropped |
| `input_nonfinite` | decoded samples contained NaN/Inf (zeroed before inference) |
| `decoder_forced_advance` | the per-frame guard fired |
| `untranscribed_gap` | transcript not empty, but a chunk still has a long non-silent stretch without words after collapse recovery |
| `no_speech` | transcript empty and every chunk is digital silence |
| `empty_transcript` | transcript empty and audio not silent |
| `unreadable` | assigned by `commands/inference.py` for files discovery could not read |
| `ok` | none of the above |

Statuses are derived from objective facts only. Do not add heuristics that judge transcript quality. A repeated-token "gibberish" rule was removed because it flagged natural speech (D5).

### Error classes

| Situation | Behavior |
|---|---|
| One file is not decodable audio | `ValueError`; the file becomes an `unreadable` row and the run continues. |
| `ffprobe`/`ffmpeg` not installed | `CodecUnavailableError`; per-file `unreadable`. |
| `FFMPEG_BINARY`/`FFPROBE_BINARY` set to a missing path | `RuntimeError`; the run stops (setup error). |
| OS cannot open the file (missing, permission, too many open files) | `OSError` surfaces. It is never treated as an unsupported format. |
| Invalid settings or checkpoint JSON | `ValueError` before model allocation. |
| Accelerator OOM | `RuntimeError` naming the batch items, frames, memory, and the two keys to lower. |
| CSV write fails | Fallback to `logs/transcriptions_<run id>.csv`; exit code 1. |

Never swallow an exception without logging it. Never report success for work that did not happen.

### Logging

- `logs/log_<YYYY-MM-DD>.txt` in `paths.log_dir`, appended. Each line carries `run=<id>`.
- INFO: run start, input, checkpoint action, device report, load time, plan summary, one line per file (status, duration, chunks, words, `processing_seconds`, `real_time_factor`). Per-file processing time is attributed: the file's own decode time plus a feature-frame share of each batch it was in, plus collapse re-decodes.
- WARNING: non-`ok` files, unreadable files, stale readiness marker, rejected legacy checkpoint.
- DEBUG: one line per batch with its items, frames, and stage timings.
- Never log secrets or full transcripts.

## 7. Coding standards

- **Readability over brevity.**
  - One transformation per statement, with descriptive intermediate names.
  - Multi-line signatures and calls when they are dense.
  - Shape comments such as `# (B, T, H)` on tensor transitions that are not obvious. Shape comments must be true.
- **Comments explain why:** a constraint, a trade-off, or an external quirk. Never restate the code. No emojis or decorative characters in code, comments, or terminal output.
- **Docstrings:** on public functions and classes, and on non-trivial internals. Cover purpose, arguments, returns, shapes, raised errors, and side effects.
- **Hot loop cost:** in `ParakeetTDT.generate`, every elementwise op runs once per emitted symbol. Count kernel launches before adding ops there, and measure (see [Performance work](#10-performance-work)).
- **No dead code:** no unused imports, no placeholder logic. Remove what your change makes unused.
- **Terminal output:** plain text only, for example `Batch: 12 / 241, Progress: 4.98 percent`.
- **Surgical changes:** keep diffs focused. Do not reformat or refactor unrelated code in the same change.

## 8. Testing strategy

The suite uses pytest (`requirements-dev.txt`, configured by `pytest.ini`). Tiers are assigned by directory in `tests/conftest.py`, so a test cannot sit in the wrong tier:

```powershell
venv\Scripts\python.exe -m pytest                     # all tiers
venv\Scripts\python.exe -m pytest -m "not full"       # fast suite: no checkpoint, no GPU
venv\Scripts\python.exe -m pytest -m full             # real checkpoint and real speech
venv\Scripts\python.exe -m pytest -m regression       # only the fixed-defect guards
```

| Tier | Location | Needs | Contents |
|---|---|---|---|
| unit | `tests/unit/` | nothing | `test_planning.py` (frame estimate, tiling and batch properties), `test_model.py` (config validation, features, TDT loop, duration table, non-finite rows, strict loading, tokenizer, attention), `test_offline.py` (sequential decoding bit-identity, downmix, ffprobe metadata, merge, statuses), `test_merging.py` (word grouping, seam alignment, fallback ownership), `test_recovery.py` (gap detection, recovery windows, collapse recovery wiring), `test_resampling.py` (pass band, alias rejection, interval equals whole file), `test_ffmpeg_stream.py` (persistent stream equals one continuous decode, one process per file, reads behind the stream, process closed; skipped without FFmpeg), `test_benchmark.py` (peak process memory, summary statistics, warm-up and measured rounds appended as one JSON line), `test_checkpoint.py` (download reuse, retry, pin rejection, conversion, against a real local HTTP server), `test_bootstrap.py`, `test_cli.py`, `test_runtime_services.py` |
| regression | `tests/regression/test_fixed_defects.py` | nothing | one test per fixed defect; each docstring states the original failure and each section names the fixing commit |
| full | `tests/full/` | prepared checkpoint | `test_accuracy.py` (exact transcripts, corpus WER budget, batching parity, determinism), `test_long_form.py` (chunking at scale, WER budget, CUDA memory bound, optional user recordings), `test_anomalies.py` (silence, noise, NaN samples, 48 kHz stereo), `test_command.py` (the real `inference.py` in a subprocess, folder CSV, FFmpeg fallback, error exit), `test_reference_parity.py` (HF `ParakeetForTDT` on CPU, skipped without `transformers`) |

Shared helpers live in `tests/support.py`:
- the tiny checkpoint writer;
- the WAV writer;
- settings factories;
- `ScriptedModel`, a stand-in for the model's `generate`;
- the speech-clip loader, which checks each file's SHA-256 against `manifest.json`;
- a dependency-free WER.

Fixtures live in `tests/conftest.py`: `tiny_checkpoint_dir`, `tiny_configuration`, and the session-scoped `real_settings`, `real_model`, `speech_clips`.

### Real speech data

`tests/data/librispeech/` holds five LibriSpeech dev-clean clips (CC BY 4.0, attribution in `SOURCE.md`) and `manifest.json`. For each clip the manifest records:

- the SHA-256;
- the LibriSpeech reference text;
- `expected_transcript`, this runtime's output, recorded and cross-checked against Hugging Face;
- `full_context_transcript`, Hugging Face on the unchunked clip.

If a deliberate change alters a transcript, update `expected_transcript` in the same commit and state why in the commit body. Never regenerate expectations to make an unexplained diff disappear.

### Known-defect tests

pytest runs with `xfail_strict = true`. When a known defect is tracked by an `xfail` test, fixing the defect makes the test pass unexpectedly and fail the run. Remove the marker in the same change that fixes the defect. There are no known-defect tests at the moment. The chunk-boundary tests were converted to budgets when word-level merging landed.

### Thresholds

The WER budgets are measured values plus a small margin, with the measurement written next to each constant. WER normalization spells out titles ("Mr" becomes "mister"), because LibriSpeech references spell them out.

- corpus (clip by clip) 3.67% measured, budget 4.6%;
- chunked clip against the unchunked reference 1.49% measured, budget 3%;
- long form (196 s, 15 chunks) 4.59% measured, budget 5.5%;
- cost of chunking over clip by clip: at most 1.5 points.

Tighten a budget when accuracy improves. Never loosen one without a written reason.

### Rules

- Every bug fix gets a regression test in `tests/regression/` that fails on the old code. Name the defect and the fixing commit.
- Prefer the tiny fixture checkpoint so tests run without the 2.5 GB weights or a GPU. Use `full` only for what needs real weights or real speech.
- Prefer randomized or property-style checks for planner math, with fixed seeds.
- The `full` tier never downloads the checkpoint. It skips with an explanation when `.ready` is missing.
- Update existing tests when behavior changes on purpose. Do not delete or weaken them to make a change pass.
- Do not claim tests pass unless they were run in this environment.

## 9. Verification protocol for inference changes

Unit tests cannot prove numerical parity with the full model. Any change under `audio/`, `models/`, `inference/`, or `text/` must pass `pytest -m full`, which automates checks 1 and 2 (and part of 3 and 6) on the committed clips. The remaining checks need longer recordings and instrumentation. Run them with the real checkpoint on the development GPU before merge.

1. **Token parity, single vs mixed.** Transcribe each sample WAV alone, then all together as one folder. Compare token IDs and durations after stripping trailing padding (`pad_token_id` with duration 0), because rows in a batch are right-padded to the longest row. They must be identical.
2. **Parity with the reference.** For each sample WAV, run Hugging Face `transformers` `ParakeetForTDT.from_pretrained(weights_dir)` with its `AutoProcessor` and `model.generate(..., return_dict_in_generate=True)`. Compare content tokens (blank and pad removed) and decoded text with this runtime. `transformers` is only a verification tool here; it must never become a runtime import.
3. **Long-form regression.** Transcribe long recordings (tens of minutes) with the previous commit and the new one. Transcripts must be byte-identical unless the change intends otherwise.
4. **Budget and session instrumentation.** Wrap the feature extractor and `MediaSession` for a mixed run. Check that:
   - the tensor frames per batch equal the planned maximum;
   - no chunk is over `max_chunk_feature_frames`;
   - no padded batch is over `max_batch_feature_frames`;
   - there is one session per file;
   - decoded source frames equal container frames (no re-decoded overlap).
5. **Memory bound.** Record peak CUDA allocation and working-set growth for two recordings of very different length. They must be the same.
6. **Anomaly behavior.** Check that:
   - silence gives `no_speech`;
   - noise gives `empty_transcript`;
   - NaN samples give `input_nonfinite`;
   - a forced stuck joint gives `decoder_forced_advance` without raising;
   - a simulated OOM stops once with no CSV;
   - an M4A through FFmpeg gives the same text as its WAV source.

Keep verification scripts outside the repository (a scratch directory). Report the actual numbers in the pull request.

## 10. Performance work

- **Measure first.** The dominant cost is the greedy TDT loop (about 85% of wall time on long audio). Media decoding is about 12% for MP3, and features about 2%.
- **Where the loop time goes** (RTX 3050 Ti Laptop, one batch of two 15 s chunks, float32): about 291 ms in total, of which the encoder takes 142 ms and the decoding loop 149 ms over 108 steps. The loop issues thousands of small kernel launches, so it is bound by launch overhead, not arithmetic. cuDNN's LSTM step costs about 0.30 ms; a hand-written fused LSTM cell was slower (0.98 ms) and was dropped.
- **Batch size and speed:** per chunk, 1 row took 243 ms, 2 rows 136 ms, 3 rows 111 ms, 4 rows 107 ms, 6 rows 126 ms, and 8 rows 131 ms. Four 15 s rows peaked at 2781 MiB allocated.
- **Benchmark command.** `python -m src.commands.benchmark --input <path>` loads once, warms up, runs the measured rounds, and appends one JSON record with the commit, settings, per-round stage timings, per-file metrics, memory peaks, and whether the transcripts stayed identical. Use it for A/B comparisons.
- **A/B protocol.**
  1. Check out the baseline in a separate `git worktree`.
  2. Point it at the same checkpoint (a directory junction or symlink; never copy or move the weights). Remove the junction with `rmdir` before removing the worktree, never by deleting through it.
  3. Run both versions on the same input for at least three rounds, alternating which one runs first.
  4. Compare wall and generation seconds (mean and spread) and the transcripts.

  Laptop GPUs drift several percent with temperature, so a difference smaller than the spread between rounds is noise.
- **Output must not change.** Every performance change must keep token parity. Faster but different output is a behavior change and must be justified and measured as one.
- **Windows:** `torch.compile` must not be used on Windows.
- **Memory is the binding constraint on 4 GB GPUs.** After loading, only about 0.7 GB may be free when other processes hold VRAM. Batch and chunk budgets are the controls.
- **Windows sysmem fallback.** Under WDDM the NVIDIA driver can back allocations that do not fit in VRAM with shared system memory instead of raising OOM. A budget that looks faster in a short test can therefore become several times slower on a busier GPU. Check the peak allocation against free VRAM, not just the timing.

## 11. Dependencies and environment

- Runtime dependencies: PyTorch, soundfile (brings NumPy), the standard library. FFmpeg is optional and external.
- Do not add torchaudio, torchcodec, librosa, NeMo, or Transformers as runtime dependencies. Do not hardcode an FFmpeg path; resolve it through `FFMPEG_BINARY`/`FFPROBE_BINARY` or `PATH`.
- Install with `pip ... --no-cache-dir`. Do not pin versions (`==`, `>=`, `<=`, `~=`) in install commands or dependency files.
- Never upgrade, downgrade, or reinstall PyTorch as a side effect of another install. If a package would change torch, stop and ask.
- Runtime dependencies are listed in `requirements.txt`, and test dependencies (pytest) in `requirements-dev.txt`. Both are unpinned. PyTorch is installed separately per platform.
- Use the repository's `venv` interpreter.

## 12. Git and change management

- **`main`** is the stable branch. It is protected by a ruleset: no deletion, no force push, changes only through pull requests.
- **Branch names** use a prefix that says what the work is:
  - `feature/` for new capability;
  - `fix/` for defects (several small related fixes may share one branch);
  - `perf/` for measured performance work;
  - `experiment/` for exploratory work that may never merge;
  - `research/` for research-specific pipelines.

  Avoid vague names such as `latest`, `new`, or `final`.
- **Commits** follow Conventional Commits (`feat`, `fix`, `perf`, `refactor`, `test`, `docs`). Each commit is one logical change with a body that explains why, what changed, and how it was verified. Do not write one-line commits for non-trivial changes.
- **Pull requests** contain: a summary, the motivation, the behavior changes, the tests run, and the verification numbers (Section 9). For performance work, add before/after timings. Also list known limitations.
- **Tags** mark milestones after merge to `main`, only when tests and verification passed.
- **Never commit** weights (`*.safetensors`, `*.pth`), `venv/`, `logs/`, `docs/` sample audio, datasets, credentials, or generated CSVs. `.gitignore` covers these; check `git status` before committing.

## 13. What agents may and must not modify

May, with tests:

- planning, scheduling, decoding orchestration, and merge logic, keeping every invariant in Section 4;
- performance of the TDT loop, feature extraction, and media decoding, with A/B evidence and parity;
- settings, adding validated keys;
- statuses, adding objective ones;
- documentation, kept consistent with the code.

Must not, without an explicit request from the maintainer:

- change the model architecture's math, activation choices, or state-dict attribute names;
- modify the downloaded weights or the pinned hashes;
- switch `torch.load` to `weights_only=False`;
- add text post-processing (dedup, dictionaries, corrections);
- add runtime dependencies listed as forbidden in Section 11, or pin versions;
- add CLI flags (the public contract is `--transcribe <path>` only; tunables go in `config.toml`);
- push to `main`, force-push, delete branches, or change repository settings;
- create or overwrite README.md or AGENTS.md unasked.

## 14. Troubleshooting

| Message (excerpt) | Cause | Fix |
|---|---|---|
| `Out of memory during model generation for batch items [...]` | Batch too large for free accelerator memory. | Lower `inference.max_batch_feature_frames`, then `inference.max_chunk_feature_frames`. Close other GPU processes. |
| `ffprobe was not found in PATH. Set FFPROBE_BINARY externally.` (in an `unreadable` reason) | FFmpeg not installed, and the file is not a libsndfile format. | Install FFmpeg or set `FFPROBE_BINARY`/`FFMPEG_BINARY`. |
| `FFPROBE_BINARY points to a missing executable` | The environment variable is set to a wrong path. | Fix or unset the variable. |
| `Input path does not exist` | Typo in `--transcribe`. | Check the path; nothing was loaded. |
| `No readable audio files found under ...` | The folder has no decodable audio, or the extension filter excludes everything. | Check `inference.audio_extensions` and the files. |
| `readiness marker is stale (model.pth missing or resized); re-preparing` (log) | `model.pth` changed since the marker was written. | None; preparation runs automatically. |
| `Downloaded content identity mismatch for <file>` | Download corrupted, or the upstream file changed. | Retry. If it persists, upstream changed and the pins need a deliberate update plus re-verification. |
| `Remote size changed for <file>` | Upstream artifact differs from the pinned size. | Same as above. |
| `Strict state-dict load reported incompatibility` | The checkpoint does not match the architecture. | Delete `weights_dir/.ready` and `model.pth`, then rerun to reconvert. |
| `Tensors were not materialized from the checkpoint` | A new non-persistent buffer was not created on CPU. | Create it with `device="cpu"` in the module constructor (Invariant 4). |
| `inference.<key> must be ...` | Invalid `config.toml` value. | Fix the named key. |
| `FFmpeg failed while decoding <file> (exit N): ...` (in the log) | ffmpeg could not decode part of the stream; the message ends with ffmpeg's own error lines. | Check the file with `ffmpeg -i`. |
| A run is several times slower than usual, with no error | On Windows, VRAM was overcommitted and the driver moved allocations to shared system memory. | Lower `inference.max_batch_feature_frames`, close other GPU processes, and compare `Peak accelerator memory allocated` with free VRAM. |
| `Error: the CSV could not be written ...; transcripts were saved to ...` | Output CSV locked (for example open in Excel) or folder not writable. | Close the file; the transcripts are at the printed fallback path. |

## 15. Decision records

- **D1. Native PyTorch instead of NeMo/Transformers at runtime.** Keeps the dependency set to PyTorch + soundfile, gives full control over batching and memory, and makes every numerical step inspectable. Parity with Transformers is verified instead of assumed.
- **D2. Meta-device construction + `weights_only=True`.** Random initialization of 600M parameters that are immediately overwritten cost about 2.7 s per load. Building on the meta device and assigning the loaded tensors cut model load from about 5.5 s to about 2 s. `weights_only=True` closes arbitrary-code execution through a replaced `model.pth`.
- **D3. Per-frame symbol guard.** Hugging Face's TDT generation has no per-frame limit and only truncates at `max_length`. The earlier runtime raised an exception there, which aborted a whole folder. The guard follows NeMo's greedy transducer behavior: bounded output, flagged by status. It never fires on normal speech (0 times across all verification audio), so parity is unaffected.
- **D4. Explicit status column.** The CSV shows anomalies per file instead of hiding them in logs. It is the only way to avoid "normal-looking but wrong" output in batch runs.
- **D5. No transcript-quality heuristics.** A repeated-adjacent-token rule marked natural speech and both long test recordings as gibberish. Statuses are limited to objective facts.
- **D6. Padded batch budget.** Memory scales with rows x longest row, so the budget counts that. Summing unpadded frames allowed about 33% overshoot at default settings.
- **D7. Lazy decoder sessions.** Opening every file up front could exhaust the process file limit (the Windows C runtime defaults to about 512). Sessions now live only while a file has pending chunks.
- **D8. Pinned upstream hashes in code.** The checkpoint identity is part of the verified contract, so it is not user configuration.
- **D9. Clamped normalization divisors.** Inputs with fewer than two valid frames produced NaN (division by n - 1 = 0) and hung decoding. Clamping keeps them finite and is bit-identical for n >= 2.
- **D10. Windowed-sinc resampling replaced linear interpolation.** Linear interpolation folded content above 8 kHz into the speech band. With 9 to 15 kHz hiss at 0.5, 1 and 2 times the speech RMS mixed into 48 kHz LibriSpeech, WER was 11.0%, 14.7% and 15.6% with linear, and 3.67% (the clean value) at every level with windowed sinc. Decode time on a 36-minute 48 kHz MP3 rose from about 3.3 s to 3.8 s. The kernel is the torchaudio `sinc_interp_hann` design, implemented in PyTorch because torchaudio is excluded.
- **D11. Word-level chunk merging.** Neighboring chunks timestamp the same word up to about 0.5 s apart. Token-midpoint ownership therefore split words across seams ("smile atile at one"), and WER over 15 chunks was 13.30% against 6.42% per clip. Grouping tokens into words and aligning the overlap on agreed words removes those artifacts. Chunk-by-chunk accuracy is now within one point of clip-by-clip.
- **D12. Collapse recovery by shifted windows.** On some windows the decoder emits only long blanks and skips seconds of speech, and the Hugging Face reference does the same on the same samples. Probing showed a start shift of 0.25 to 0.5 s restores the speech, but not one fixed shift for every case, so several offsets are tried in order. Cost on a 36-minute podcast: 6 of 155 chunks re-decoded, about 3 s extra, 16 words recovered. Doubling the overlap instead cost 7% more time and still left a gap, so the default overlap stays 0.5 s.

- **D13. One persistent FFmpeg process per file.** Spawning ffmpeg per chunk meant one process start, one seek, and one resampler restart per chunk, plus overlap decoded twice. On a 36-minute M4A (155 chunks) media decoding took 12.18 s; with one stream per file it takes 1.72 s, and wall time went from 41.4 s to 30.5 s.
- **D14. Fewer host synchronizations in the decoding loop.** The completion check (`finished.all()`) now runs every `FINISHED_CHECK_INTERVAL` (8) steps, and the extra steps, which only emit padding, are trimmed. The decoder's all-blank fast path was removed, because deciding it needed a host read every step. About 1% faster on the 36-minute recording, byte-identical transcripts.
- **D15. No micro-optimization of temporary tensors.** The loop is bound by kernel launches, not memory traffic. Removing a few temporaries saves a few launches out of dozens per step, within measurement noise. Capturing a step in a CUDA Graph removes the launch cost for every op at once, so that is the planned fix.

## 16. Open optimization items

These are known and deliberately not done yet. Each one needs the verification of Sections 9 and 10.

- **Decoding loop:** capture the per-step work in a CUDA Graph (the loop is launch-bound, see D15); consider TF32 for the encoder and custom kernels only where measurement shows a gain. `torch.compile` stays out (Section 10).
- **Budget tuning:** benchmark chunk sizes (1000 to 1750 frames), overlaps (0 to 100 frames), and batch budgets for speed and boundary accuracy. Four rows per batch was fastest per chunk (Section 10), which suggests `max_batch_feature_frames` around 6000, but check VRAM headroom and the Windows sysmem fallback first.
- **First-chunk collapse:** the first chunk of a file cannot shift its start, so a collapse there is reported, not recovered. Shrinking the window end instead is a possible extension.
- **Overlap tuning:** overlap 1.0 s gave lower WER on the LibriSpeech concatenation (5.28% against 7.57% before titles were normalized), but on a real podcast it cost 7% more time and left one untranscribed gap. Decide with a larger labeled long-form set.
- **Precision:** evaluate BF16/FP16 inference for memory headroom on 4 GB GPUs. Adopt only with measured parity or a documented, measured accuracy trade-off.
- **Media diagnostics:** an ffprobe capability report, classification of codec errors, explicit channel-layout handling, and clipping and silence metadata per file.
