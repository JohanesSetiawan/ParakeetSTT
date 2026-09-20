# Agent Guide for Parakeet TDT Runtime

This document is the repository operating guide for AI agents. It describes only
the Parakeet model, native PyTorch implementation, checkpoint artifacts,
processing, inference, conversion, and tests. It does not describe unrelated
future systems.

A newer explicit user instruction takes precedence over this document. Do not
invent undocumented repository behavior.

## 1. Repository Purpose

This repository reconstructs the local `nvidia/parakeet-tdt-0.6b-v3` checkpoint
as a standalone native PyTorch runtime. The current objective is reliable local
Parakeet checkpoint handling, native model behavior, checkpoint conversion, and
simple offline inference.

The runtime preserves checkpoint behavior, reads JSON artifacts as the model
configuration source of truth, and does not import the Hugging Face model runtime
during native inference.

## 2. Current Scope

Included:

- Parakeet TDT JSON configuration loading and validation.
- Native Fast Conformer encoder.
- Native relative-position SDPA attention.
- Native convolutional subsampling.
- Native feed-forward and convolutional blocks.
- Native LSTM prediction network and decoder cache.
- Native TDT joint network.
- Native greedy token-duration decoding.
- Native JSON BPE and Metaspace decoding.
- Strict `model.pth` loading.
- Per-file checkpoint acquisition and identity validation.
- Atomic safetensors-to-PyTorch conversion.
- First-run readiness marker.
- Single-file and folder inference.
- Inference duration, latency, RTF, and throughput metrics.

Explicitly excluded:

- Changing full model weights.
- Quantization, pruning, or distillation.
- Training or fine-tuning.
- Additional language adaptation.
- Realtime streaming.
- Long-form chunking and timestamp reconciliation.
- External data collection or scraping.
- Accuracy claims without trusted references.

## 3. Verified Model Facts

- Checkpoint: `nvidia/parakeet-tdt-0.6b-v3`.
- Architecture: `ParakeetForTDT`.
- Encoder: 24 layers, hidden size 1024, intermediate size 4096, 8 heads.
- Decoder: hidden size 640, 2 LSTM layers.
- Vocabulary: 8193 entries.
- Blank ID: 8192.
- Padding ID: 2.
- Durations: `[0, 1, 2, 3, 4]`.
- Mel features: 128.
- Subsampling factor: 8.
- Native model parameters: 627,008,134.
- State-dictionary keys: 723.

## 4. Repository Structure

```text
src/
  config.py                     JSON configuration and validation
  audio.py                      Input container parsing
  bootstrap.py                  Readiness marker and repair flow
  checkpoint.py                 Acquisition and conversion orchestration
  conversion.py                 Safetensors parsing and PTH serialization
  inference.py                  Transcriber application service
  processing.py                 Tensor preprocessing
  settings.py                   TOML settings
  tokenization.py               BPE and Metaspace decoder

  model/
    attention.py                Relative-position SDPA attention
    conformer.py                Feed-forward and convolution blocks
    decoder.py                  LSTM prediction network and cache
    encoder.py                  Encoder stack orchestration
    joint.py                    TDT joint projection
    parakeet.py                 Model composition and strict loading
    subsampling.py              Conv2d subsampling

  utils/
    artifacts.py                Bounded-memory hashing
    download.py                 Artifact download and manifest handling
    runtime.py                  Runtime diagnostics

  cli/
    inference.py                Inference command
    prepare_checkpoint.py       Checkpoint preparation command

inference.py                    Root inference launcher
config.toml                     User-facing settings
tests/                          Regression tests
weights/                        Ignored checkpoint artifacts
```

## 5. Configuration Source of Truth

Checkpoint JSON artifacts define model and processing behavior:

- `config.json`: architecture, dimensions, vocabulary, token IDs, durations.
- `processor_config.json`: processing parameters and feature dimensions.
- `generation_config.json`: decoder and generation IDs.
- `tokenizer.json`: vocabulary, merges, added tokens, Metaspace rules.
- `tokenizer_config.json`: tokenizer metadata.

Do not duplicate these values in implementation code. Derived values may be
calculated from loaded JSON. Validate configuration before model allocation.

User-facing behavior comes from `config.toml`:

```toml
[inference]
batch_size = 8
recursive = true
output_filename = "transcriptions.csv"
audio_extensions = [".wav"]
```

## 6. Native Dependency Boundary

Native runtime code uses PyTorch and Python standard-library code. It must not
directly import Transformers, Hugging Face Hub, Safetensors, Tokenizers, Librosa,
NumPy, or SoundFile.

## 7. Architecture and Tensor Flow

```text
Input tensor
-> configured preprocessing
-> Conv2d subsampling
-> relative positional encoding
-> Fast Conformer encoder
-> encoder projection
-> LSTM prediction network/cache
-> TDT token and duration logits
-> duration-aware greedy decoding
-> JSON BPE Metaspace decoding
-> transcript
```

Important shapes:

- Features: `(B, T_features, 128)`.
- Encoder states: `(B, T_encoder, 1024)`.
- Projected states: `(B, T_encoder, 640)`.
- Generated IDs: `(B, U)`.
- Generated durations: `(B, U)`.

State-dict attribute names mirror the checkpoint and are compatibility boundaries.

## 8. Easy Inference Contract

Single file:

```powershell
venv\Scripts\python.exe inference.py --transcribe input_file
```

Folder:

```powershell
venv\Scripts\python.exe inference.py --transcribe input_folder
```

Folder inference writes `input_folder/transcriptions.csv` with:

```text
filename,duration_audio,transcribe
```

The first run prepares and strict-loads the checkpoint, then writes the
checkpoint-local `.ready` marker atomically. Later runs check only marker
existence and skip repeated download, hash, remote metadata, and conversion
validation. `--repair` explicitly removes readiness and repeats preparation.

## 9. Checkpoint Acquisition

Run:

```powershell
venv\Scripts\python.exe -m src.cli.prepare_checkpoint
```

Rules:

- Use the existing `venv`.
- Do not modify installed PyTorch.
- Every authorized pip command uses `--no-cache-dir`.
- Do not add dependency version pins without authorization.
- Validate each artifact by official size and Git blob/LFS SHA-256 identity.
- Reuse matching files and redownload only mismatching files.
- Use same-directory `.part` files and atomic replacement.
- Convert only after all six source artifacts are valid.
- Bind `model.pth` to source/output hashes with conversion manifest.

## 10. Mandatory Commented Code

Commented code is mandatory. Every meaningful module requires a module
Docstring. Public functions, public classes, complex components, and ML modules
require detailed docstrings.

Every non-trivial section must include technical comments covering algorithm,
invariant, tensor shape/layout, format, compatibility, performance, numerical
stability, or failure behavior not obvious from syntax.

ML comments are required around preprocessing, STFT, mel filters, normalization,
masking, attention, relative positions, convolution, LSTM cache, TDT durations,
device placement, checkpoint state, and numerical safety.

Comments must be accurate, synchronized with code, non-decorative, and must not
replace clear naming, types, validation, tests, or decomposition.

## 11. Testing and Quality Gate

Run:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover configuration, parsing, tokenization, processing, device selection,
padded attention safety, per-file acquisition, isolated redownload, conversion
bootstrap, synthetic conversion, readiness marker behavior, repair, and
inference metric formulas.

Before completion verify syntax, diagnostics, strict loading when model code
changes, finite individual/batch output, parity where expected, manifests, and
absence of secrets or stale documentation. Never claim tests or benchmarks that
did not run.

## 12. Verified Baseline and Limitations

Observed local baseline:

- 627,008,134 native model parameters.
- Strict `model.pth` loading success.
- 15/15 individual inference success.
- 15-file mixed-duration batch success.
- Exact transcript, token, and duration parity between individual and batch native inference.
- Observed batch RTF around `0.013`.
- Observed peak allocated CUDA memory around `3.89 GB`.

These are local observations, not universal guarantees. Long-form processing,
realtime streaming, and large-scale capacity are not verified in this milestone.
