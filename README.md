# Parakeet TDT Native PyTorch Runtime

## 1. Overview

This repository contains a standalone native PyTorch implementation and runtime
for the local `nvidia/parakeet-tdt-0.6b-v3` checkpoint.

The repository is focused exclusively on the Parakeet checkpoint and its native
runtime:

- JSON model configuration;
- checkpoint artifact acquisition;
- safetensors-to-PyTorch conversion;
- native Fast Conformer implementation;
- native LSTM prediction network;
- native TDT joint network;
- native token-duration decoding;
- native tokenizer decoding;
- checkpoint loading and validation;
- offline inference service;
- inference timing and throughput metrics;
- regression testing.

The native runtime uses PyTorch and Python standard-library code. It does not
import the Hugging Face model runtime during native inference.

## 2. Current Scope

### Included

- Parakeet TDT JSON configuration loading and validation.
- Native Fast Conformer encoder reconstruction.
- Native relative-position attention through PyTorch SDPA.
- Native convolutional subsampling.
- Native feed-forward and convolutional Conformer blocks.
- Native LSTM prediction network and decoder cache.
- Native TDT token-duration joint projection.
- Native greedy token-duration decoding.
- Native JSON BPE vocabulary loading and Metaspace detokenization.
- Strict `model.pth` state-dictionary loading.
- Per-file checkpoint download and identity validation.
- Atomic safetensors-to-PyTorch conversion.
- First-run readiness marker.
- Single-file and folder inference command.
- Runtime duration, latency, RTF, and throughput metrics.

### Explicitly excluded

- Changes to full model weights.
- Quantization.
- Pruning.
- Knowledge distillation.
- Training.
- Fine-tuning.
- Additional language adaptation.
- Realtime streaming.
- Long-form chunking and timestamp-aware boundary reconciliation.
- Scraping or external data collection.
- Claims of language accuracy without trusted references.
- Claims of unverified large-scale capacity.

The current model remains full precision and full weights. Runtime changes must
not silently replace or alter checkpoint parameters.

## 3. Model and Checkpoint

The checkpoint directory is:

```text
weights/parakeet-tdt-0.6b-v3/
```

Source artifacts:

```text
config.json
generation_config.json
model.safetensors
processor_config.json
tokenizer.json
tokenizer_config.json
```

Converted runtime bundle:

```text
model.pth
```

`model.pth` contains:

```python
{
    "state_dict": ...,
    "config": ...,
    "metadata": ...,
}
```

Verified model properties:

- Architecture: `ParakeetForTDT`.
- Model type: `parakeet_tdt`.
- Encoder layers: 24.
- Encoder hidden size: 1024.
- Encoder intermediate size: 4096.
- Attention heads: 8.
- Decoder hidden size: 640.
- Decoder layers: 2.
- Vocabulary size: 8193.
- Blank token ID: 8192.
- Padding token ID: 2.
- TDT durations: `[0, 1, 2, 3, 4]`.
- Mel features: 128.
- Subsampling factor: 8.
- State-dictionary keys: 723.
- Native model parameters: 627,008,134.

These values are loaded from JSON and validated. They must not be independently
redefined in model code.

## 4. Repository Structure

```text
AGENTS.md                       Agent operating contract
README.md                       User-facing Parakeet documentation
config.toml                     User-facing inference settings
inference.py                    Root inference launcher

src/
  config.py                     JSON configuration and validation
  audio.py                      Input container parsing boundary
  bootstrap.py                  First-run readiness marker
  checkpoint.py                 Acquisition and conversion orchestration
  conversion.py                 Safetensors parsing and PTH serialization
  inference.py                  Transcriber application service
  processing.py                 Native tensor preprocessing
  settings.py                   TOML settings loader
  tokenization.py               JSON BPE and Metaspace decoder

  model/
    attention.py                Relative-position SDPA attention
    conformer.py                Feed-forward and convolution blocks
    decoder.py                  LSTM prediction network and cache
    encoder.py                  Encoder stack orchestration
    joint.py                    TDT token-duration projection
    parakeet.py                 Model composition and strict loading
    subsampling.py              Conv2d subsampling

  utils/
    artifacts.py                Bounded-memory hashing
    download.py                 Artifact download and manifest handling
    runtime.py                  Runtime diagnostics and import checks

  cli/
    inference.py                User-facing inference command
    prepare_checkpoint.py       Checkpoint preparation command

tests/                          Standard-library regression tests
weights/                        Ignored local checkpoint artifacts
```

## 5. Python Environment

Use the existing repository environment:

```text
venv/
```

Run commands through:

```powershell
venv\Scripts\python.exe command.py
```

Rules:

- Do not recreate, replace, delete, or switch the environment.
- Do not modify the installed PyTorch package.
- Do not upgrade, downgrade, uninstall, or reinstall PyTorch.
- Do not install dependencies unless a concrete need has been established.
- Every authorized pip command must use `--no-cache-dir`.
- Do not add dependency versions or constraints without authorization.
- Prefer the standard library and packages already present in `venv`.

## 6. Native Dependency Boundary

The native runtime uses:

- Python standard library.
- PyTorch.

Native implementation modules must not directly import:

- Transformers.
- Hugging Face Hub.
- Safetensors.
- Tokenizers.
- Librosa.
- NumPy.
- SoundFile.

Reference and inspection tools may use installed external packages, but those
packages must not leak into the native model runtime.

## 7. Configuration

Checkpoint JSON artifacts are the model configuration source of truth:

- `config.json`: architecture, dimensions, vocabulary, token IDs, durations.
- `processor_config.json`: processing parameters and feature dimensions.
- `generation_config.json`: decoder and generation IDs.
- `tokenizer.json`: BPE vocabulary, merges, added tokens, and Metaspace rules.
- `tokenizer_config.json`: tokenizer metadata.

User-facing command behavior is configured in `config.toml`:

```toml
[inference]
batch_size = 8
recursive = true
output_filename = "transcriptions.csv"
audio_extensions = [".wav"]
```

Runtime code must load configuration rather than duplicate model-specific values.
Configuration is validated before model allocation.

## 8. Checkpoint Download and Conversion

Run:

```powershell
venv\Scripts\python.exe -m src.cli.prepare_checkpoint
```

The preparation command validates each source artifact independently using
official size and Git blob/LFS content identity.

Behavior:

- Matching files are reused.
- Missing files are downloaded individually.
- Only mismatching files are redownloaded.
- Downloads use same-directory `.part` files.
- Temporary files are validated before atomic replacement.
- `download_manifest.json` records per-file evidence.
- Conversion waits until all six source artifacts are valid.
- `conversion_manifest.json` binds `model.pth` to source and output hashes.
- Existing valid `model.pth` is reused or bootstrapped.
- Missing, stale, or invalid `model.pth` is regenerated atomically.

## 9. Easy Inference

Single-file command:

```powershell
venv\Scripts\python.exe inference.py --transcribe input_file
```

The command prints the decoded transcript and measured runtime metrics.

Folder command:

```powershell
venv\Scripts\python.exe inference.py --transcribe input_folder
```

The command discovers configured files, processes them in configured batches,
and writes:

```text
input_folder/transcriptions.csv
```

CSV columns:

```text
filename,duration_audio,transcribe
```

First-run behavior:

1. Check the checkpoint-local `.ready` marker.
2. If absent, prepare and strict-load the checkpoint.
3. Atomically create `.ready` after success.
4. Continue with inference.

After `.ready` exists, normal runs check only marker existence. They do not
repeat remote metadata, hashing, download, or conversion validation. Use
`--repair` to explicitly repeat preparation.

## 10. Model Architecture and Data Flow

The native model data flow is:

```text
Input tensor
-> configured preprocessing
-> Conv2d subsampling
-> relative positional encoding
-> Fast Conformer encoder
-> encoder projection
-> LSTM prediction network/cache
-> TDT joint token and duration logits
-> duration-aware greedy decoding
-> BPE ID lookup
-> Metaspace detokenization
-> transcript
```

Tensor contracts:

- Features: `(B, T_features, 128)`.
- Encoder states: `(B, T_encoder, 1024)`.
- Projected encoder states: `(B, T_encoder, 640)`.
- Generated token IDs: `(B, U)`.
- Generated durations: `(B, U)`.

State-dict attribute names mirror the checkpoint and must not be casually renamed.

## 11. Testing

Run:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover:

- JSON configuration.
- Tokenizer decoding.
- Tensor preprocessing.
- Device selection.
- Model attention padding safety.
- Checkpoint download reuse.
- Isolated artifact redownload.
- Conversion manifest bootstrap.
- Synthetic conversion round-trip.
- Readiness marker fast path and repair.
- Inference metric formulas.

Never calculate WER or CER without trusted reference transcripts. Never claim
tests, downloads, conversions, or benchmarks that did not run.

## 12. Verified Baseline and Limitations

Verified local baseline:

- 723 safetensors tensors.
- 627,008,134 native model parameters.
- Strict `model.pth` loading success.
- 15/15 individual inference success.
- 15-file mixed-duration batch success.
- Exact transcript, token, and duration parity between individual and batch native inference.
- Observed batch RTF around `0.013`.
- Observed peak allocated CUDA memory around `3.89 GB`.

These are local measurements, not universal hardware guarantees. Long-form
processing, realtime streaming, and large-scale workload capacity are not
verified in this milestone.
