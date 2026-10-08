"""Model math and loading on the tiny checkpoint: config, features, TDT loop, weights."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.audio.features import ParakeetFeatureExtractor, build_mel_filter_bank
from src.configuration.config import ParakeetConfig, load_config
from src.models.attention import Attention, EncoderMasks
from src.models.conformer import ConvolutionModule
from src.models.parakeet import ParakeetTDT, load_model
from src.text.tokenization import BpeTokenizer
from support import TINY_BLANK_ID, TINY_PAD_ID, TINY_VOCAB_SIZE, write_tiny_checkpoint

CPU = torch.device("cpu")


def build_tiny_model(directory: Path, durations: list[int] | None = None) -> ParakeetTDT:
    torch.manual_seed(0)
    return ParakeetTDT(load_config(write_tiny_checkpoint(directory, durations))).eval()


def features_for(model: ParakeetTDT, sample_counts: tuple[int, ...]):
    generator = torch.Generator().manual_seed(1)
    extractor = ParakeetFeatureExtractor(model.configuration)
    waveforms = [torch.randn(count, generator=generator) * 0.1 for count in sample_counts]
    return extractor(waveforms, [16000] * len(sample_counts), CPU)


def scripted_joint(token_id: int, duration_class: int, duration_count: int = 5):
    """Joint replacement that always picks one token and one duration class."""

    def forward(decoder_hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = decoder_hidden_states.shape[0]
        logits = torch.full((batch_size, 1, TINY_VOCAB_SIZE + duration_count), -1e4)
        logits[..., token_id] = 1e4
        logits[..., TINY_VOCAB_SIZE + duration_class] = 1e4
        return logits

    return forward


# =============================================================================
# Checkpoint configuration
# =============================================================================


def test_tiny_checkpoint_is_schema_valid(tiny_configuration: ParakeetConfig) -> None:
    assert tiny_configuration.blank_token_id == TINY_BLANK_ID
    assert tiny_configuration.durations == (0, 1, 2, 3, 4)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "hidden_act", "gelu"),
        ("encoder_config", "hidden_act", "relu"),
        ("encoder_config", "scale_input", True),
        (None, "model_type", "parakeet_ctc"),
        (None, "blank_token_id", TINY_VOCAB_SIZE),
        (None, "durations", [0, 1, 1]),
        ("encoder_config", "subsampling_factor", 6),
    ],
)
def test_incompatible_checkpoint_config_is_rejected(tmp_path: Path, section, key: str, value) -> None:
    directory = write_tiny_checkpoint(tmp_path)
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    (config[section] if section else config)[key] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(directory)


def test_missing_checkpoint_artifact_is_reported(tmp_path: Path) -> None:
    directory = write_tiny_checkpoint(tmp_path)
    (directory / "processor_config.json").unlink()

    with pytest.raises(FileNotFoundError, match="processor_config.json"):
        load_config(directory)


# =============================================================================
# Feature extraction
# =============================================================================


def test_normalized_features_have_zero_mean_and_unit_variance(tiny_configuration: ParakeetConfig) -> None:
    generator = torch.Generator().manual_seed(3)
    waveform = torch.randn(16000, generator=generator) * 0.1

    features, mask = ParakeetFeatureExtractor(tiny_configuration)([waveform], [16000], CPU)

    valid = features[0, mask[0]]
    assert torch.allclose(valid.mean(dim=0), torch.zeros(valid.shape[1]), atol=1e-4)
    assert torch.allclose(valid.std(dim=0, unbiased=True), torch.ones(valid.shape[1]), atol=1e-3)


def test_padding_rows_do_not_change_a_recording_features(tiny_configuration: ParakeetConfig) -> None:
    """A recording's features must not depend on what else is in the batch."""

    extractor = ParakeetFeatureExtractor(tiny_configuration)
    generator = torch.Generator().manual_seed(4)
    short = torch.randn(8000, generator=generator) * 0.1
    long = torch.randn(24000, generator=generator) * 0.1

    alone, alone_mask = extractor([short], [16000], CPU)
    batched, batched_mask = extractor([short, long], [16000, 16000], CPU)

    valid_frames = int(alone_mask[0].sum())
    assert int(batched_mask[0].sum()) == valid_frames
    assert torch.allclose(alone[0, :valid_frames], batched[0, :valid_frames], atol=1e-5)


def test_wrong_sample_rate_is_rejected(tiny_configuration: ParakeetConfig) -> None:
    with pytest.raises(ValueError, match="sample rate"):
        ParakeetFeatureExtractor(tiny_configuration)([torch.zeros(800)], [8000], CPU)


def test_slaney_filter_bank_shape_and_values() -> None:
    filters = build_mel_filter_bank(16000, 512, 128, 0.0, 8000.0)

    assert tuple(filters.shape) == (128, 257)
    assert torch.isfinite(filters).all()
    assert float(filters.sum()) > 0.0


# =============================================================================
# Greedy TDT decoding
# =============================================================================


def test_blank_with_duration_class_advances_through_duration_table(tmp_path: Path) -> None:
    """Duration class 1 of the table [0, 2, 4] advances two frames, not one."""

    model = build_tiny_model(tmp_path, durations=[0, 2, 4])
    features, mask = features_for(model, (16000,))

    with patch.object(model.joint, "forward", scripted_joint(TINY_BLANK_ID, 1, duration_count=3)):
        result = model.generate(features, mask)

    length = int(result.encoder_lengths[0])
    steps = (length + 1) // 2
    assert result.durations[0, 1 : 1 + steps].tolist() == [2] * steps
    assert result.frame_starts[0, 1 : 1 + steps].tolist() == list(range(0, 2 * steps, 2))
    assert torch.equal(result.frame_ends, result.frame_starts + result.durations)
    assert int(result.forced_advances[0]) == 0


def test_blank_with_zero_duration_still_advances_one_frame(tmp_path: Path) -> None:
    model = build_tiny_model(tmp_path)
    features, mask = features_for(model, (16000,))

    with patch.object(model.joint, "forward", scripted_joint(TINY_BLANK_ID, 0)):
        result = model.generate(features, mask)

    length = int(result.encoder_lengths[0])
    assert result.durations[0, 1 : 1 + length].tolist() == [1] * length


def test_non_finite_encoder_row_emits_nothing(tmp_path: Path) -> None:
    model = build_tiny_model(tmp_path)
    features, mask = features_for(model, (16000, 16000))
    original = model.encoder_projector.forward

    def poisoned(hidden_states: torch.Tensor) -> torch.Tensor:
        output = original(hidden_states).clone()
        output[1, 0, 0] = float("nan")
        return output

    with patch.object(model.encoder_projector, "forward", poisoned):
        result = model.generate(features, mask)

    assert result.encoder_finite.tolist() == [True, False]
    assert (result.sequences[1, 1:] == TINY_PAD_ID).all()


def test_finished_rows_are_right_padded(tmp_path: Path) -> None:
    model = build_tiny_model(tmp_path)
    features, mask = features_for(model, (4000, 16000))

    with patch.object(model.joint, "forward", scripted_joint(TINY_BLANK_ID, 1)):
        result = model.generate(features, mask)

    short_steps = int(result.encoder_lengths[0])
    assert (result.sequences[0, 1 + short_steps :] == TINY_PAD_ID).all()
    assert (result.durations[0, 1 + short_steps :] == 0).all()


# =============================================================================
# Checkpoint loading
# =============================================================================


def test_model_pth_round_trip_is_strict_and_fully_materialized(tmp_path: Path) -> None:
    source = build_tiny_model(tmp_path)
    torch.save(
        {"state_dict": source.state_dict(), "config": {}, "metadata": {"note": "tiny"}},
        tmp_path / "model.pth",
    )

    loaded, _configuration, metadata = load_model(tmp_path, device=CPU, fold_batch_norm=False)

    assert metadata == {"note": "tiny"}
    for name, tensor in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    for name, tensor in list(loaded.named_parameters()) + list(loaded.named_buffers()):
        assert not tensor.is_meta, name
    assert loaded.duration_values.tolist() == [0, 1, 2, 3, 4]
    assert torch.equal(loaded.encoder.encode_positions.inv_freq, source.encoder.encode_positions.inv_freq)
    assert not loaded.training


@pytest.mark.parametrize(("device", "mapped"), [(torch.device("cpu"), False), (torch.device("meta"), True)])
def test_checkpoint_is_mapped_only_for_accelerators_and_always_loaded_safely(
    tmp_path: Path,
    device: torch.device,
    mapped: bool,
) -> None:
    """
    For an accelerator, mmap keeps the 2.4 GB checkpoint out of private memory.
    On the CPU the loaded tensors are the model, so a mapping would hold the
    file open for the whole process (on Windows it could then not be replaced
    or repaired). weights_only blocks pickled code either way. The meta device
    stands in for an accelerator: anything that is not the CPU.
    """

    source = build_tiny_model(tmp_path)
    torch.save({"state_dict": source.state_dict(), "config": {}, "metadata": {}}, tmp_path / "model.pth")
    calls: list[dict] = []
    real_load = torch.load

    def spying_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    with patch.object(torch, "load", spying_load):
        load_model(tmp_path, device=device)

    assert len(calls) == 1
    assert calls[0]["mmap"] is mapped
    assert calls[0]["weights_only"] is True
    assert calls[0]["map_location"] == "cpu"


def test_missing_state_dict_key_is_rejected(tmp_path: Path) -> None:
    state_dict = build_tiny_model(tmp_path).state_dict()
    state_dict.pop("joint.head.weight")
    torch.save({"state_dict": state_dict, "config": {}, "metadata": {}}, tmp_path / "model.pth")

    with pytest.raises(RuntimeError, match="joint.head.weight"):
        load_model(tmp_path, device=CPU)


# =============================================================================
# Tokenizer and attention
# =============================================================================


def test_tokenizer_removes_special_ids_and_restores_spaces(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(
            {
                "model": {
                    "type": "BPE",
                    "vocab": {"<pad>": 2, "<blank>": 3, "▁hello": 4, "▁world": 5, "s": 6},
                    "merges": [],
                },
                "added_tokens": [
                    {"id": 2, "content": "<pad>", "special": True},
                    {"id": 3, "content": "<blank>", "special": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    tokenizer = BpeTokenizer(tokenizer_path, pad_token_id=2, blank_token_id=3)

    assert tokenizer.decode([2, 4, 5, 6, 3]) == "hello worlds"
    assert tokenizer.decode([]) == ""


def test_attention_masks_fully_padded_queries_without_nan() -> None:
    """Mixed-duration padding must not create all-negative-infinity softmax rows."""

    configuration = ParakeetConfig(
        model={
            "encoder_config": {
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "attention_bias": False,
            }
        },
        processor={},
        generation={},
        weights_dir=Path("."),
    )
    attention = Attention(configuration, layer_index=0).eval()
    valid_lengths = torch.tensor([2, 4, 0])
    valid_frames = torch.arange(4)[None, :] < valid_lengths[:, None]
    masks = EncoderMasks.from_valid_frames(valid_frames, torch.float32)

    # The third row is all padding: every key is blocked, which must still
    # give finite values (the bias is finite), and padded rows come out zero.
    output = attention(torch.randn(3, 4, 8), torch.randn(1, 7, 8), masks)

    assert tuple(output.shape) == (3, 4, 8)
    assert torch.isfinite(output).all()
    padded_rows = output.masked_select(masks.padded_frames)
    o_proj_bias = attention.o_proj.bias
    assert o_proj_bias is None and torch.all(padded_rows == 0)


def test_encoder_masks_bias_only_padded_keys() -> None:
    valid_frames = torch.tensor([[True, True, False], [True, True, True]])

    masks = EncoderMasks.from_valid_frames(valid_frames, torch.float16)

    assert tuple(masks.key_bias.shape) == (2, 1, 1, 3)
    assert tuple(masks.padded_frames.shape) == (2, 3, 1)
    blocked = torch.finfo(torch.float16).min / 2
    assert masks.key_bias[0, 0, 0].tolist() == [0.0, 0.0, blocked]
    assert masks.key_bias[1, 0, 0].tolist() == [0.0, 0.0, 0.0]
    # Adding any realistic score to the bias must not overflow to -inf.
    assert torch.isfinite(masks.key_bias + torch.tensor(-1000.0, dtype=torch.float16)).all()


def randomize_batch_norms(model: ParakeetTDT) -> None:
    """Give every convolution BatchNorm non-trivial statistics, as a trained checkpoint has."""

    generator = torch.Generator().manual_seed(7)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            channels = module.num_features
            module.running_mean.copy_(torch.randn(channels, generator=generator) * 0.3)
            module.running_var.copy_(torch.rand(channels, generator=generator) + 0.5)
            module.weight.data.copy_(torch.rand(channels, generator=generator) + 0.5)
            module.bias.data.copy_(torch.randn(channels, generator=generator) * 0.2)


def tiny_features(lengths: list[int], mel_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random features padded to the longest row, with the extractor's mask convention."""

    generator = torch.Generator().manual_seed(3)
    longest = max(lengths) + 1
    features = torch.randn(len(lengths), longest, mel_bins, generator=generator)
    mask = torch.arange(longest)[None, :] < torch.tensor(lengths)[:, None]
    return features * mask[:, :, None], mask


def test_folded_batch_norm_gives_the_same_encoder_output(tmp_path: Path) -> None:
    model = build_tiny_model(tmp_path).eval()
    randomize_batch_norms(model)
    mel_bins = model.configuration.encoder["num_mel_bins"]
    features, mask = tiny_features([40, 23], mel_bins)

    with torch.inference_mode():
        unfolded, _ = model.encoder(features, mask)
        model.fold_batch_norm()
        folded, _ = model.encoder(features, mask)
        model.fold_batch_norm()  # a second fold changes nothing
        folded_again, _ = model.encoder(features, mask)

    assert all(
        isinstance(module.norm, torch.nn.Identity)
        for module in model.encoder.modules()
        if isinstance(module, ConvolutionModule)
    )
    assert torch.allclose(folded, unfolded, atol=1e-5)
    assert torch.equal(folded_again, folded)


def test_padded_rows_match_running_each_row_alone(tmp_path: Path) -> None:
    """Shared masks: padding in a mixed batch must not change any real frame."""

    model = build_tiny_model(tmp_path).eval()
    randomize_batch_norms(model)
    model.fold_batch_norm()
    mel_bins = model.configuration.encoder["num_mel_bins"]
    lengths = [40, 23, 40]
    features, mask = tiny_features(lengths, mel_bins)

    with torch.inference_mode():
        batch_states, batch_mask = model.encoder(features, mask)
        for row, length in enumerate(lengths):
            row_features = features[row : row + 1, : length + 1]
            row_mask = mask[row : row + 1, : length + 1]
            alone, alone_mask = model.encoder(row_features, row_mask)
            valid = int(alone_mask.sum())
            assert int(batch_mask[row].sum()) == valid
            assert torch.allclose(batch_states[row, :valid], alone[0, :valid], atol=1e-5), row


@pytest.mark.parametrize(
    ("shortest", "padded", "expected"),
    [
        (1499, 1500, False),  # a full chunk: the extractor's padding frame vanishes at stride 2
        (1498, 1500, True),
        (41, 42, False),
        (40, 41, True),  # odd padded length: the padding frame survives the first stage
        (23, 41, True),
    ],
)
def test_subsampling_knows_when_masks_change_nothing(
    tmp_path: Path,
    shortest: int,
    padded: int,
    expected: bool,
) -> None:
    model = build_tiny_model(tmp_path)

    assert model.encoder.subsampling.leaves_padding(shortest, padded) is expected
