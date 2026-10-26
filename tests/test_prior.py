# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Whole-cell completion, restricted checkpoints, and strict bit-cache contracts."""

import copy
import json
import os
import pickle

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from orbit_folding.checkpoints import CHECKPOINT_FORMAT, load_checkpoint, load_model, save_checkpoint
from orbit_folding.data import GeometryDataset
from orbit_folding.models.prior import CompletionPrior, masked_completion, spatial_mask
from orbit_folding.token_cache import TokenDataset, export_tokens, tokenizer_hash


class _RecordingPrior(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.num_bits = channels
        self.logits = nn.Parameter(torch.full((1, channels, 1, 1), -2.0))

    def forward(self, bits, observed):
        self.inputs = bits.detach().clone()
        self.observed = observed.detach().clone()
        return self.logits.expand_as(bits)


class _CacheTokenizer(nn.Module):
    """Test double for the cache's encode/metadata interface, not a model test."""

    def __init__(self):
        super().__init__()
        self.specs = {"Task": "tokenizer", "fixture": True}
        self.register_buffer("offset", torch.tensor(0, dtype=torch.int64))

    def encode(self, points, frame):
        assert frame.shape == (points.shape[0], 3, 3)
        pattern = (torch.arange(48, device=points.device).reshape(1, 4, 3, 4) + self.offset) % 2
        return {"bits": pattern.expand(points.shape[0], -1, -1, -1).float()}


class _UnsafePayload:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return os.mkdir, (self.marker,)


@pytest.fixture
def bits():
    return (torch.arange(3 * 4 * 3 * 5).reshape(3, 4, 3, 5) % 2).float()


@pytest.fixture
def token_cache(tmp_path, geometry_root):
    tokenizer = _CacheTokenizer()
    tokenizer.train()
    dataset = GeometryDataset(geometry_root, "all", pc_size=8, samples_per_mesh=9, uniform_samples=7)
    root = tmp_path / "cache"
    manifest = export_tokens(tokenizer, DataLoader(dataset, batch_size=2), root, "cpu")
    assert tokenizer.training
    return root, manifest, tokenizer


@pytest.mark.parametrize("ratio", [0.0, 0.001, 0.2, 0.5, 1.0])
def test_completion_masks_exact_whole_cells_and_preserves_observations(bits, ratio):
    model = _RecordingPrior()
    model.train()
    result = masked_completion(model, bits, ratio=ratio, seed=31)
    hidden = result["masked"]
    expected = max(int(ratio > 0), round(ratio * 15))
    assert hidden.dtype == torch.bool
    assert hidden.shape == (3, 1, 3, 5)
    assert torch.all(hidden.flatten(1).sum(1) == expected)
    expanded = hidden.expand_as(bits)
    torch.testing.assert_close(model.inputs[expanded], torch.full_like(bits[expanded], 0.5))
    torch.testing.assert_close(model.inputs[~expanded], bits[~expanded], atol=0, rtol=0)
    assert torch.equal(model.observed, ~hidden)
    for key in ("probabilities", "hard_bits"):
        assert result[key].shape == bits.shape
        torch.testing.assert_close(result[key][~expanded], bits[~expanded], atol=0, rtol=0)
    torch.testing.assert_close(result["probabilities"][expanded],
                               torch.full_like(bits[expanded], torch.sigmoid(torch.tensor(-2.)).item()))
    assert torch.all(result["hard_bits"][expanded] == 0)
    assert model.training


def test_completion_is_deterministic_without_changing_global_rng(bits):
    model = _RecordingPrior().eval()
    torch.manual_seed(77)
    state = torch.get_rng_state().clone()
    first = masked_completion(model, bits, 0.5, 14)
    assert torch.equal(state, torch.get_rng_state())
    torch.rand(101)
    second = masked_completion(model, bits, 0.5, 14)
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    assert not torch.equal(first["masked"], masked_completion(model, bits, 0.5, 15)["masked"])
    assert not model.training


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_completion_ratios_raise(bits, ratio):
    with pytest.raises(ValueError):
        masked_completion(_RecordingPrior(), bits, ratio, 0)


@pytest.mark.parametrize("shape", [(3, 4, 3, 5), (2, 4, 1, 1)])
def test_zero_training_mask_ratio_still_hides_one_full_cell(prior_config, shape):
    prior_config["model"].update(mask_ratio_min=0.0, mask_ratio_max=0.0, uncond_prob=0.0)
    model = CompletionPrior(prior_config)
    recorder = _RecordingPrior()
    model.model = recorder
    batch = {"bits": torch.ones(shape)}
    torch.manual_seed(6)
    loss, metrics = model(batch)
    hidden = ~recorder.observed
    assert hidden.shape == (shape[0], 1, *shape[2:])
    assert torch.all(hidden.flatten(1).sum(1) == 1)
    assert int(metrics["token_count"]) == shape[0]
    assert int(metrics["bit_count"]) == shape[0] * shape[1]
    assert torch.all(recorder.inputs[hidden.expand(shape)] == 0.5)
    torch.testing.assert_close(metrics["loss_sum"] / metrics["bit_count"], loss)
    loss.backward()
    assert torch.isfinite(recorder.logits.grad).all()


@pytest.mark.parametrize("ratio_min,ratio_max,uncond", [(0, 0, 0), (0.2, 0.8, 0), (1, 1, 0), (0, 0, 1)])
def test_training_masks_are_seeded_whole_cells(bits, ratio_min, ratio_max, uncond):
    torch.manual_seed(45)
    first = spatial_mask(bits, ratio_min, ratio_max, uncond)
    torch.manual_seed(45)
    second = spatial_mask(bits, ratio_min, ratio_max, uncond)
    assert first.shape == (3, 1, 3, 5)
    assert first.dtype == torch.bool
    assert first.flatten(1).any(1).all()
    assert torch.equal(first, second)
    if ratio_min == 1 or uncond == 1:
        assert first.all()


def test_bit_error_counts_binary_channels_not_ordinal_token_indices(prior_config):
    prior_config["model"].update(mask_ratio_min=1.0, mask_ratio_max=1.0, uncond_prob=0.0)
    model = CompletionPrior(prior_config)
    model.model = _RecordingPrior()
    # 0111 and 1000 differ by one ordinal index, but ALL four binary bits differ.
    with torch.no_grad():
        model.model.logits.copy_(torch.tensor([2., -2., -2., -2.]).reshape(1, 4, 1, 1))
    target = torch.tensor([0., 1., 1., 1.]).reshape(1, 4, 1, 1).expand(1, 4, 2, 3)
    loss, metrics = model({"bits": target})
    assert torch.isfinite(loss)
    assert int(metrics["bit_count"]) == 24
    assert int(metrics["token_count"]) == 6
    assert int(metrics["bit_correct"]) == 0
    assert int(metrics["token_correct"]) == 0
    assert float(metrics["bit_accuracy"]) == 0
    assert float(metrics["token_exact"]) == 0
    assert float(loss.detach()) > 2


def test_training_metrics_exclude_observed_cells(prior_config):
    prior_config["model"].update(mask_ratio_min=0., mask_ratio_max=0., uncond_prob=0.)
    model = CompletionPrior(prior_config)
    model.model = _RecordingPrior()
    _, metrics = model({"bits": torch.ones(2, 4, 3, 5)})
    assert int(metrics["bit_count"]) == 8
    assert int(metrics["token_count"]) == 2
    assert int(metrics["bit_correct"]) == 0
    assert int(metrics["token_correct"]) == 0


@pytest.mark.parametrize("bad", [torch.zeros(1, 3, 2, 2), torch.zeros(1, 4, 0, 2),
                                  torch.full((1, 4, 2, 2), 0.25), torch.full((1, 4, 2, 2), float("nan")),
                                  torch.zeros(4, 2, 2)])
def test_prior_rejects_nonbinary_or_malformed_bits(prior_config, bad):
    with pytest.raises(ValueError):
        CompletionPrior(prior_config)({"bits": bad})
    with pytest.raises(ValueError):
        masked_completion(_RecordingPrior(), bad, 0.5, 0)


@pytest.mark.parametrize("settings", [{"mask_ratio_min": -0.1}, {"mask_ratio_max": 1.1},
                                       {"mask_ratio_min": 0.9, "mask_ratio_max": 0.1},
                                       {"uncond_prob": 1.1}, {"spatial_mask": False}])
def test_invalid_prior_mask_config_raises(prior_config, settings):
    prior_config["model"].update(settings)
    with pytest.raises(ValueError):
        CompletionPrior(prior_config)


def test_prior_checkpoint_round_trip_is_restricted_and_strict(tmp_path, prior_config, bits, monkeypatch):
    model = CompletionPrior(prior_config).eval()
    path = tmp_path / "prior.pt"
    save_checkpoint(path, model, prior_config, epoch=2, step=7)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert raw["format"] == CHECKPOINT_FORMAT
    assert raw["model_type"] == "prior"
    assert raw["config"] == prior_config
    assert (raw["epoch"], raw["step"]) == (2, 7)
    original_load = torch.load
    calls = []

    def restricted_load(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("weights_only") is True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", restricted_load)
    restored, config = load_model(path, "cpu")
    assert calls and config == prior_config
    first, second = masked_completion(model, bits, 0.5, 9), masked_completion(restored, bits, 0.5, 9)
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)


def test_restricted_load_does_not_execute_pickled_objects(tmp_path):
    marker = tmp_path / "unpickler_executed"
    path = tmp_path / "untrusted.pt"
    torch.save(_UnsafePayload(marker), path)
    with pytest.raises((pickle.UnpicklingError, ValueError, RuntimeError)):
        load_checkpoint(path)
    assert not marker.exists()


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape", "task", "format"])
def test_checkpoint_mismatches_raise(tmp_path, prior_config, mutation):
    path = tmp_path / "prior.pt"
    payload = save_checkpoint(path, CompletionPrior(prior_config), prior_config)
    key = next(iter(payload["state_dict"]))
    if mutation == "missing":
        del payload["state_dict"][key]
    elif mutation == "unexpected":
        payload["state_dict"]["unrecognized.weight"] = torch.zeros(1)
    elif mutation == "shape":
        payload["state_dict"][key] = torch.empty(1)
    elif mutation == "task":
        payload["model_type"] = "tokenizer"
    else:
        payload["format"] = "legacy"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_model(path)


def test_checkpoint_refuses_unsafe_metadata_before_writing(tmp_path, prior_config):
    path = tmp_path / "unsafe.pt"
    with pytest.raises(TypeError):
        save_checkpoint(path, CompletionPrior(prior_config), prior_config, extra={"unsafe": object()})
    assert not path.exists()


def test_cache_round_trip_preserves_all_splits_and_binary_layout(token_cache, geometry_root):
    root, manifest, tokenizer = token_cache
    assert manifest["tokenizer_hash"] == tokenizer_hash(tokenizer)
    assert manifest["num_samples"] == 6
    assert manifest["split_counts"] == {"train": 4, "val": 1, "test": 1}
    assert manifest["layout"] == "CHW" and manifest["dtype"] == "uint8"
    assert manifest["bit_shape"] == [4, 3, 4]
    original = json.loads((geometry_root / "manifest.json").read_text())
    expected = {item["id"]: (item["split"], item["category"]) for item in original["samples"]}
    seen = set()
    for split, count in (("train", 4), ("val", 1), ("test", 1)):
        dataset = TokenDataset(root, split)
        assert len(dataset) == count
        for index in range(len(dataset)):
            row = dataset.get_item(index)
            assert row["bits"].dtype == torch.float32
            assert row["bits"].shape == (4, 3, 4)
            assert torch.all((row["bits"] == 0) | (row["bits"] == 1))
            assert (row["split"], row["category"]) == expected[row["sample_id"]]
            assert row["split"] == split
            assert row["sample_id"] not in seen
            seen.add(row["sample_id"])
            dataset.set_epoch(4)
            torch.testing.assert_close(row["bits"], dataset[index]["bits"], atol=0, rtol=0)
    assert seen == set(expected)
    assert len(TokenDataset(root, "all")) == 6
    assert str(geometry_root) not in (root / "manifest.json").read_text()
    for item in manifest["items"]:
        with np.load(root / item["path"], allow_pickle=False) as record:
            assert record["bits"].dtype == np.uint8
            assert record["tokenizer_hash"].item() == tokenizer_hash(tokenizer)


def test_tokenizer_hash_depends_on_configuration_and_tensor_state():
    tokenizer = _CacheTokenizer()
    original = tokenizer_hash(tokenizer)
    assert len(original) == 64
    assert original == tokenizer_hash(copy.deepcopy(tokenizer))
    tokenizer.offset.add_(1)
    assert original != tokenizer_hash(tokenizer)
    tokenizer.offset.sub_(1)
    tokenizer.specs["fixture"] = False
    assert original != tokenizer_hash(tokenizer)


@pytest.mark.parametrize("field", ["tokenizer_hash", "sample_id", "split", "category", "provenance"])
def test_cache_rejects_mismatched_record_metadata(token_cache, rewrite_npz, field):
    root, manifest, _ = token_cache
    value = "0" * 64 if field == "tokenizer_hash" else ("{}" if field == "provenance" else "different")
    rewrite_npz(root / manifest["items"][0]["path"], **{field: np.str_(value)})
    with pytest.raises(ValueError):
        TokenDataset(root)[0]


@pytest.mark.parametrize("value", [np.zeros((4, 3, 4), dtype=np.float32),
                                    np.full((4, 3, 4), 2, dtype=np.uint8),
                                    np.zeros((4, 4, 3), dtype=np.uint8)])
def test_cache_rejects_nonbinary_wrong_dtype_or_shape(token_cache, rewrite_npz, value):
    root, manifest, _ = token_cache
    rewrite_npz(root / manifest["items"][0]["path"], bits=value)
    with pytest.raises(ValueError):
        TokenDataset(root)[0]


@pytest.mark.parametrize("mutation", ["hash", "duplicate_id", "duplicate_path", "split", "count",
                                      "split_counts", "path", "layout", "format", "empty_split"])
def test_cache_rejects_invalid_manifests_without_implicit_splitting(token_cache, mutation):
    root, manifest, _ = token_cache
    if mutation == "hash":
        manifest["tokenizer_hash"] = "not-a-sha256"
    elif mutation == "duplicate_id":
        manifest["items"][-1]["id"] = manifest["items"][0]["id"]
    elif mutation == "duplicate_path":
        manifest["items"][-1]["path"] = manifest["items"][0]["path"]
    elif mutation == "split":
        manifest["items"][0]["split"] = "unspecified"
    elif mutation == "count":
        manifest["num_samples"] += 1
    elif mutation == "split_counts":
        manifest["split_counts"]["train"] += 1
    elif mutation == "path":
        manifest["items"][0]["path"] = "../outside.npz"
    elif mutation == "layout":
        manifest["layout"] = "HWC"
    elif mutation == "format":
        manifest["format"] = "legacy"
    else:
        for item in manifest["items"]:
            if item["split"] == "test":
                item["split"] = "train"
        manifest["split_counts"] = {"train": 5, "val": 1}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        TokenDataset(root, "test" if mutation == "empty_split" else "train")


def test_cache_export_refuses_overwrite_and_empty_input(token_cache, tmp_path):
    root, _, tokenizer = token_cache
    with pytest.raises(FileExistsError):
        export_tokens(tokenizer, [], root, "cpu")
    with pytest.raises(ValueError):
        export_tokens(tokenizer, [], tmp_path / "empty", "cpu")


def test_cache_export_rejects_duplicate_sample_identity(geometry_root, tmp_path):
    dataset = GeometryDataset(geometry_root, pc_size=8, samples_per_mesh=9, uniform_samples=7)
    row = next(iter(DataLoader(dataset, batch_size=1)))
    with pytest.raises(ValueError, match="(?i)duplicate"):
        export_tokens(_CacheTokenizer(), [row, row], tmp_path / "duplicate", "cpu")
