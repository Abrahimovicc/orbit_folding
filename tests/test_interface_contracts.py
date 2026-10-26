# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Cross-module regressions for normalization, frame policies and sampling."""

import json

import numpy as np
import pytest
import torch

from orbit_folding.checkpoints import save_checkpoint
from orbit_folding.data import geometry_dataset
from orbit_folding.evaluation import _field_options, _geometry_dataset, load_observation
from orbit_folding.geometry import normalization_parameters, random_rotations, representative_frame
from orbit_folding.inference import encode_observation
from orbit_folding.training import build_model


@pytest.mark.parametrize("strategy", ["random", "spatial_mixed", "spatial_balanced"])
def test_raw_encode_is_rotation_consistent_including_subsampling(tmp_path, tokenizer_config, monkeypatch, strategy):
    import orbit_folding.inference as inference

    torch.manual_seed(122)
    config = {**tokenizer_config, "PointSamplingStrategy": strategy}
    checkpoint = tmp_path / "tokenizer.pt"
    model = build_model(config).eval()
    save_checkpoint(checkpoint, model, config)
    monkeypatch.setattr(inference, "load_model", lambda *args: (model, config))
    representatives = []
    encode = model.encode_tokens

    def record(points, frame):
        representatives.append((points @ frame).clone())
        return encode(points, frame)

    monkeypatch.setattr(model, "encode_tokens", record)
    points = np.random.default_rng(96).normal(size=(512, 3)) * [3, 1, 0.3]
    points[:, 1] += 0.2 * points[:, 0] ** 2
    torch.manual_seed(72)
    rotation = random_rotations(1, "cpu", torch.float64)[0].numpy()
    encoded_bits = []
    scales = []
    for index, cloud in enumerate((points, points @ rotation.T)):
        source, encoded = tmp_path / f"points_{index}.npy", tmp_path / f"encoded_{index}.npz"
        np.save(source, cloud)
        report = encode_observation(checkpoint, source, encoded)
        assert report["normalization"] == "centroid" and report["sampling_frame"] == "representative"
        with np.load(encoded) as record:
            scales.append(float(record["scale"]))
            encoded_bits.append(record["bits"].copy())
    assert scales[0] == pytest.approx(scales[1], rel=1e-12)
    torch.testing.assert_close(representatives[0], representatives[1], atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(*encoded_bits)


def test_legacy_bbox_encoding_preserves_the_old_transform(tmp_path, tokenizer_config):
    checkpoint = tmp_path / "tokenizer.pt"
    save_checkpoint(checkpoint, build_model(tokenizer_config), tokenizer_config)
    points = np.random.default_rng(9).normal(size=(96, 3))
    source, output = tmp_path / "points.npy", tmp_path / "tokens.npz"
    np.save(source, points)
    report = encode_observation(checkpoint, source, output, normalization="legacy-bbox")
    center = (points.min(0) + points.max(0)) / 2
    scale = 2 * np.linalg.norm(points - center, axis=1).max()
    with np.load(output) as record:
        np.testing.assert_array_equal(record["center"], center)
        assert float(record["scale"]) == scale
    assert report["sampling_frame"] == "observation"


@pytest.mark.parametrize("policy", ["invalid", "model-space"])
def test_unknown_raw_normalization_is_rejected(policy):
    with pytest.raises(ValueError, match="Normalization"):
        normalization_parameters(np.eye(3), policy)


def test_raw_stability_loader_uses_the_shared_normalization(tmp_path):
    points = np.random.default_rng(3).normal(size=(96, 3)) * [3, 2, 1]
    np.save(tmp_path / "points.npy", points)
    center, scale = normalization_parameters(points)
    loaded = load_observation(tmp_path, {"path": "points.npy", "processed": False})
    np.testing.assert_allclose(loaded.numpy(), (points - center) / scale, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("cached", [False, True, None])
def test_configured_frame_is_correct_before_encoder_subsampling(
        geometry_root, tokenizer_config, rewrite_npz, requested, cached):
    manifest_path = geometry_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if cached is not None:
        manifest["frame_policy"] = {"method": "pca", "axis_lock": cached}
        manifest_path.write_text(json.dumps(manifest))
    with np.load(geometry_root / "generated_0.npz") as record:
        points = torch.from_numpy(record["points"])
    if cached is not None:
        rewrite_npz(geometry_root / "generated_0.npz", frame=representative_frame(points, lock=cached).numpy())
    dataset = geometry_dataset(geometry_root, {**tokenizer_config, "PCsize": 8,
                                              "EnableAxisRepresentativeLock": requested})
    torch.testing.assert_close(dataset[0]["frame"], representative_frame(points, lock=requested), rtol=0, atol=0)
    assert dataset.sampling_settings()["axis_lock"] == requested
    assert dataset.sampling_settings()["frame_source"] == ("cache" if requested == cached else "full_observation")


def test_evaluation_and_training_share_sampling_contract(geometry_root, tokenizer_config):
    options = _field_options(dict(pc_size=64, samples_per_mesh=64, uniform_samples=32, seed=2026))
    configured = geometry_dataset(geometry_root, tokenizer_config, "test", seed=2026)
    evaluated = _geometry_dataset(geometry_root, "test", tokenizer_config, options)
    assert configured.sampling_settings() == evaluated.sampling_settings() == options["sampling"]
    for key, value in configured[0].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, evaluated[0][key], rtol=0, atol=0)


def test_evaluation_budgets_override_counts_without_changing_strategy(geometry_root, tokenizer_config):
    options = _field_options(dict(pc_size=12, samples_per_mesh=16, uniform_samples=8))
    dataset = _geometry_dataset(geometry_root, "test", tokenizer_config, options)
    assert dataset[0]["points"].shape == (12, 3)
    assert dataset[0]["query_points"].shape == (16, 3)
    assert options["sampling"]["point_strategy"] == tokenizer_config["PointSamplingStrategy"]
    assert options["sampling"]["query_strategy"] == tokenizer_config["QuerySamplingStrategy"]


@pytest.mark.parametrize("value", [-1, 16, 99, 1.9, float("nan"), float("inf"), True])
@pytest.mark.parametrize("method", ["indices_to_bits", "decode_indices"])
def test_invalid_token_indices_are_never_truncated_or_wrapped(tokenizer_config, value, method):
    tokenizer = build_model(tokenizer_config).tokenizer
    with pytest.raises(ValueError, match="indices"):
        getattr(tokenizer, method)(torch.full((1, 8, 8), value))


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_bit_probabilities_are_rejected(tokenizer_config, value):
    model = build_model(tokenizer_config)
    with pytest.raises(ValueError, match="probabilities"):
        model.decode_probabilities(torch.full((1, 4, 8, 8), value))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
                         + [getattr(torch, name) for name in ("uint16", "uint32", "uint64") if hasattr(torch, name)])
def test_small_integer_indices_are_valid_for_large_vocabularies(tokenizer_config, dtype):
    config = {**tokenizer_config, "Tokenizer": {**tokenizer_config["Tokenizer"], "codebook_size": 512}}
    tokenizer = build_model(config).tokenizer
    indices = torch.ones((1, 8, 8), dtype=dtype)
    torch.testing.assert_close(tokenizer.indices_to_bits(indices), tokenizer.indices_to_bits(indices.long()))
