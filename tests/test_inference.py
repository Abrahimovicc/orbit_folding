# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Inference needs observed coordinates, never training SDF targets."""

import json

import numpy as np
import pytest
import torch

from orbit_folding.checkpoints import load_model, save_checkpoint
from orbit_folding.inference import decode_observation, encode_observation
from orbit_folding.provenance import file_sha256
from orbit_folding.training import build_model
from orbit_folding.weights import WEIGHTS_FORMAT, verify_weights


@pytest.fixture
def inference_setup(tmp_path, tokenizer_config):
    checkpoint = tmp_path / "tokenizer.pt"
    save_checkpoint(checkpoint, build_model(tokenizer_config), tokenizer_config)
    rng = np.random.default_rng(54)
    source = tmp_path / "observation.npy"
    np.save(source, rng.normal(size=(96, 3)).astype(np.float32) * [3, 2, 1] + [10, -4, 7])
    return checkpoint, source


def test_encode_accepts_plain_points_without_sdf_arrays(inference_setup, tmp_path):
    checkpoint, source = inference_setup
    target = tmp_path / "encoded.npz"
    report = encode_observation(checkpoint, source, target)
    assert report["observed_points"] == 96 and report["encoder_points"] == 64
    assert report["source_sha256"] == file_sha256(source)
    with np.load(target) as encoded:
        assert encoded["bits"].dtype == np.uint8
        np.testing.assert_allclose(encoded["frame"].T @ encoded["frame"], np.eye(3), atol=1e-5)
    with pytest.raises(FileExistsError):
        encode_observation(checkpoint, source, target)


def test_inference_cli_reports_mesh_status(inference_setup, tmp_path, run_cli):
    checkpoint, source = inference_setup
    encoded = tmp_path / "encoded.npz"
    result = run_cli("encode", "--checkpoint", checkpoint, "--input", source,
                     "--output", encoded, "--threads", "1")
    assert json.loads(result.stdout)["observed_points"] == 96
    output = tmp_path / "decoded.ply"
    result = run_cli("decode", "--checkpoint", checkpoint, "--tokens", encoded,
                     "--output", output, "--resolution", "8", "--threads", "1", check=False)
    report = json.loads(output.with_suffix(".json").read_text())
    assert result.returncode == (0 if report["mesh"]["valid"] else 2)
    assert output.exists() == report["mesh"]["valid"]


def test_empty_isosurface_is_a_failure_not_a_proxy_mesh(inference_setup, tmp_path):
    checkpoint, source = inference_setup
    encoded = tmp_path / "encoded.npz"
    encode_observation(checkpoint, source, encoded)
    report = decode_observation(checkpoint, encoded, tmp_path / "empty.ply", resolution=8)
    assert report["mesh"]["status"] == "no_isosurface"
    assert not (tmp_path / "empty.ply").exists()
    assert json.loads((tmp_path / "empty.json").read_text())["mesh"]["valid"] is False


def test_decode_checks_tokenizer_identity(inference_setup, tmp_path, tokenizer_config):
    checkpoint, source = inference_setup
    encoded = tmp_path / "encoded.npz"
    encode_observation(checkpoint, source, encoded)
    other = tmp_path / "other.pt"
    save_checkpoint(other, build_model(tokenizer_config), tokenizer_config)
    with pytest.raises(ValueError, match="different tokenizer"):
        decode_observation(other, encoded, tmp_path / "wrong.ply", resolution=4)


def test_decode_rejects_incorrect_spatial_dimensions(inference_setup, tmp_path):
    checkpoint, source = inference_setup
    encoded = tmp_path / "encoded.npz"
    encode_observation(checkpoint, source, encoded)
    with np.load(encoded) as data:
        record = {key: data[key] for key in data.files}
    record["bits"] = record["bits"][:, :-1]
    np.savez_compressed(encoded, **record)
    with pytest.raises(ValueError, match="spatial dimensions"):
        decode_observation(checkpoint, encoded, tmp_path / "wrong.ply", resolution=4)


def test_encode_honors_the_configured_axis_lock(tmp_path, tokenizer_config):
    from orbit_folding.geometry import representative_frame

    config = {**tokenizer_config, "EnableAxisRepresentativeLock": False}
    checkpoint = tmp_path / "unlocked.pt"
    save_checkpoint(checkpoint, build_model(config), config)
    points = np.random.default_rng(3).normal(size=(96, 3)).astype(np.float32) * [3, 2, 1]
    points = points.astype(np.float32)
    source = tmp_path / "points.npy"
    np.save(source, points)
    encoded = tmp_path / "encoded.npz"
    encode_observation(checkpoint, source, encoded, model_space=True)
    with np.load(encoded) as record:
        np.testing.assert_allclose(record["frame"], representative_frame(torch.from_numpy(points), lock=False), atol=1e-6)


def test_decode_restores_input_coordinates(inference_setup, tmp_path, monkeypatch):
    import open3d as o3d
    import orbit_folding.inference as inference

    checkpoint, source = inference_setup
    encoded = tmp_path / "encoded.npz"
    encode_observation(checkpoint, source, encoded)
    model, config = load_model(checkpoint)
    model.decode = lambda planes, queries: queries.norm(dim=-1) - 0.3
    monkeypatch.setattr(inference, "load_model", lambda *args, **kwargs: (model, config))
    for space in ("input", "representative"):
        report = decode_observation(checkpoint, encoded, tmp_path / f"{space}.ply", resolution=12, space=space)
        assert report["mesh"]["valid"]
    with np.load(encoded) as record:
        frame, center, scale = record["frame"], record["center"], float(record["scale"])
    representative = np.asarray(o3d.io.read_triangle_mesh(str(tmp_path / "representative.ply")).vertices)
    actual = np.asarray(o3d.io.read_triangle_mesh(str(tmp_path / "input.ply")).vertices)
    np.testing.assert_allclose(actual, representative @ frame.T * scale + center, rtol=1e-6, atol=2e-6)


def test_pretrained_manifest_checks_pair_hash_and_file_checksums(inference_setup, tmp_path, prior_config):
    from orbit_folding.token_cache import tokenizer_hash

    checkpoint, _ = inference_setup
    tokenizer, _ = load_model(checkpoint)
    digest = tokenizer_hash(tokenizer)
    prior_path = tmp_path / "prior.pt"
    save_checkpoint(prior_path, build_model(prior_config), prior_config, extra={"tokenizer_hash": digest})
    manifest = {"format": WEIGHTS_FORMAT, "pairs": [{"id": "test", "tokenizer_hash": digest,
        "tokenizer": {"path": checkpoint.name, "sha256": file_sha256(checkpoint)},
        "prior": {"path": prior_path.name, "sha256": file_sha256(prior_path)}}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    verify_weights(tmp_path)
    prior_path.write_bytes(prior_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_weights(tmp_path)
