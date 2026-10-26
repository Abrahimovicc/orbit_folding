# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Model-space bounds, paper mesh export and observed-cloud fidelity contracts."""

import json

import numpy as np
import open3d as o3d
import pytest
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from orbit_folding.checkpoints import load_model, save_checkpoint
from orbit_folding.data import geometry_dataset
from orbit_folding.fidelity_benchmark import evaluate_fidelity, surface_distances
from orbit_folding.geometry import representative_frame
from orbit_folding.inference import decode_observation, encode_observation
from orbit_folding.meshing import MODEL_BOUNDS, clean_components, mesh_topology, validate_bounds
from orbit_folding.metrics import chamfer_distance
from orbit_folding.training import build_model


@pytest.mark.parametrize("bounds", [(0, 0), (1, -1), (0, float("nan")), (float("inf"), 1), (0,)])
def test_mesh_bounds_reject_invalid_ranges(bounds):
    with pytest.raises(ValueError, match="bounds"):
        validate_bounds(bounds)


def test_component_extent_keeps_thin_support_without_smoothing():
    body = o3d.geometry.TriangleMesh.create_sphere(radius=.3, resolution=12)
    rod = o3d.geometry.TriangleMesh.create_box(width=.02, height=1.0, depth=.02).translate([.8, 0, 0])
    speck = o3d.geometry.TriangleMesh.create_box(width=.005, height=.005, depth=.005).translate([-1, 0, 0])
    combined = body + rod + speck
    filtered = clean_components(combined, .02, 128)
    preserved = clean_components(combined, .02, 128, .08)
    assert len(filtered.triangles) == len(body.triangles)
    assert len(preserved.triangles) == len(body.triangles) + len(rod.triangles)
    expected = np.vstack([np.asarray(body.vertices), np.asarray(rod.vertices)])
    np.testing.assert_array_equal(np.asarray(preserved.vertices), expected)
    assert len(combined.triangles) > len(preserved.triangles)
    assert mesh_topology(np.asarray(preserved.vertices), np.asarray(preserved.triangles))["watertight"]


def test_fidelity_distances_match_original_torch_cd_and_hungarian_emd():
    rng = np.random.default_rng(3)
    predicted, reference = rng.normal(size=(11, 3)), rng.normal(size=(13, 3))
    actual = surface_distances(predicted, reference, emd_points=8)
    expected_cd = float(chamfer_distance(torch.from_numpy(predicted), torch.from_numpy(reference))) * 1e3
    costs = cdist(predicted[:8], reference[:8])
    rows, cols = linear_sum_assignment(costs)
    assert actual["cd_x1e3"] == pytest.approx(expected_cd, rel=1e-12)
    assert actual["emd_x100"] == pytest.approx(costs[rows, cols].mean() * 100, rel=1e-12)


def test_fidelity_uses_subsampled_observation_frame_and_counts_surface_ties(
        geometry_root, tokenizer_config, tmp_path, monkeypatch):
    import orbit_folding.fidelity_benchmark as fidelity

    checkpoint = tmp_path / "tokenizer.pt"
    save_checkpoint(checkpoint, build_model(tokenizer_config), tokenizer_config)
    model, config = load_model(checkpoint)
    recorded = []
    encode = model.encode

    def capture(points, frame=None):
        recorded.append((points.clone(), frame.clone()))
        return encode(points, frame)

    model.encode = capture
    model.decode = lambda planes, queries: torch.full_like(queries[..., 0], .1)
    monkeypatch.setattr(fidelity, "load_model", lambda *args: (model, config))
    report = evaluate_fidelity(checkpoint, geometry_root, tmp_path / "fidelity", metric_points=16,
                              resolution=8, emd_points=8, seed=14)
    points, frame = recorded[0]
    assert points.shape == (1, 16, 3)
    torch.testing.assert_close(frame, representative_frame(points))
    assert report["protocol"]["bounds"] == list(MODEL_BOUNDS)
    assert report["protocol"]["frame_source"] == "subsampled_observation"
    assert report["protocol"]["reference_surface"] == "same_subsampled_observation"
    target = geometry_dataset(geometry_root, config, "test", 14)[0]["query_sdf"].numpy()
    for branch in ("raw", "continuous", "quantized"):
        row = report["records"][0]["paths"][branch]
        assert row["sign_accuracy"] == pytest.approx(float((target >= 0).mean()))
        assert row["query_count"] == len(target)
        assert row["cd_x1e3"] is None and not row["mesh"]["valid"]
        assert report["summary"][branch]["samples"] == 1


@pytest.mark.parametrize("kind", ["lfq", "standard_vq"])
def test_fidelity_cli_is_independent_of_the_research_checkout(
        geometry_root, tokenizer_config, tmp_path, run_cli, kind):
    tokenizer_config["Tokenizer"]["type"] = kind
    checkpoint = tmp_path / "tokenizer.pt"
    save_checkpoint(checkpoint, build_model(tokenizer_config), tokenizer_config)
    output = tmp_path / "fidelity"
    result = run_cli("benchmark", "fidelity", "--checkpoint", checkpoint, "--data", geometry_root,
                     "--output", output, "--resolution", "8", "--metric-points", "16", "--emd-points", "8")
    assert json.loads(result.stdout)["samples"] == 1
    report = json.loads((output / "fidelity.json").read_text())
    assert report["codebook"]["total_tokens"] == tokenizer_config["plane_res"] ** 2
    assert report["failed_samples"] == []


def test_fidelity_retains_failed_observation_in_denominator(
        geometry_root, tokenizer_config, tmp_path, rewrite_npz):
    checkpoint = tmp_path / "tokenizer.pt"
    save_checkpoint(checkpoint, build_model(tokenizer_config), tokenizer_config)
    rewrite_npz(geometry_root / "generated_5.npz", query_sdf=np.full(160, np.nan))
    report = evaluate_fidelity(checkpoint, geometry_root, tmp_path / "fidelity", metric_points=16,
                              resolution=8, emd_points=8)
    assert report["samples"] == 1 and report["failed_samples"] == ["generated_5"]
    assert report["codebook"] is None
    for values in report["summary"].values():
        assert values["samples"] == 1 and values["valid_fraction"] == 0
        assert values["cd_x1e3"]["mean"] is None


def test_fidelity_retains_nonfinite_decoder_failure(
        geometry_root, tokenizer_config, tmp_path, monkeypatch):
    import orbit_folding.fidelity_benchmark as fidelity

    checkpoint = tmp_path / "tokenizer.pt"
    save_checkpoint(checkpoint, build_model(tokenizer_config), tokenizer_config)
    model, config = load_model(checkpoint)
    model.decode = lambda planes, queries: torch.full_like(queries[..., 0], float("nan"))
    monkeypatch.setattr(fidelity, "load_model", lambda *args: (model, config))
    report = evaluate_fidelity(checkpoint, geometry_root, tmp_path / "fidelity", metric_points=16,
                               resolution=8, emd_points=8)
    assert report["samples"] == 1 and report["failed_samples"] == ["generated_5"]
    assert "Nonfinite" in report["records"][0]["error"]
    assert report["summary"]["quantized"]["valid_fraction"] == 0


def test_fidelity_sampling_budget_is_independent_of_training_config(
        geometry_root, tokenizer_config, tmp_path):
    checkpoint = tmp_path / "tokenizer.pt"
    save_checkpoint(checkpoint, build_model(tokenizer_config), tokenizer_config)
    report = evaluate_fidelity(checkpoint, geometry_root, tmp_path / "fidelity", metric_points=16,
        resolution=8, emd_points=8, encoder_points=48, query_points=40, uniform_points=24)
    assert report["protocol"]["encoder_draw_points"] == 48
    assert report["protocol"]["query_draw_points"] == 40
    assert report["protocol"]["uniform_draw_points"] == 24
    assert report["records"][0]["paths"]["quantized"]["query_count"] == 40


def test_decode_reports_paper_bounds_and_extent_filtering(tokenizer_config, tmp_path, monkeypatch):
    import orbit_folding.inference as inference

    checkpoint, source = tmp_path / "tokenizer.pt", tmp_path / "points.npy"
    save_checkpoint(checkpoint, build_model(tokenizer_config), tokenizer_config)
    np.save(source, np.random.default_rng(4).normal(size=(96, 3)).astype(np.float32))
    encoded = tmp_path / "encoded.npz"
    encode_observation(checkpoint, source, encoded)
    model, config = load_model(checkpoint)
    model.decode = lambda planes, queries: queries.norm(dim=-1) - .3
    monkeypatch.setattr(inference, "load_model", lambda *args: (model, config))
    for label, bounds in [("paper", MODEL_BOUNDS), ("legacy", (-.6, .6))]:
        report = decode_observation(checkpoint, encoded, tmp_path / f"{label}.ply", resolution=16,
            bounds=bounds, component_min_fraction=.02, component_min_triangles=128,
            component_min_extent_fraction=.08)
        assert report["mesh"]["valid"]
        assert report["protocol"]["bounds"] == list(bounds)
        assert report["protocol"]["component_min_extent_fraction"] == .08
        assert report["protocol"]["smoothing"] is False
        assert "mesh_before_cleanup" in report
        mesh = o3d.io.read_triangle_mesh(str(tmp_path / f"{label}.ply"))
        assert report["mesh"]["vertices"] == len(mesh.vertices)
        assert report["mesh"]["triangles"] == len(mesh.triangles)


def test_decode_cleanup_updates_exported_mesh_counts(tokenizer_config, tmp_path, monkeypatch):
    import orbit_folding.inference as inference

    checkpoint, source = tmp_path / "tokenizer.pt", tmp_path / "points.npy"
    save_checkpoint(checkpoint, build_model(tokenizer_config), tokenizer_config)
    np.save(source, np.random.default_rng(4).normal(size=(96, 3)).astype(np.float32))
    encoded = tmp_path / "encoded.npz"
    encode_observation(checkpoint, source, encoded)
    model, config = load_model(checkpoint)

    def field(planes, queries):
        center = torch.tensor([.4, .4, .4], device=queries.device)
        return torch.minimum(queries.norm(dim=-1) - .3, (queries - center).norm(dim=-1) - .025)

    model.decode = field
    monkeypatch.setattr(inference, "load_model", lambda *args: (model, config))
    report = decode_observation(checkpoint, encoded, tmp_path / "clean.ply", resolution=32,
                                component_min_triangles=128)
    assert report["mesh_before_cleanup"]["triangles"] > report["mesh"]["triangles"]
    mesh = o3d.io.read_triangle_mesh(str(tmp_path / "clean.ply"))
    assert report["mesh"]["vertices"] == len(mesh.vertices)
    assert report["mesh"]["triangles"] == len(mesh.triangles)
    assert report["mesh"]["watertight"]
