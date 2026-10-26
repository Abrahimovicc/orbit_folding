# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Paper definitions, missing-cohort accounting and token-only equivalence."""

import json
import copy
import importlib.util

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from orbit_folding.benchmark_decode import decode_benchmark
from orbit_folding.checkpoints import save_checkpoint
from orbit_folding.data import geometry_dataset, reference_spatial_indices, spatial_indices
from orbit_folding.mesh_benchmark import compare_meshes, evaluate_mesh_directory, laplacian_scores
from orbit_folding.paper_evaluation import evaluate_token_prior
from orbit_folding.stress_benchmark import evaluate_stress
from orbit_folding.token_cache import export_tokens, tokenizer_hash
from orbit_folding.token_support import compare_caches
from orbit_folding.training import build_model


@pytest.fixture
def categorical_config(project_root):
    return yaml.safe_load((project_root / "configs/prior_categorical_demo.yaml").read_text())


@pytest.mark.parametrize("kind", ["lfq", "standard_vq"])
def test_fast_tokens_skip_decoders_and_preserve_indices(kind, tokenizer_config):
    config = copy.deepcopy(tokenizer_config)
    config["Tokenizer"]["type"] = kind
    model = build_model(config).eval()
    cloud = torch.randn(1, 64, 3) * .1
    frame = torch.eye(3)[None]
    full = model.encode(cloud, frame)
    before = tokenizer_hash(model)
    calls = []
    handle = model.tokenizer.decoder.register_forward_hook(lambda *args: calls.append(True))
    tokens = model.encode_tokens(cloud, frame)
    handle.remove()
    assert not calls and torch.equal(tokens["indices"], full["indices"])
    assert tokenizer_hash(model) == before and "quantized_planes" not in tokens
    if kind == "lfq":
        assert torch.equal(tokens["bits"], full["bits"])
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        model.encode_tokens(cloud, frame)


def test_point_only_loader_preserves_sampling_without_queries(geometry_root, tokenizer_config, rewrite_npz):
    full = geometry_dataset(geometry_root, tokenizer_config, "test", seed=17)[0]
    rewrite_npz(geometry_root / "generated_5.npz", query_sdf=np.full(160, np.nan))
    light = geometry_dataset(geometry_root, tokenizer_config, "test", seed=17, points_only=True)[0]
    assert torch.equal(light["points"], full["points"]) and torch.equal(light["frame"], full["frame"])
    assert "query_sdf" not in light and "normals" not in light


@pytest.mark.parametrize("requested", [0, 17, 80, 300])
def test_reference_spatial_selection_preserves_original_draws(requested):
    points = np.random.default_rng(71).normal(size=(240, 3)).astype(np.float32)
    candidates = np.arange(0, len(points), 2)
    actual_rng, reference_rng = np.random.RandomState(31), np.random.RandomState(31)
    actual = reference_spatial_indices(points, candidates, requested, actual_rng, bins=4)
    if requested <= 0:
        expected = np.empty(0, dtype=np.int64)
    elif len(candidates) <= requested:
        expected = candidates.copy()
    else:
        xyz = points[candidates]
        cells = np.floor((xyz - xyz.min(0)) / np.maximum(xyz.max(0) - xyz.min(0), 1e-6) * 4).astype(int)
        cells = np.clip(cells, 0, 3)
        keys = cells[:, 0] + 4 * (cells[:, 1] + 4 * cells[:, 2])
        visit = np.unique(keys)
        reference_rng.shuffle(visit)
        chosen, occupied = [], np.zeros(len(candidates), dtype=bool)
        for key in visit:
            local = np.flatnonzero(keys == key)
            take = min(max(1, requested // len(visit)), len(local), requested - len(chosen))
            if take <= 0:
                break
            selection = reference_rng.choice(local, take, replace=False)
            chosen.extend(candidates[selection].tolist())
            occupied[selection] = True
        if len(chosen) < requested:
            remaining = candidates[~occupied]
            chosen.extend(reference_rng.choice(remaining, requested - len(chosen), replace=False).tolist())
        expected = np.asarray(chosen, dtype=np.int64)
        reference_rng.shuffle(expected)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_rng.rand(12), reference_rng.rand(12))


def test_reference_cache_sampling_is_explicit_and_does_not_change_default(geometry_root, tokenizer_config, tmp_path, run_cli):
    with pytest.raises(ValueError, match="encoder-only"):
        geometry_dataset(geometry_root, tokenizer_config, sampling_protocol="reference")
    dataset = geometry_dataset(geometry_root, tokenizer_config, "test", seed=71,
                               points_only=True, sampling_protocol="reference")
    before = np.random.get_state()
    first = dataset[0]["points"]
    assert torch.equal(first, dataset[0]["points"])
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
    assert before[2:] == np.random.get_state()[2:]
    model = build_model(tokenizer_config)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, tokenizer_config)
    output = tmp_path / "reference_cache"
    run_cli("cache-tokens", "--checkpoint", checkpoint, "--data", geometry_root,
        "--sampling-protocol", "reference", "--seed", 71, "--output", output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["provenance"]["encoder_sampling"]["sampling_protocol"] == "reference"
    assert manifest["provenance"]["encoder_sampling"]["seed"] == 71
    assert geometry_dataset(geometry_root, tokenizer_config).sampling_protocol == "portable"


@pytest.mark.parametrize("strategy", ["random", "spatial_balanced", "spatial_mixed"])
def test_reference_encoder_sampling_chain(geometry_root, tokenizer_config, strategy):
    config = copy.deepcopy(tokenizer_config)
    config["PointSamplingStrategy"] = strategy
    dataset = geometry_dataset(geometry_root, config, "test", seed=83,
                               points_only=True, sampling_protocol="reference")
    with np.load(geometry_root / dataset.items[0]["path"]) as record:
        points = record["points"]
    rng = np.random.RandomState(83)
    if len(points) <= dataset.pc_size:
        selected = (rng.choice(len(points), dataset.pc_size, replace=True) if len(points)<dataset.pc_size
                    else np.arange(len(points)))
    elif strategy == "random":
        selected = rng.choice(len(points), dataset.pc_size, replace=False)
    else:
        random_count = round(dataset.pc_size * dataset.point_random_ratio) if strategy == "spatial_mixed" else 0
        selected = reference_spatial_indices(points, np.arange(len(points)), dataset.pc_size-random_count,
                                             rng, dataset.point_bins)
        if strategy == "spatial_mixed":
            selected = np.r_[selected, rng.choice(len(points),random_count,replace=False)]
            rng.shuffle(selected)
    np.testing.assert_array_equal(dataset[0]["points"].numpy(), points[selected])


def test_portable_encoder_sampling_keeps_preexisting_draw_order(geometry_root, tokenizer_config):
    config = copy.deepcopy(tokenizer_config)
    config["PointSamplingStrategy"] = "spatial_mixed"
    dataset = geometry_dataset(geometry_root, config, "test", seed=83, points_only=True)
    with np.load(geometry_root / dataset.items[0]["path"]) as record:
        points = record["points"]
    rng = np.random.default_rng(83)
    random_count = round(dataset.pc_size * dataset.point_random_ratio)
    spatial = spatial_indices(points,np.arange(len(points)),dataset.pc_size-random_count,rng,dataset.point_bins)
    random = rng.choice(len(points),random_count,replace=len(points)<random_count)
    selected = rng.permutation(np.r_[spatial,random])
    np.testing.assert_array_equal(dataset[0]["points"].numpy(),points[selected])


def test_boundary_repetition_uses_paired_objects_not_independent_repeats(project_root):
    spec = importlib.util.spec_from_file_location("boundary_check", project_root / "tools/check_boundary_reproduction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"sample_id":name,"mode":mode,"valid":mode!="categorical" or name!="b",
             "edge_distance":value if mode!="categorical" or name!="b" else None}
            for name in ("a","b") for mode,value in (("relaxed",.1),("hard",.2),("categorical",.05))]
    first = {"seed":1,"samples":2,"mesh_hashes":[{"id":"a"},{"id":"b"}],
        "summary":{name:{} for name in ("relaxed","hard","categorical")}, "rows":rows,
        "sample_points":1024,"edge_percentile":80,"drift_threshold":.005}
    second = copy.deepcopy(first)
    second["seed"] = 2
    result = module.summarize_repeats([first,second],bootstrap=32)
    assert result["paired_edge"]["hard"]["mean_delta"] == pytest.approx(.1)
    assert result["paired_edge"]["hard"]["bootstrap_ci95"] == pytest.approx([.1,.1])
    assert result["paired_edge"]["categorical"]["valid_pairs"] == 1
    assert result["paired_edge"]["categorical"]["excluded_pairs"] == 1
    assert result["paired_edge"]["categorical"]["mean_delta"] == pytest.approx(-.05)
    second["mesh_hashes"][0]["id"] = "changed"
    with pytest.raises(ValueError,match="unchanged"):
        module.summarize_repeats([first,second])


def test_boundary_repetition_cli_keeps_missing_meshes(tmp_path, run_python):
    root = tmp_path / "decoded"
    root.mkdir()
    (root / "decoded_manifest.json").write_text(json.dumps({
        "format":"orbit_folding.decoded_set.v1", "modes":["relaxed","hard"],
        "samples":[{"id":"missing","oracle":"oracle.ply","relaxed":"soft.ply","hard":"hard.ply"}]}))
    output = tmp_path / "repeat"
    run_python("tools/check_boundary_reproduction.py","--decode-dir",root,"--output",output,
        "--seeds",1,2,"--expected-samples",1,"--bootstrap",16)
    result = json.loads((output / "boundary_repetition.json").read_text())
    assert result["samples"] == 1 and result["seeds"] == [1,2]
    assert result["paired_edge"]["hard"]["valid_pairs"] == 0
    assert result["paired_edge"]["hard"]["mean_delta"] is None
    assert all(r["summary"]["hard"]["failure_inclusive_fraction"] == 1 for r in result["per_seed"])


def test_mesh_benchmark_scale_invariance_and_failure():
    import open3d as o3d
    oracle = o3d.geometry.TriangleMesh.create_box().compute_vertex_normals()
    predicted = o3d.geometry.TriangleMesh(oracle).translate([.1, 0, 0])
    first = compare_meshes(oracle, predicted, sample_points=256, seed=2)
    second = compare_meshes(o3d.geometry.TriangleMesh(oracle).scale(3, [0,0,0]),
                          o3d.geometry.TriangleMesh(predicted).scale(3, [0,0,0]), sample_points=256, seed=2)
    assert first["cd_n"] == pytest.approx(second["cd_n"], rel=1e-6)
    assert first["edge_distance"] == pytest.approx(second["edge_distance"], rel=1e-6)
    assert compare_meshes(oracle, None)["drift_failure"] is None
    assert compare_meshes(oracle, o3d.geometry.TriangleMesh(oracle).translate([10,0,0]))["drift_failure"]


def test_laplacian_matches_unvectorized_definition():
    vertices = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    faces = np.array([[0,1,2],[0,2,3],[0,3,1],[1,3,2]])
    edges = np.unique(np.sort(np.concatenate([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]]),axis=1),axis=0)
    expected = []
    length = np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1).mean()
    for i, point in enumerate(vertices):
        neighbors = [v if u == i else u for u,v in edges if i in (u,v)]
        expected.append(np.linalg.norm(point-vertices[neighbors].mean(0))/length)
    np.testing.assert_allclose(laplacian_scores(vertices, faces), expected)


def test_missing_meshes_remain_in_explicit_cohort(tmp_path):
    root = tmp_path / "decoded"
    root.mkdir()
    manifest = {"format":"orbit_folding.decoded_set.v1", "modes":["relaxed"],
        "samples":[{"id":"missing", "oracle":"oracle.ply", "relaxed":"relaxed.ply"}]}
    (root / "decoded_manifest.json").write_text(json.dumps(manifest))
    result = evaluate_mesh_directory(root, tmp_path / "metrics", expected_samples=1)
    summary = result["summary"]["relaxed"]
    assert summary["valid"] == 0 and summary["drift_valid_fraction"] is None
    assert summary["failure_inclusive_fraction"] == 1
    with pytest.raises(ValueError, match="cohort"):
        evaluate_mesh_directory(root, tmp_path / "bad", expected_samples=2)
    manifest["samples"][0]["oracle"] = "../escape.ply"
    (root / "decoded_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        evaluate_mesh_directory(root, tmp_path / "bad")


def test_seeded_stress_is_frame_isolated_and_records_units(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    cloud = np.random.default_rng(8).normal(size=(64,3)) * [3,1,.2]
    np.save(raw / "object.npy", cloud)
    result = evaluate_stress(raw, tmp_path / "metric", split="all", points=64)
    assert result["requested_samples"] == 1 and not result["failed_samples"]
    assert result["summary"]["base_observation"]["orbit"]["p95"] < 1e-14
    assert result["protocol"]["jitter_sigma_model"] == .0025
    again = evaluate_stress(raw, tmp_path / "again", split="all", points=64)
    assert result["rows"] == again["rows"]


def test_scan_adapter_needs_no_sdf_labels_or_visibility_masks(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "scan.h5"
    with h5py.File(source, "w") as record:
        record["data"] = np.random.default_rng(17).normal(size=(3,64,3)).astype(np.float32)
    result = evaluate_stress(source, tmp_path / "scan_metrics", kind="scanobjectnn",
                             points=32, limit=2, expected_samples=2)
    assert result["requested_samples"] == 2 and not result["failed_samples"]
    assert result["protocol"]["source_sha256"]


def test_shared_decoder_benchmark_cli_and_safe_categorical_default(geometry_root, tokenizer_config, prior_config,
                                                                  categorical_config, tmp_path, run_cli):
    tokenizer = build_model(tokenizer_config).eval()
    cache = tmp_path / "tokens"
    export_tokens(tokenizer, DataLoader(geometry_dataset(geometry_root, tokenizer_config, "all")), cache, "cpu")
    tokenizer_path = tmp_path / "tokenizer.pt"
    save_checkpoint(tokenizer_path, tokenizer, tokenizer_config)
    prior_paths = []
    for config, name in ((prior_config,"bit"), (categorical_config,"categorical")):
        path = tmp_path / f"{name}.pt"
        save_checkpoint(path, build_model(config), config, extra={"tokenizer_hash":tokenizer_hash(tokenizer)})
        prior_paths.append(path)
    report = evaluate_token_prior(prior_paths[1], token_root=cache, output=tmp_path / "cat", ratios=[0,.5])
    assert report["batch_size"] == 1 and report["rows"][0]["ce"] is None
    output = tmp_path / "decoded"
    run_cli("benchmark", "decode", "--tokenizer", tokenizer_path, "--bit-prior", prior_paths[0],
            "--categorical-prior", prior_paths[1], "--tokens", cache, "--output", output,
            "--resolution", 8, "--threads", 1)
    manifest = json.loads((output / "decoded_manifest.json").read_text())
    assert manifest["modes"] == ["relaxed","hard","categorical"] and len(manifest["samples"]) == 1
    assert all(manifest["prior_bindings_verified"].values())
    masks = np.load(output / "masks.npy")
    replay = decode_benchmark(tokenizer_path, tmp_path / "replay", bit_prior=prior_paths[0],
        token_root=cache, mask_npy=output / "masks.npy", resolution=8)
    assert np.array_equal(masks, np.load(tmp_path / "replay/masks.npy"))
    assert replay["protocol"]["supplied_mask_sha256"]


def test_fresh_support_control_requires_matched_cohorts(geometry_root, tokenizer_config, tmp_path):
    roots = []
    for kind in ("standard_vq","lfq"):
        config = copy.deepcopy(tokenizer_config)
        config["Tokenizer"]["type"] = kind
        model = build_model(config).eval()
        root = tmp_path / kind
        export_tokens(model, DataLoader(geometry_dataset(geometry_root, config, "all")), root, "cpu")
        roots.append(root)
    report = compare_caches(*roots, tmp_path / "comparison", vocabulary=16)
    assert len(report["series"]) == 2 and report["series"][0]["total_tokens"] == 64
    with pytest.raises(ValueError, match="vocabulary"):
        compare_caches(*roots, tmp_path / "bad")


def test_validation_split_keeps_test_and_preserves_training_cohort(project_root):
    path = project_root / "tools/make_validation_split.py"
    spec = importlib.util.spec_from_file_location("validation_split", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = {"train": [f"shape_{i}" for i in range(20)], "test": ["test_only"]}
    split = module.make_split(source, .1, 3)
    assert split["test"] == source["test"] and len(split["val"]) == 2
    assert not set(split["val"]) & set(split["train"])
    assert set(split["train"] + split["val"]) == set(source["train"])
    assert split == module.make_split(source, .1, 3)


@pytest.mark.integration
def test_reproduction_workflow_records_smoke_without_claiming_quality(geometry_root, tmp_path, run_python):
    output = tmp_path / "workflow"
    run_python("tools/reproduce.py", "--data", geometry_root, "--output", output, "--smoke", "--device", "cpu", timeout=180)
    report = json.loads((output / "workflow.json").read_text())
    assert report["status"] == "complete" and report["numerical_agreement"] == "not_applicable_smoke"
    assert not report["full_training_budget_executed"] and len(report["steps"]) == 7
    assert all(row["returncode"] == 0 for row in report["steps"])
    assert report["checkpoint_selection"] == "last"
    cache_command = report["steps"][1]["command"]
    assert cache_command[cache_command.index("--checkpoint")+1].endswith("last.pt")
