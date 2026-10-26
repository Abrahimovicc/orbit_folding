# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Manifest isolation, model-unit supervision, and exact epoch sampling."""

import json

import numpy as np
import pytest
import torch

from orbit_folding.data import GeometryDataset, balanced_indices, spatial_indices
from orbit_folding.preprocessing import import_preprocessed


def _manifest(root):
    return json.loads((root / "manifest.json").read_text())


def _save_manifest(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_splits_are_explicit_and_disjoint(geometry_root):
    identities = {}
    for split, count in (("train", 4), ("val", 1), ("test", 1), ("all", 6)):
        dataset = GeometryDataset(geometry_root, split, pc_size=8, samples_per_mesh=9, uniform_samples=7)
        assert len(dataset) == count
        rows = [dataset[i] for i in range(len(dataset))]
        identities[split] = {row["sample_id"] for row in rows}
        assert len(identities[split]) == count
        assert all(row["category"] == "generated_sphere" for row in rows)
        if split != "all":
            assert all(row["split"] == split for row in rows)
    assert not identities["train"] & identities["val"]
    assert not identities["train"] & identities["test"]
    assert not identities["val"] & identities["test"]
    assert identities["all"] == identities["train"] | identities["val"] | identities["test"]


@pytest.mark.parametrize("point_strategy", ["random", "spatial_mixed", "spatial_balanced"])
@pytest.mark.parametrize("query_strategy", ["sdf_value_balanced", "spatial_sdf_value_balanced"])
@pytest.mark.parametrize("counts", [(17, 31, 13), (101, 173, 71)])
def test_exact_counts_and_units(geometry_root, point_strategy, query_strategy, counts):
    pc, queries, uniform = counts
    dataset = GeometryDataset(geometry_root, pc_size=pc, samples_per_mesh=queries,
                              uniform_samples=uniform, point_strategy=point_strategy,
                              query_strategy=query_strategy)
    row = dataset[0]
    expected = {"points": (pc, 3), "normals": (pc, 3), "frame": (3, 3),
                "query_points": (queries, 3), "query_sdf": (queries,),
                "uniform_points": (uniform, 3), "uniform_sdf": (uniform,)}
    for key, shape in expected.items():
        assert row[key].shape == shape
        assert row[key].dtype == torch.float32
        assert torch.isfinite(row[key]).all()
    radius = _manifest(geometry_root)["samples"][0]["radius"]
    torch.testing.assert_close(row["points"].norm(dim=-1), torch.full((pc,), radius))
    torch.testing.assert_close(row["normals"], row["points"] / radius)
    for prefix in ("query", "uniform"):
        torch.testing.assert_close(row[f"{prefix}_sdf"],
                                   row[f"{prefix}_points"].norm(dim=-1) - radius,
                                   atol=1e-7, rtol=1e-5)
    with np.load(geometry_root / dataset.items[0]["path"], allow_pickle=False) as record:
        torch.testing.assert_close(row["frame"], torch.from_numpy(record["frame"]), rtol=0, atol=0)


@pytest.mark.parametrize("point_strategy", ["random", "spatial_mixed", "spatial_balanced"])
@pytest.mark.parametrize("query_strategy", ["sdf_value_balanced", "spatial_sdf_value_balanced"])
def test_sampling_is_repeatable_per_epoch_and_independent_of_access_order(
        geometry_root, point_strategy, query_strategy):
    options = dict(pc_size=17, samples_per_mesh=23, uniform_samples=11, seed=123,
                   point_strategy=point_strategy, query_strategy=query_strategy)
    first, second = (GeometryDataset(geometry_root, **options) for _ in range(2))
    initial = first[0]
    first[1]
    np.random.seed(999)
    for other in (first[0], second[0]):
        for key, value in initial.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(other[key], value, atol=0, rtol=0)
    first.set_epoch(3)
    second.set_epoch(3)
    changed = first[0]
    for key in ("points", "query_points", "uniform_points"):
        assert not torch.equal(changed[key], initial[key])
        torch.testing.assert_close(changed[key], second[0][key], atol=0, rtol=0)
    first.set_epoch(0)
    torch.testing.assert_close(first[0]["points"], initial["points"], atol=0, rtol=0)


def test_import_normalizes_points_and_both_sdf_fields_in_same_units(geometry_root, tmp_path):
    source, output = tmp_path / "legacy", tmp_path / "imported"
    source.mkdir()
    entry = _manifest(geometry_root)["samples"][0]
    with np.load(geometry_root / entry["path"], allow_pickle=False) as record:
        original = {key: record[key].copy() for key in record.files}
    np.savez_compressed(
        source / "record.npz", encoder_points=original["points"] * 2,
        encoder_normals=original["normals"], query_points=original["query_points"] * 2,
        query_sdf=original["query_sdf"] * 2, uniform_points=original["uniform_points"] * 2,
        uniform_sdf=original["uniform_sdf"] * 2,
    )
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["record.npz"], "val": [], "test": []}), encoding="utf-8")
    manifest = import_preprocessed(source, splits, output, input_scale=2.0)
    assert manifest["schema"] == "orbit-geometry-v1"
    assert [item["split"] for item in manifest["samples"]] == ["train"]
    with np.load(output / manifest["samples"][0]["path"], allow_pickle=False) as record:
        for key in ("points", "normals", "query_points", "query_sdf", "uniform_points", "uniform_sdf"):
            np.testing.assert_array_equal(record[key], original[key])
    row = GeometryDataset(output, pc_size=8, samples_per_mesh=19, uniform_samples=11)[0]
    for prefix in ("query", "uniform"):
        torch.testing.assert_close(row[f"{prefix}_sdf"],
                                   row[f"{prefix}_points"].norm(dim=-1) - entry["radius"],
                                   atol=1e-7, rtol=1e-5)


def test_import_rejects_cross_split_overlap(geometry_root, tmp_path):
    entry = _manifest(geometry_root)["samples"][0]
    source = tmp_path / "legacy"
    source.mkdir()
    with np.load(geometry_root / entry["path"], allow_pickle=False) as record:
        np.savez_compressed(source / entry["path"], encoder_points=record["points"],
                            encoder_normals=record["normals"], query_points=record["query_points"],
                            query_sdf=record["query_sdf"], uniform_points=record["uniform_points"],
                            uniform_sdf=record["uniform_sdf"])
    splits = tmp_path / "overlap.json"
    splits.write_text(json.dumps({"train": [entry["path"]], "test": [entry["path"]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)leak|split|overlap"):
        import_preprocessed(source, splits, tmp_path / "imported")


@pytest.mark.parametrize("strategy", ["sdf_value_balanced", "spatial_sdf_value_balanced"])
def test_all_five_query_bands_have_exact_quotas(geometry_root, strategy):
    row = GeometryDataset(geometry_root, pc_size=8, samples_per_mesh=20, uniform_samples=5,
                          query_strategy=strategy, query_ratios=[1, 2, 2, 2.5, 2.5])[0]
    sdf = row["query_sdf"]
    counts = [int((sdf.abs() <= 0.0005).sum()),
              int(((sdf < -0.0005) & (sdf >= -0.02)).sum()),
              int(((sdf > 0.0005) & (sdf <= 0.02)).sum()),
              int((sdf < -0.02).sum()), int((sdf > 0.02).sum())]
    assert counts == [2, 4, 4, 5, 5]


@pytest.mark.parametrize("spatial_bins", [None, 4])
@pytest.mark.parametrize("value", [0.0, -0.01, 0.01, -0.1, 0.1])
def test_empty_query_bands_fall_back_without_losing_samples(spatial_bins, value):
    points = np.arange(21, dtype=np.float32).reshape(7, 3)
    sdf = np.full(7, value, dtype=np.float32)
    first = balanced_indices(sdf, 29, np.random.default_rng(42), points=points, spatial_bins=spatial_bins)
    second = balanced_indices(sdf, 29, np.random.default_rng(42), points=points, spatial_bins=spatial_bins)
    assert first.shape == (29,)
    assert first.dtype.kind in "iu"
    assert np.all((0 <= first) & (first < len(sdf)))
    np.testing.assert_array_equal(first, second)


def test_zero_requested_samples_are_empty():
    assert balanced_indices(np.array([0.1]), 0, np.random.default_rng(1)).shape == (0,)
    assert spatial_indices(np.empty((0, 3)), [], 0, np.random.default_rng(1)).shape == (0,)


@pytest.mark.parametrize("options", [
    {"pc_size": 0}, {"pc_size": -1}, {"samples_per_mesh": 0}, {"uniform_samples": 0},
    {"point_strategy": "unsupported"}, {"query_strategy": "unsupported"},
    {"point_bins": 0}, {"query_bins": 0}, {"point_random_ratio": -0.1},
    {"point_random_ratio": 1.1}, {"query_ratios": [-1, 1, 1, 1, 1]},
    {"query_ratios": [0, 0, 0, 0, 0]}, {"query_ratios": [1, 2]},
])
def test_invalid_sampling_options_raise(geometry_root, options):
    with pytest.raises(ValueError):
        GeometryDataset(geometry_root, **options)[0]


@pytest.mark.parametrize("split", ["missing", "validation", ""])
def test_unknown_splits_raise(geometry_root, split):
    with pytest.raises(ValueError):
        GeometryDataset(geometry_root, split=split)


@pytest.mark.parametrize("mutation", ["schema", "empty_split", "duplicate_id", "duplicate_path", "unknown_split"])
def test_invalid_manifests_raise(geometry_root, mutation):
    manifest = _manifest(geometry_root)
    if mutation == "schema":
        manifest["schema"] = "unsupported"
    elif mutation == "empty_split":
        manifest["samples"] = [row for row in manifest["samples"] if row["split"] != "train"]
    elif mutation == "duplicate_id":
        manifest["samples"][-1]["id"] = manifest["samples"][0]["id"]
    elif mutation == "duplicate_path":
        manifest["samples"][-1]["path"] = manifest["samples"][0]["path"]
    else:
        manifest["samples"][-1]["split"] = "unassigned"
    _save_manifest(geometry_root, manifest)
    with pytest.raises(ValueError):
        GeometryDataset(geometry_root)[0]


@pytest.mark.parametrize("path", ["../outside.npz", "/tmp/outside.npz"])
def test_sample_paths_cannot_escape_root(geometry_root, path):
    manifest = _manifest(geometry_root)
    manifest["samples"][0]["path"] = path
    _save_manifest(geometry_root, manifest)
    with pytest.raises(ValueError, match="(?i)path|directory|root"):
        GeometryDataset(geometry_root)[0]


@pytest.mark.parametrize("field", ["points", "normals", "query_points", "query_sdf",
                                   "uniform_points", "uniform_sdf", "frame"])
def test_nonfinite_arrays_raise(geometry_root, rewrite_npz, field):
    path = geometry_root / _manifest(geometry_root)["samples"][0]["path"]
    with np.load(path, allow_pickle=False) as record:
        bad = record[field].copy()
    bad[...] = np.nan
    rewrite_npz(path, **{field: bad})
    with pytest.raises(ValueError):
        GeometryDataset(geometry_root)[0]


@pytest.mark.parametrize("field,shape", [
    ("points", (0, 3)), ("normals", (1, 3)), ("query_points", (0, 3)),
    ("query_sdf", (0,)), ("uniform_points", (0, 3)), ("uniform_sdf", (0,)),
    ("frame", (2, 2)), ("points", (96, 2)),
])
def test_malformed_arrays_raise_value_error(geometry_root, rewrite_npz, field, shape):
    path = geometry_root / _manifest(geometry_root)["samples"][0]["path"]
    rewrite_npz(path, **{field: np.zeros(shape, dtype=np.float32)})
    with pytest.raises(ValueError):
        GeometryDataset(geometry_root)[0]
