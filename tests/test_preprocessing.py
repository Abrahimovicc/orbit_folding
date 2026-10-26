# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Real bundled mesh preprocessing and same-unit research-data adaptation."""
import json

import numpy as np
import pytest

from orbit_folding.preprocessing import import_preprocessed, preprocess


def test_real_examples_have_consistent_scaled_sdf(project_root, tmp_path):
    import open3d as o3d
    output = tmp_path / "processed"
    manifest = preprocess(project_root / "data/samples/raw", output,
                          surface_points=128, query_points=512, uniform_points=64)
    assert sorted(row["split"] for row in manifest["samples"]) == ["test", "train", "train", "val"]
    assert len(manifest["samples"]) == 4
    assert manifest["preprocessing"]["normalization"] == "centroid"
    assert manifest["frame_policy"] == {"method": "pca", "axis_lock": True}
    for entry in manifest["samples"]:
        assert entry["license_spdx"] and entry["authors"] and entry["source_url"].startswith("https://")
        with np.load(output / entry["path"], allow_pickle=False) as sample:
            assert np.linalg.norm(sample["points"], axis=1).max() <= 0.50001
            assert np.isfinite(sample["query_sdf"]).all()
            np.testing.assert_allclose(sample["frame"].T @ sample["frame"], np.eye(3), atol=1e-5)
            mesh = o3d.io.read_triangle_mesh(str(output / f"{entry['id']}.ply"))
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            actual = scene.compute_signed_distance(o3d.core.Tensor(sample["query_points"])).numpy()
            np.testing.assert_allclose(actual, sample["query_sdf"], atol=2e-6)


@pytest.mark.parametrize("uniform_schema", ["arrays", "segments"])
@pytest.mark.parametrize("storage", ["compressed", "uncompressed"])
def test_import_applies_same_scale_once(geometry_root, tmp_path, uniform_schema, storage):
    raw = tmp_path / "source"
    raw.mkdir()
    original_manifest = json.loads((geometry_root / "manifest.json").read_text())
    splits = {"train": [], "val": [], "test": []}
    for entry in original_manifest["samples"]:
        with np.load(geometry_root / entry["path"], allow_pickle=False) as sample:
            values = dict(encoder_points=sample["points"] * 2, encoder_normals=sample["normals"],
                          query_points=sample["query_points"] * 2, query_sdf=sample["query_sdf"] * 2)
            if uniform_schema == "arrays":
                values.update(uniform_points=sample["uniform_points"] * 2, uniform_sdf=sample["uniform_sdf"] * 2)
            else:
                values.update(query_segment_names=np.array(["uniform"]),
                              query_segment_counts=np.array([len(sample["query_points"])]))
            np.savez_compressed(raw / entry["path"], **values)
        splits[entry["split"]].append(entry["path"])
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps(splits))
    output = tmp_path / "imported"
    result = import_preprocessed(raw, split_file, output, storage=storage)
    assert len(result["samples"]) == len(original_manifest["samples"])
    for entry, original in zip(result["samples"], sorted(original_manifest["samples"], key=lambda row: ["train", "val", "test"].index(row["split"]))):
        with np.load(output / entry["path"], allow_pickle=False) as converted, np.load(geometry_root / original["path"], allow_pickle=False) as expected:
            np.testing.assert_array_equal(converted["points"], expected["points"])
            np.testing.assert_array_equal(converted["query_points"], expected["query_points"])
            np.testing.assert_array_equal(converted["query_sdf"], expected["query_sdf"])
            np.testing.assert_allclose(converted["frame"], expected["frame"], atol=1e-6)
            target = expected["uniform_sdf"] if uniform_schema == "arrays" else expected["query_sdf"]
            np.testing.assert_array_equal(converted["uniform_sdf"], target)


def test_preprocess_rejects_open_geometry(tmp_path):
    import open3d as o3d
    root = tmp_path / "open"
    root.mkdir()
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                                    o3d.utility.Vector3iVector([[0, 1, 2]]))
    o3d.io.write_triangle_mesh(str(root / "triangle.ply"), mesh)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["triangle"], "val": [], "test": []}))
    with pytest.raises(ValueError, match="watertight"):
        preprocess(root, tmp_path / "rejected", split_file=split)


@pytest.mark.parametrize("normalization", ["centroid", "legacy-bbox"])
def test_preprocess_records_legacy_protocol_and_pca_only_policy(tmp_path, normalization):
    import open3d as o3d
    import torch
    from orbit_folding.geometry import normalization_parameters, representative_frame

    root = tmp_path / "raw"
    root.mkdir()
    mesh = o3d.geometry.TriangleMesh.create_cone(radius=0.4, height=1.3, resolution=12)
    o3d.io.write_triangle_mesh(str(root / "cone.ply"), mesh)
    split = tmp_path / "split.json"
    split.write_text('{"train": ["cone"]}')
    output = tmp_path / "processed"
    manifest = preprocess(root, output, surface_points=96, query_points=64, uniform_points=32,
                          split_file=split, normalization=normalization, axis_lock=False)
    assert manifest["frame_policy"] == {"method": "pca", "axis_lock": False}
    assert manifest["preprocessing"]["normalization"] == normalization
    center, scale = normalization_parameters(np.asarray(mesh.vertices), normalization)
    with np.load(output / "cone.npz") as sample:
        np.testing.assert_allclose(sample["center"], center, atol=1e-12)
        assert float(sample["scale"]) == pytest.approx(scale)
        np.testing.assert_array_equal(sample["frame"], representative_frame(torch.from_numpy(sample["points"]), lock=False))


def test_explicit_legacy_uniform_segment_and_original_frame_cache(geometry_root, tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    with np.load(geometry_root / "generated_0.npz") as sample:
        np.savez_compressed(root / "sample.npz", encoder_points=sample["points"] * 2,
            encoder_normals=sample["normals"], query_points=sample["query_points"] * 2,
            query_sdf=sample["query_sdf"] * 2)
        expected_uniform = sample["query_points"][100:].copy()
    split, frames = tmp_path / "split.json", tmp_path / "frames.npz"
    split.write_text('{"test": ["sample.npz"]}')
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    np.savez_compressed(frames, rel_paths=np.array(["sample.npz"]), rotations=rotation[None])
    output = tmp_path / "imported"
    manifest = import_preprocessed(root, split, output, legacy_uniform_start=100, frame_caches=[frames])
    assert len(manifest["frame_policy"]["cache_sha256"]) == 1
    assert manifest["preprocessing"]["legacy_uniform_start"] == 100
    with np.load(output / "sample.npz") as sample:
        np.testing.assert_array_equal(sample["uniform_points"], expected_uniform)
        np.testing.assert_array_equal(sample["frame"], rotation)


@pytest.mark.parametrize("start", [-1, 160, 1000])
def test_invalid_legacy_uniform_offset_is_rejected(geometry_root, tmp_path, start):
    root = tmp_path / "legacy"
    root.mkdir()
    with np.load(geometry_root / "generated_0.npz") as sample:
        np.savez_compressed(root / "sample.npz", encoder_points=sample["points"],
            encoder_normals=sample["normals"], query_points=sample["query_points"], query_sdf=sample["query_sdf"])
    split = tmp_path / "split.json"
    split.write_text('{"test": ["sample.npz"]}')
    with pytest.raises(ValueError, match="nonempty query suffix"):
        import_preprocessed(root, split, tmp_path / "imported", legacy_uniform_start=start)
