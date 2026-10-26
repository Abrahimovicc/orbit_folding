# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Paper decoded-boundary metrics, separate from field-grid diagnostics."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .metrics import summary
from .provenance import file_sha256


LEGACY_MODES = {"relaxed": "pred_soft_bits", "hard": "pred_hard_bits", "categorical": "categorical_index_prior"}


def read_mesh(path):
    import open3d as o3d

    path = Path(path)
    if not path.is_file():
        return None
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not len(mesh.vertices) or not len(mesh.triangles) or not np.isfinite(np.asarray(mesh.vertices)).all():
        return None
    mesh.remove_duplicated_vertices().remove_duplicated_triangles().remove_degenerate_triangles().remove_unreferenced_vertices()
    if not len(mesh.triangles):
        return None
    mesh.compute_vertex_normals()
    return mesh


def sample_surface(mesh, count, seed):
    import open3d as o3d

    o3d.utility.random.seed(int(seed) % (2**31 - 1))
    cloud = mesh.sample_points_uniformly(number_of_points=int(count))
    points, normals = np.asarray(cloud.points), np.asarray(cloud.normals).copy()
    if points.shape != normals.shape or not np.isfinite(points).all():
        raise ValueError("Invalid surface samples or normals")
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return points, normals


def laplacian_scores(vertices, faces):
    """Uniform vertex Laplacian, normalized by mean unique-edge length."""
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    neighbors, degree = np.zeros_like(vertices, dtype=float), np.zeros(len(vertices), dtype=float)
    np.add.at(neighbors, edges[:, 0], vertices[edges[:, 1]])
    np.add.at(neighbors, edges[:, 1], vertices[edges[:, 0]])
    np.add.at(degree, edges.ravel(), 1)
    length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1).mean()
    valid = degree > 0
    result = np.zeros(len(vertices))
    result[valid] = np.linalg.norm(vertices[valid] - neighbors[valid] / degree[valid, None], axis=1) / max(length, 1e-12)
    return result


def compare_meshes(oracle, predicted, *, sample_points=1024, seed=20260518,
                   edge_percentile=80, drift_threshold=.005):
    if int(sample_points) != sample_points or sample_points < 1:
        raise ValueError("sample_points must be a positive integer")
    if not 0 <= edge_percentile <= 100 or not np.isfinite(drift_threshold) or drift_threshold < 0:
        raise ValueError("Invalid edge percentile or drift threshold")
    empty = dict(valid=False, cd_n=None, fscore_1pct=None, normal_consistency=None,
                 edge_distance=None, drift_failure=None)
    if oracle is None or predicted is None:
        return {**empty, "status": "missing_or_invalid_mesh"}
    diagonal = float(np.linalg.norm(np.ptp(np.asarray(oracle.vertices), axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 1e-12:
        return {**empty, "status": "degenerate_oracle"}
    try:
        truth, truth_normals = sample_surface(oracle, sample_points, seed)
        points, normals = sample_surface(predicted, sample_points, seed + 17)
    except (ValueError, RuntimeError) as error:
        return {**empty, "status": str(error)}
    pred_to_true, pred_indices = cKDTree(truth).query(points)
    true_to_pred, true_indices = cKDTree(points).query(truth)
    cd = float((np.mean(pred_to_true**2) + np.mean(true_to_pred**2)) / diagonal**2)
    precision, recall = float(np.mean(pred_to_true <= .01 * diagonal)), float(np.mean(true_to_pred <= .01 * diagonal))
    fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    normal = float(np.mean(np.r_[np.abs((normals * truth_normals[pred_indices]).sum(1)),
                                      np.abs((truth_normals * normals[true_indices]).sum(1))]))
    vertices, faces = np.asarray(oracle.vertices), np.asarray(oracle.triangles)
    scores = laplacian_scores(vertices, faces)
    selected = vertices[scores >= np.percentile(scores, edge_percentile)]
    edge = float(cKDTree(points).query(selected)[0].mean() / diagonal)
    return dict(valid=True, status="ok", cd_n=cd, fscore_1pct=fscore,
                normal_consistency=normal, edge_distance=edge, drift_failure=cd > drift_threshold)


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Mesh manifest path escapes the decode directory")
    return path


def evaluate_mesh_directory(decode_dir, output, *, sample_points=1024, seed=20260518,
                            edge_percentile=80, drift_threshold=.005, expected_samples=None):
    root = Path(decode_dir).resolve()
    manifest_path = root / "decoded_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "orbit_folding.decoded_set.v1":
            raise ValueError("Unknown decoded manifest format")
        samples, modes = manifest["samples"], manifest["modes"]
        provenance = {"manifest_sha256": file_sha256(manifest_path)}
    else:
        modes = [key for key, folder in LEGACY_MODES.items() if (root / folder).is_dir()]
        names = sorted({p.name for folder in ["hard_index_oracle", *[LEGACY_MODES[key] for key in modes]]
                        for p in (root / folder).glob("sample_*.obj")})
        samples = [{"id": name, "oracle": f"hard_index_oracle/{name}",
                    **{key: f"{LEGACY_MODES[key]}/{name}" for key in modes}} for name in names]
        provenance = {"source": "legacy mesh directory; no checkpoint/cache binding supplied"}
    if not samples or not modes or len({s["id"] for s in samples}) != len(samples) or len(set(modes)) != len(modes):
        raise ValueError("Benchmark needs unique sample identities and nonempty unique modes")
    if expected_samples is not None and len(samples) != expected_samples:
        raise ValueError("Decoded sample count differs from the requested cohort")
    rows, files = [], []
    for index, sample in enumerate(samples):
        paths = {key: _inside(root, sample[key]) for key in ["oracle", *modes]}
        oracle = read_mesh(paths["oracle"])
        files.append({"id": sample["id"], **{key: file_sha256(path) if path.is_file() else None for key, path in paths.items()}})
        for mode in modes:
            row = compare_meshes(oracle, read_mesh(paths[mode]), sample_points=sample_points,
                seed=seed + index * 97, edge_percentile=edge_percentile, drift_threshold=drift_threshold)
            rows.append({"sample_id": sample["id"], "mode": mode, **row})
    results = {}
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode]
        valid = sum(r["valid"] for r in selected)
        failures = sum(r["drift_failure"] is True for r in selected)
        results[mode] = {"samples": len(selected), "valid": valid, "valid_fraction": valid / len(selected),
            "drift_failures": failures, "drift_valid_fraction": failures / valid if valid else None,
            "failure_inclusive_fraction": (failures + len(selected) - valid) / len(selected),
            **{key: summary(r[key] for r in selected) for key in ("cd_n", "fscore_1pct", "normal_consistency", "edge_distance")}}
    if "relaxed" in results:
        base = results["relaxed"]
        for row in results.values():
            for key, value in (("drift", row["drift_valid_fraction"]), ("edge", row["edge_distance"]["mean"])):
                denominator = base["drift_valid_fraction"] if key == "drift" else base["edge_distance"]["mean"]
                row["excess_" + key] = value / denominator - 1 if value is not None and denominator and denominator > 0 else None
    report = {"format": "orbit_folding.mesh_benchmark.v1", "samples": len(samples), "summary": results,
        "sample_points": sample_points, "seed": seed, "edge_percentile": edge_percentile,
        "drift_threshold": drift_threshold, "provenance": provenance, "mesh_hashes": files, "rows": rows,
        "definitions": {"cd_n": "Sum of both mean squared nearest-neighbor distances divided by oracle bbox diagonal squared; NO factor 0.5.",
            "drift_valid_fraction": "Fraction of valid pairs with CD_n > threshold; missing pairs excluded and separately counted.",
            "failure_inclusive_fraction": "Missing/invalid pairs plus drift failures divided by ALL requested pairs.",
            "edge_distance": "Mean distance from oracle vertices at/above the configured Laplacian percentile to sampled prediction, divided by oracle bbox diagonal.",
            "sampling": "Original Open3D uniform surface sampler, now explicitly seeded per pair. The original run did not seed Open3D's global stream; small differences are expected."}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "geometry.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "geometry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report
