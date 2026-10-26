# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Observed-cloud reconstruction protocol used by the paper fidelity table."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance

from .checkpoints import load_model
from .data import geometry_dataset
from .evaluation import _decode, _grid, _limit, _positive, _write_report
from .geometry import representative_frame
from .mesh_benchmark import sample_surface
from .meshing import MODEL_BOUNDS
from .metrics import extract_mesh, summary
from .provenance import dataset_fingerprint, file_sha256
from .token_support import summarize_counts


def surface_distances(predicted, reference, emd_points=512):
    """Paper CD and Hungarian EMD, with exact KD-tree nearest neighbours."""
    a = cKDTree(reference).query(predicted)[0]
    b = cKDTree(predicted).query(reference)[0]
    cd = .5 * (np.square(a).mean() + np.square(b).mean()) * 1e3
    count = min(emd_points, len(predicted), len(reference))
    costs = distance.cdist(predicted[:count], reference[:count])
    rows, columns = linear_sum_assignment(costs)
    return {"cd_x1e3": float(cd), "emd_x100": float(costs[rows, columns].mean() * 100)}


@torch.random.fork_rng(devices=[])
@torch.no_grad()
def evaluate_fidelity(checkpoint, data_root, output, *, split="test", device="cpu",
                      max_samples=None, metric_points=2048, resolution=64,
                      emd_points=512, seed=20260513, sampling_protocol="reference",
                      encoder_points=None, query_points=None, uniform_points=None):
    import open3d as o3d

    metric_points = _positive(metric_points, "metric_points", 3)
    emd_points = _positive(emd_points, "emd_points")
    model, config = load_model(checkpoint, device)
    if config["Task"] != "tokenizer":
        raise ValueError("Fidelity evaluation requires a tokenizer checkpoint")
    config = dict(config)
    for key, value in (("PCsize", encoder_points), ("SampPerMesh", query_points),
                       ("UniformFieldSamples", uniform_points)):
        if value is not None:
            config[key] = _positive(value, key)
    queries = geometry_dataset(data_root, config, split, seed)
    observations = geometry_dataset(data_root, config, split, seed, points_only=True,
                                    sampling_protocol=sampling_protocol)
    grid = _grid(resolution, MODEL_BOUNDS, device)
    generator = torch.Generator().manual_seed(seed)
    histogram = np.zeros(model.tokenizer.codebook_size, dtype=np.int64)
    records = []
    branches = ("raw", "continuous", "quantized")
    for index in range(_limit(len(queries), max_samples)):
        item = queries.items[index]
        row = {"sample_id": item["id"], "category": item.get("category", "unknown"), "paths": {}}
        try:
            batch = queries[index]
            cloud = observations[index]["points"]
            if len(cloud) > metric_points:
                cloud = cloud[torch.randperm(len(cloud), generator=generator)[:metric_points]]
            points = cloud[None].to(device)
            frame = representative_frame(points, lock=bool(config.get("EnableAxisRepresentativeLock", True)))
            encoded = model.encode(points, frame=frame)
            reference = (points @ frame)[0].cpu().numpy()
            histogram += np.bincount(encoded["indices"].cpu().numpy().ravel(), minlength=len(histogram))
            xyz = batch["query_points"][None].to(device) @ frame
            target = batch["query_sdf"].numpy()
            row["observed_points"] = len(cloud)
            for branch in branches:
                planes = encoded[f"{branch}_planes"]
                query_field = _decode(model, planes, xyz, 32768)
                if not np.isfinite(query_field).all():
                    raise FloatingPointError("Nonfinite fidelity query field")
                target_inside, inside = target < 0, query_field < 0
                values = {"sign_accuracy": float((inside == target_inside).mean()),
                    "inside_gap_pp": float(abs(inside.mean() - target_inside.mean()) * 100),
                    "target_inside_ratio": float(target_inside.mean()), "inside_ratio": float(inside.mean()),
                    "query_count": len(target), "cd_x1e3": None, "emd_x100": None}
                field = _decode(model, planes, grid, 32768).reshape((int(resolution),) * 3)
                status, vertices, faces = extract_mesh(field, MODEL_BOUNDS)
                values["mesh"] = status
                if status["valid"]:
                    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                                    o3d.utility.Vector3iVector(faces))
                    mesh.compute_vertex_normals()
                    predicted, _ = sample_surface(mesh, metric_points, seed + index * 97)
                    values.update(surface_distances(predicted, reference, emd_points))
                row["paths"][branch] = values
        except (ValueError, RuntimeError, OSError, KeyError, FloatingPointError) as error:
            row["error"] = str(error)
        records.append(row)
        if (index + 1) % 32 == 0:
            print(f"Fidelity evaluated {index + 1}/{_limit(len(queries), max_samples)} observations", flush=True)
    by_branch = {}
    for branch in branches:
        values = [row["paths"].get(branch, {}) for row in records]
        valid = sum(row.get("mesh", {}).get("valid", False) for row in values)
        by_branch[branch] = {"samples": len(values), "valid": valid,
            "valid_fraction": valid / len(values),
            **{metric: summary(row.get(metric) for row in values)
               for metric in ("cd_x1e3", "emd_x100", "sign_accuracy", "inside_gap_pp")}}
    report = {"format": "orbit_folding.fidelity_benchmark.v1", "samples": len(records), "records": records,
        "attempted_samples": len(records), "failed_samples": [row["sample_id"] for row in records if "error" in row],
        "summary": by_branch,
        "codebook": ({key: value for key, value in summarize_counts(histogram).items() if key != "cumulative_mass"}
                     if histogram.sum() else None),
        "checkpoint_sha256": file_sha256(checkpoint), "data_fingerprint": dataset_fingerprint(data_root),
        "protocol": {"bounds": list(MODEL_BOUNDS), "resolution": int(resolution), "seed": seed,
            "metric_points": metric_points, "emd_points": emd_points, "sampling_protocol": sampling_protocol,
            "encoder_draw_points": observations.pc_size, "query_draw_points": queries.samples_per_mesh,
            "uniform_draw_points": queries.uniform_samples, "frame_source": "subsampled_observation",
            "reference_surface": "same_subsampled_observation", "surface_sampling": "Open3D, seeded per object",
            "point_sampling": observations.sampling_settings(), "query_sampling": queries.sampling_settings()},
        "definitions": {"cd_x1e3": "Half-sum of mean squared nearest-neighbour distances, times 1000, model units.",
            "emd_x100": "Mean Hungarian assignment distance on the first emd_points points, times 100.",
            "sign_accuracy": "All sampled query signs; exact-zero target is outside, matching the paper evaluator.",
            "inside_gap_pp": "Absolute predicted/target inside-fraction difference per observation, in percentage points."},
        "scope": "Paper observed-cloud fidelity contract, with explicit deterministic draws; not a replay of the unrecorded original worker/global RNG stream. No visualization smoothing or component filtering."}
    return _write_report(output, "fidelity", report)
