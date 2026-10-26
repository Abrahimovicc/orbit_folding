# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Point-only rotation/noise stress on ShapeNet, Objaverse and ScanObjectNN."""

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from .evaluation import _rotation_pair, load_observation, observation_entries
from .geometry import representative_frame
from .metrics import summary
from .provenance import file_sha256


def observations(data, kind, *, split="test", split_file=None, limit=None, points=2048, seed=20260704,
                 categories=None, max_per_category=None):
    """Yield one fixed observation per identity; never substitute another asset."""
    root = Path(data)
    if kind == "scanobjectnn":
        try:
            import h5py
        except ImportError as error:
            raise ImportError('ScanObjectNN requires pip install -e ".[benchmark]"') from error
        with h5py.File(root, "r") as record:
            indices = list(range(len(record["data"])))
            random.Random(seed).shuffle(indices)
            for index in indices[:limit]:
                yield f"{root.name}:{index}", np.asarray(record["data"][index], dtype=np.float64)
    elif kind == "shapenet":
        if split_file is None:
            raise ValueError("ShapeNet source NPZs require explicit split identifiers")
        ids = json.loads(Path(split_file).read_text())[split]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate ShapeNet identities")
        selected, counts = [], {}
        for identity in ids:
            category = identity.split("/")[0]
            if categories is not None and category not in categories:
                continue
            if max_per_category is not None and counts.get(category, 0) >= max_per_category:
                continue
            counts[category] = counts.get(category, 0) + 1
            selected.append(identity)
        for identity in selected[:limit]:
            path = (root / identity / "data.npz").resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("ShapeNet identity escapes dataset root")
            with np.load(path, allow_pickle=False) as sample:
                yield identity, np.asarray(sample["encoder_points"], dtype=np.float64)
    else:
        if kind == "meshes":
            paths = sorted(path for path in root.rglob("*.obj") if not path.name.endswith("_water.obj"))
            random.Random(seed).shuffle(paths)
            entries = [{"id": str(p.relative_to(root)), "path": str(p.relative_to(root)), "processed": False}
                       for p in paths]
        elif kind == "observations":
            entries = observation_entries(root, split)
        else:
            raise ValueError("Unknown benchmark dataset format")
        for index, entry in enumerate(entries[:limit]):
            try:
                yield entry["id"], load_observation(root, entry, points, seed + index).numpy()
            except (ValueError, RuntimeError, OSError) as error:
                yield entry["id"], error


@torch.no_grad()
def evaluate_stress(data, output, *, kind="observations", split="test", split_file=None,
                    limit=None, expected_samples=None, points=2048, seed=20260704,
                    jitter_sigma_unit_radius=.005, tail_threshold=16.18, device="cpu",
                    categories=None, max_per_category=None):
    if (points < 8 or int(points) != points or (limit is not None and limit < 1)
            or not np.isfinite(jitter_sigma_unit_radius) or jitter_sigma_unit_radius < 0
            or not np.isfinite(tail_threshold) or tail_threshold < 0):
        raise ValueError("Invalid point budget, limit, jitter or tail threshold")
    rows, failures, identities = [], [], []
    for index, (identity, cloud) in enumerate(observations(data, kind, split=split, split_file=split_file,
        limit=limit, points=points, seed=seed, categories=categories, max_per_category=max_per_category)):
        identities.append(identity)
        try:
            if isinstance(cloud, Exception):
                raise ValueError(str(cloud))
            if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 8 or not np.isfinite(cloud).all():
                raise ValueError("Observation must be a finite N x 3 cloud with at least eight points")
            rng = np.random.default_rng(seed + index)
            cloud = cloud[rng.choice(len(cloud), points, replace=len(cloud) < points)]
            cloud -= cloud.mean(0)
            radius = np.linalg.norm(cloud, axis=1).max()
            if radius <= 1e-12:
                raise ValueError("Degenerate observation")
            reference = torch.from_numpy(cloud / radius * .5)
            for condition, sigma in (("base_observation", 0), ("jitter", jitter_sigma_unit_radius / 2)):
                clean, observed, _ = _rotation_pair(reference, sigma, seed + index * 100003)
                clean, observed = clean.to(device), observed.to(device)
                row = {"sample_id": identity, "condition": condition}
                for lock, name in ((False, "pca"), (True, "orbit")):
                    frame = representative_frame(observed, lock)
                    first, second = (clean @ frame).cpu().numpy()
                    distances = (cKDTree(first).query(second)[0], cKDTree(second).query(first)[0])
                    row[name + "_pae"] = float(.5 * sum(np.mean(d**2) for d in distances) * 1e4)
                rows.append(row)
        except (ValueError, RuntimeError, OSError) as error:
            failures.append({"sample_id": identity, "error": str(error)})
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("Benchmark needs nonempty unique sample identities")
    if expected_samples is not None and len(identities) != expected_samples:
        raise ValueError("Source cohort size differs from expected_samples")
    result = {}
    for condition in ("base_observation", "jitter"):
        selected = [r for r in rows if r["condition"] == condition]
        values = {name: summary(r[name + "_pae"] for r in selected) for name in ("orbit", "pca")}
        ratio = values["pca"]["p95"] / values["orbit"]["p95"] if values["orbit"]["p95"] else None
        result[condition] = {**values, "p95_ratio": ratio, "successful_samples": len(selected),
            **{name + "_tail_fraction": sum(r[name + "_pae"] > tail_threshold for r in selected) / len(selected)
               if selected else None for name in ("orbit", "pca")}}
    report = {"format": "orbit_folding.stress_benchmark.v1", "dataset_format": kind,
        "requested_samples": len(identities), "failed_samples": failures, "summary": result, "rows": rows,
        "protocol": {"points": points, "seed": seed, "normalization": "centroid, radius 0.5",
            "categories": categories, "max_per_category": max_per_category,
            "jitter_sigma_unit_radius": jitter_sigma_unit_radius, "jitter_sigma_model": jitter_sigma_unit_radius / 2,
            "jitter_clip_unit_radius": 3 * jitter_sigma_unit_radius, "tail_threshold": tail_threshold,
            "pae": "0.5 * (both mean squared NN distances), times 1e4; same unperturbed reference for both frames.",
            "frame_precision": "float64", "nn_engine": "scipy.cKDTree float64",
            "selection_sha256": hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
            "split_sha256": file_sha256(split_file) if split_file else None,
            "source_sha256": file_sha256(data) if Path(data).is_file() else None},
        "scope": "Fixed observed point set under two rotations and independent bounded jitter. Frame stability only; no semantic labels or scan SDF claim. Seeded protocol; float64 NN precision differs from the original float32 CUDA floor."}
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "stability.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (root / "stability.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "condition", "pca_pae", "orbit_pae"])
        writer.writeheader()
        writer.writerows(rows)
    return report
