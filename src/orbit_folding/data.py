"""Geometric Orbit Representatives for Unaligned 3D Tokenization.

Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig.
Contact: Xiang Chen <xiang.chen@uni-leipzig.de>. SPDX-License-Identifier: MIT.

Explicit sample splits and deterministic, epoch-dependent field sampling.
Stored points and distances already share model-space units.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _indices(count, requested, rng):
    if count <= 0 and requested > 0:
        raise ValueError("Cannot sample an empty array")
    return rng.choice(count, requested, replace=count < requested)


def spatial_indices(points, candidates, requested, rng, bins=24):
    """Stratify occupied spatial cells without a Python loop over points/cells."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if not requested:
        return np.empty(0, dtype=np.int64)
    if len(candidates) <= requested:
        return candidates[_indices(len(candidates), requested, rng)]
    xyz = np.asarray(points[candidates], dtype=np.float32)
    lower, upper = xyz.min(0), xyz.max(0)
    cells = np.floor((xyz - lower) / np.maximum(upper - lower, 1e-6) * bins).astype(int)
    cells = np.clip(cells, 0, bins - 1)
    keys = cells[:, 0] + bins * (cells[:, 1] + bins * cells[:, 2])
    _, groups, counts = np.unique(keys, return_inverse=True, return_counts=True)
    group_order = rng.permutation(len(counts))
    order = np.lexsort((rng.random(len(candidates)), group_order[groups]))
    ordered_groups = group_order[groups[order]]
    starts = np.flatnonzero(np.r_[True, ordered_groups[1:] != ordered_groups[:-1]])
    lengths = np.diff(np.r_[starts, len(order)])
    local_rank = np.arange(len(order)) - np.repeat(starts, lengths)
    chosen = order[local_rank < max(1, requested // len(counts))][:requested]
    if len(chosen) < requested:
        remaining = np.ones(len(candidates), dtype=bool)
        remaining[chosen] = False
        chosen = np.r_[chosen, rng.choice(np.flatnonzero(remaining), requested - len(chosen), replace=False)]
    return rng.permutation(candidates[chosen])


def reference_spatial_indices(points, candidates, requested, rng, bins=24):
    """Original per-cell selection order with grouped, rather than repeated, scans.

    This compatibility path preserves RandomState draws for a given input and
    seed. It does not recover the original exporter's unrecorded worker RNG.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if requested <= 0 or not len(candidates):
        return np.empty(0, dtype=np.int64)
    if len(candidates) <= requested:
        return candidates.copy()
    xyz = np.asarray(points[candidates], dtype=np.float32)
    cells = np.floor((xyz - xyz.min(0)) / np.maximum(np.ptp(xyz, axis=0), 1e-6) * bins).astype(np.int64)
    cells = np.clip(cells, 0, bins - 1)
    keys = cells[:, 0] + bins * (cells[:, 1] + bins * cells[:, 2])
    _, groups, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(groups, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)]
    visit = rng.permutation(len(counts))
    per_cell = max(1, requested // len(counts))
    selected, occupied, size = [], np.zeros(len(candidates), dtype=bool), 0
    for group in visit:
        take = min(per_cell, counts[group], requested - size)
        if take <= 0:
            break
        local = order[offsets[group]:offsets[group + 1]]
        chosen = rng.choice(local, take, replace=False)
        selected.append(candidates[chosen])
        occupied[chosen] = True
        size += take
    if size < requested:
        remaining = candidates[~occupied]
        extra = rng.choice(remaining, min(requested - size, len(remaining)), replace=False)
        selected.append(extra)
        size += len(extra)
    if size < requested:
        selected.append(rng.choice(candidates, requested - size, replace=True))
    result = np.concatenate(selected)[:requested]
    rng.shuffle(result)
    return result


def balanced_indices(sdf, requested, rng, ratios=None, surface_eps=0.0005, near_eps=0.02,
                     points=None, spatial_bins=None):
    values = np.asarray(sdf).reshape(-1)
    groups = [np.abs(values) <= surface_eps,
              (values < -surface_eps) & (values >= -near_eps),
              (values > surface_eps) & (values <= near_eps),
              values < -near_eps, values > near_eps]
    weights = np.asarray(ratios or [0.1, 0.2, 0.2, 0.25, 0.25], dtype=float)
    if len(weights) != 5 or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Five nonnegative SDF-band ratios with positive sum are required")
    quotas = np.floor(weights / weights.sum() * requested).astype(int)
    quotas[np.argmax(weights)] += requested - int(quotas.sum())
    selections = []
    for group, count in zip(groups, quotas):
        candidates = np.flatnonzero(group)
        if count:
            if not len(candidates):
                candidates = np.arange(len(values))
            selections.append(spatial_indices(points, candidates, count, rng, spatial_bins)
                              if spatial_bins is not None else
                              rng.choice(candidates, count, replace=len(candidates) < count))
    indices = np.concatenate(selections) if selections else np.empty(0, dtype=np.int64)
    return rng.permutation(indices)


class GeometryDataset(Dataset):
    def __init__(self, root, split="train", pc_size=8192, samples_per_mesh=8192,
                 uniform_samples=2048, seed=2026, query_ratios=None,
                 surface_eps=0.0005, near_eps=0.02, point_strategy="random",
                 point_bins=28, point_random_ratio=0.25,
                 query_strategy="sdf_value_balanced", query_bins=24,
                 orient_normals=False, axis_lock=None, points_only=False,
                 sampling_protocol="portable"):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("schema") != "orbit-geometry-v1":
            raise ValueError("Unsupported dataset schema; run orbit preprocess or orbit import-data")
        if axis_lock is not None and type(axis_lock) is not bool:
            raise ValueError("axis_lock must be an explicit boolean or None for stored frames")
        policy = manifest.get("frame_policy", {})
        if policy and (policy.get("method") != "pca" or type(policy.get("axis_lock")) is not bool):
            raise ValueError("Unsupported cached frame policy")
        self.axis_lock, self.cached_axis_lock = axis_lock, policy.get("axis_lock")
        self.points_only = bool(points_only)
        if sampling_protocol not in {"portable", "reference"}:
            raise ValueError("Sampling protocol must be portable or reference")
        if sampling_protocol == "reference" and not self.points_only:
            raise ValueError("Reference sampling is an encoder-only cache compatibility option")
        self.sampling_protocol = sampling_protocol
        ids, paths = set(), set()
        for item in manifest["samples"]:
            if item["id"] in ids or item["path"] in paths:
                raise ValueError("Duplicate sample identity/path would leak data between splits")
            if item["split"] not in {"train", "val", "test"}:
                raise ValueError("Every sample must have an explicit train, val, or test split")
            ids.add(item["id"])
            paths.add(item["path"])
        self.items = [item for item in manifest["samples"] if split == "all" or item["split"] == split]
        if not self.items:
            raise ValueError(f"No samples in split {split!r} at {root}")
        self.pc_size, self.samples_per_mesh = int(pc_size), int(samples_per_mesh)
        self.uniform_samples, self.seed = int(uniform_samples), int(seed)
        self.query_ratios = query_ratios
        self.surface_eps, self.near_eps = surface_eps, near_eps
        self.epoch = 0
        self.point_strategy, self.point_bins = point_strategy, int(point_bins)
        self.point_random_ratio, self.query_strategy = point_random_ratio, query_strategy
        self.query_bins, self.orient_normals = int(query_bins), orient_normals
        if point_strategy not in {"random", "spatial_mixed", "spatial_balanced"}:
            raise ValueError(f"Unsupported point sampling strategy: {point_strategy}")
        if query_strategy not in {"sdf_value_balanced", "spatial_sdf_value_balanced"}:
            raise ValueError(f"Unsupported query sampling strategy: {query_strategy}")
        if self.pc_size <= 0 or self.samples_per_mesh <= 0 or self.uniform_samples <= 0:
            raise ValueError("Positive point, query and uniform-field sample counts are required")
        if not 0 <= point_random_ratio <= 1 or min(self.point_bins, self.query_bins) <= 0:
            raise ValueError("Invalid spatial sampling parameters")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.items)

    def sampling_settings(self):
        return {**{key: getattr(self, key) for key in (
                    "seed", "pc_size", "samples_per_mesh", "uniform_samples", "point_strategy",
                    "point_bins", "point_random_ratio", "query_strategy", "query_bins",
                    "query_ratios", "surface_eps", "near_eps", "orient_normals")},
                "point_sampling_frame": "observation", "sampling_protocol": self.sampling_protocol,
                "axis_lock": self.axis_lock if self.axis_lock is not None else self.cached_axis_lock,
                "frame_source": ("full_observation" if self.axis_lock is not None
                                 and self.axis_lock != self.cached_axis_lock else "cache")}

    def __getitem__(self, index):
        item = self.items[index]
        path = (self.root / item["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Sample path escapes the dataset directory")
        seed = self.seed + self.epoch * 1000003 + index * 1009
        rng = (np.random.RandomState(seed % (2**32)) if self.sampling_protocol == "reference"
               else np.random.default_rng(seed))
        with np.load(path, allow_pickle=False) as record:
            names = (["points", "frame"] if self.points_only else
                     ["points", "normals", "query_points", "query_sdf", "uniform_points", "uniform_sdf", "frame"])
            stored = {key: np.asarray(record[key], dtype=np.float32)
                      for key in names}
        for key, array in stored.items():
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite {key} array: {item['id']}")
        for key in (["points"] if self.points_only else ["points", "normals", "query_points", "uniform_points"]):
            if stored[key].ndim != 2 or stored[key].shape[1] != 3 or not len(stored[key]):
                raise ValueError(f"{key} must be a nonempty N x 3 array")
        if not self.points_only and stored["normals"].shape != stored["points"].shape:
            raise ValueError("Surface point and normal dimensions differ")
        for key, coordinates in ([] if self.points_only else [("query_sdf", "query_points"), ("uniform_sdf", "uniform_points")]):
            if stored[key].shape not in {(len(stored[coordinates]),), (len(stored[coordinates]), 1)}:
                raise ValueError(f"{key} and {coordinates} dimensions differ")
            stored[key] = stored[key].reshape(-1)
        frame = stored["frame"]
        if frame.shape != (3, 3) or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-3) or np.linalg.det(frame) < 0:
            raise ValueError("frame must be a proper 3 x 3 rotation")
        points = stored["points"]
        if self.axis_lock is not None and self.axis_lock != self.cached_axis_lock:
            from .geometry import representative_frame

            frame = representative_frame(torch.from_numpy(points), lock=self.axis_lock).numpy()
        if self.sampling_protocol == "reference" and len(points) <= self.pc_size:
            selected = (_indices(len(points), self.pc_size, rng) if len(points) < self.pc_size
                        else np.arange(len(points)))
        elif self.point_strategy == "random":
            selected = _indices(len(points), self.pc_size, rng)
        else:
            random_count = round(self.pc_size * self.point_random_ratio) if self.point_strategy == "spatial_mixed" else 0
            sample = reference_spatial_indices if self.sampling_protocol == "reference" else spatial_indices
            spatial_selected = sample(points, np.arange(len(points)), self.pc_size - random_count,
                                      rng, self.point_bins)
            random_indices = (_indices(len(points), random_count, rng)
                              if random_count or self.sampling_protocol == "portable"
                              else np.empty(0, dtype=np.int64))
            selected = np.r_[spatial_selected, random_indices]
            if self.sampling_protocol == "portable" or self.point_strategy == "spatial_mixed":
                selected = rng.permutation(selected)
        if self.points_only:
            return {"points": torch.from_numpy(points[selected].copy()),
                    "frame": torch.from_numpy(frame.copy()), "sample_id": item["id"],
                    "split": item["split"], "category": item.get("category", "unknown")}
        q = balanced_indices(stored["query_sdf"], self.samples_per_mesh, rng,
                             self.query_ratios, self.surface_eps, self.near_eps,
                             stored["query_points"],
                             self.query_bins if self.query_strategy == "spatial_sdf_value_balanced" else None)
        u = _indices(len(stored["uniform_points"]), self.uniform_samples, rng)
        normals = stored["normals"]
        valid = np.linalg.norm(normals, axis=1) > 1e-6
        if self.orient_normals and valid.any():
            score = np.mean(np.sum(normals[valid] * (points - points.mean(0))[valid], axis=1))
            if score < 0:
                normals = -normals
        arrays = {
            "points": points[selected],
            "normals": normals[selected],
            "frame": frame,
            "query_points": stored["query_points"][q],
            "query_sdf": stored["query_sdf"][q],
            "uniform_points": stored["uniform_points"][u],
            "uniform_sdf": stored["uniform_sdf"][u],
        }
        result = {key: torch.from_numpy(np.asarray(value, dtype=np.float32).copy())
                  for key, value in arrays.items()}
        result.update(sample_id=item["id"], split=item["split"], category=item.get("category", "unknown"))
        return result


def geometry_dataset(root, config, split="train", seed=2026, **overrides):
    """One configured geometry contract for training, caching and evaluation."""
    ratios = config.get("QuerySamplingRatios", {})
    settings = dict(
        pc_size=config.get("PCsize", 8192), samples_per_mesh=config.get("SampPerMesh", 8192),
        uniform_samples=config.get("UniformFieldSamples", 2048),
        query_ratios=[ratios.get(key, value) for key, value in zip(
            ["surface", "near_inside", "near_outside", "far_inside", "far_outside"],
            [0.1, 0.2, 0.2, 0.25, 0.25])],
        surface_eps=config.get("QuerySamplingBins", {}).get("surface_eps", 0.0005),
        near_eps=config.get("QuerySamplingBins", {}).get("near_eps", 0.02),
        point_strategy=config.get("PointSamplingStrategy", "random"),
        point_bins=config.get("PointSpatialBins", 28), point_random_ratio=config.get("PointRandomRatio", 0.25),
        query_strategy=config.get("QuerySamplingStrategy", "sdf_value_balanced"),
        query_bins=config.get("QuerySpatialBins", 24),
        orient_normals=config.get("AutoOrientNormalsByCentroid", False),
        axis_lock=bool(config.get("EnableAxisRepresentativeLock", True)))
    settings.update(overrides)
    return GeometryDataset(root, split=split, seed=seed, **settings)
