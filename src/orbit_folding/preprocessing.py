"""Geometric Orbit Representatives for Unaligned 3D Tokenization.

Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig.
Contact: Xiang Chen <xiang.chen@uni-leipzig.de>. SPDX-License-Identifier: MIT.

Open3D mesh validation, surface sampling and signed-distance supervision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import write_json
from .geometry import normalization_parameters
from .provenance import file_sha256


def _frame(points, axis_lock=True):
    from .geometry import representative_frame
    value = representative_frame(torch.from_numpy(points).unsqueeze(0), lock=axis_lock)
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, dict):
        value = value.get("frame", value.get("rotation"))
    return value.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _orient_components(mesh, o3d):
    """Orient each connected shell against material occupancy, including cavities."""
    if not mesh.orient_triangles():
        raise ValueError("Cannot consistently orient mesh triangles")
    labels = np.asarray(mesh.cluster_connected_triangles()[0], dtype=np.int64)
    faces = np.asarray(mesh.triangles).copy()
    triangles = np.asarray(mesh.vertices)[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(area[:, None], 1e-30)
    centers = triangles.mean(axis=1)
    components = int(labels.max()) + 1
    selected = np.concatenate([indices[np.argsort(area[indices])[-64:]]
        for component in range(components) for indices in [np.flatnonzero(labels == component)]])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    votes = np.zeros(components, dtype=float)
    counts = np.zeros(components, dtype=np.int64)
    for epsilon in (1e-4, 1e-5, 1e-6):
        probes = np.concatenate([centers[selected] - epsilon * normals[selected],
                                 centers[selected] + epsilon * normals[selected]]).astype(np.float32)
        inside = scene.compute_occupancy(o3d.core.Tensor(probes), nsamples=5).numpy().reshape(2, -1)
        direction = inside[0] - inside[1]
        votes += np.bincount(labels[selected], weights=direction, minlength=components)
        counts += np.bincount(labels[selected], weights=(direction != 0), minlength=components).astype(np.int64)
    if np.any(counts == 0) or np.any(np.abs(votes) < 0.8 * counts):
        raise ValueError("Component normals have ambiguous material occupancy; inspect thin or nested surfaces")
    reverse = votes[labels] < 0
    faces[reverse] = faces[reverse][:, [0, 2, 1]]
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return components


def _validate_splits(splits):
    if not isinstance(splits, dict) or not splits or set(splits) - {"train", "val", "test"}:
        raise ValueError("Split JSON must use train, val and test lists")
    if any(not isinstance(entries, list) or not all(isinstance(value, str) for value in entries)
           for entries in splits.values()):
        raise ValueError("Every split must be a list of sample identifiers")


def _check_source(path, split, seen_paths, seen_hashes):
    source = path.resolve()
    if source in seen_paths:
        raise ValueError(f"Split leakage or duplicate source: {path.name}")
    seen_paths.add(source)
    checksum = file_sha256(source)
    if checksum in seen_hashes and seen_hashes[checksum] != split:
        raise ValueError(f"Identical source content occurs across splits: {path.name}")
    seen_hashes[checksum] = split
    return checksum


def preprocess(input_dir, output, surface_points=8192, query_points=32768,
               uniform_points=8192, seed=2026, split_file=None, *,
               normalization="centroid", axis_lock=True):
    import open3d as o3d
    root, output = Path(input_dir), Path(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output must be outside the mesh input directory")
    if min(surface_points, query_points, uniform_points) <= 0:
        raise ValueError("Surface, query and uniform sample counts must be positive")
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".ply", ".obj", ".stl", ".off"})
    if not paths:
        raise ValueError(f"No mesh files in {root}")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Mesh stems must be unique; rename duplicates before preprocessing")
    split_map = {}
    if split_file:
        raw = json.loads(Path(split_file).read_text())
        _validate_splits(raw)
        for split, ids in raw.items():
            for sample_id in ids:
                if sample_id in split_map:
                    raise ValueError(f"Sample occurs in more than one split: {sample_id}")
                split_map[sample_id] = split
        if set(split_map) != {path.stem for path in paths}:
            raise ValueError("Split file must cover every input mesh exactly once")
    else:
        if len(paths) < 3:
            raise ValueError("Provide at least three meshes or an explicit split JSON")
        split_map = {p.stem: "train" for p in paths}
        split_map[paths[-2].stem], split_map[paths[-1].stem] = "val", "test"
    seen_paths, seen_hashes = set(), {}
    source_hashes = {path: _check_source(path, split_map[path.stem], seen_paths, seen_hashes) for path in paths}
    samples = []
    attribution = {}
    for candidate in (root / "manifest.json", root.parent / "manifest.json", root.parent.parent / "manifest.json"):
        if candidate.is_file():
            metadata = json.loads(candidate.read_text())
            for entry in metadata.get("assets", []):
                source = (candidate.parent / entry["path"]).resolve()
                attribution[source] = entry
    output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths):
        rng = np.random.default_rng(seed + index)
        o3d.utility.random.seed(seed + index)
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        mesh.remove_duplicated_vertices().remove_duplicated_triangles()
        mesh.remove_degenerate_triangles().remove_unreferenced_vertices()
        if not mesh.has_triangles() or not mesh.is_watertight():
            raise ValueError(f"{path.name} is not watertight; signed SDF targets are not valid. Use orbit evaluate stability for raw point observations instead.")
        vertices = np.asarray(mesh.vertices)
        center, scale = normalization_parameters(vertices, normalization)
        mesh.vertices = o3d.utility.Vector3dVector((vertices - center) / scale)
        components = _orient_components(mesh, o3d)
        mesh.compute_vertex_normals()
        cloud = mesh.sample_points_uniformly(number_of_points=surface_points, use_triangle_normal=True)
        points = np.asarray(cloud.points, dtype=np.float32)
        normals = np.asarray(cloud.normals, dtype=np.float32)
        near_count = query_points * 3 // 4
        selected = rng.integers(0, len(points), near_count)
        sigma = rng.choice([0.002, 0.01, 0.04], size=(near_count, 1))
        near = points[selected] + rng.normal(size=(near_count, 3)) * sigma
        queries = np.concatenate([near, rng.uniform(-0.6, 0.6, (query_points - near_count, 3))]).astype(np.float32)
        uniform = rng.uniform(-0.6, 0.6, (uniform_points, 3)).astype(np.float32)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        targets = scene.compute_signed_distance(o3d.core.Tensor(queries)).numpy()
        uniform_sdf = scene.compute_signed_distance(o3d.core.Tensor(uniform)).numpy()
        relative = f"{path.stem}.npz"
        np.savez_compressed(output / relative, points=points, normals=normals,
                            query_points=queries, query_sdf=targets,
                            uniform_points=uniform, uniform_sdf=uniform_sdf,
                            frame=_frame(points, axis_lock), center=center, scale=np.float64(scale))
        o3d.io.write_triangle_mesh(str(output / f"{path.stem}.ply"), mesh, write_ascii=False)
        record = {"id": path.stem, "path": relative, "split": split_map[path.stem],
                        "category": "unknown", "source_file": path.name,
                        "source_sha256": source_hashes[path], "sha256": file_sha256(output / relative),
                        "watertight": True, "components": components,
                        "center": center.tolist(), "scale": float(scale)}
        provenance = attribution.get(path.resolve(), {})
        if provenance and provenance.get("sha256") != record["source_sha256"]:
            raise ValueError(f"Input mesh checksum differs from its license manifest: {path.name}")
        record.update({key: provenance[key] for key in ["source_url", "license_spdx", "authors"] if key in provenance})
        samples.append(record)
    manifest = {"schema": "orbit-geometry-v1", "units": "model coordinates; maximum vertex radius 0.5",
                "sdf_convention": "negative inside; same distance units as points",
                "seed": seed, "frame_policy": {"method": "pca", "axis_lock": bool(axis_lock)},
                "preprocessing": {"version": 3, "normalization": normalization, "surface_points": surface_points,
                    "query_points": query_points, "uniform_points": uniform_points,
                    "normal_orientation": "connected shells checked against material occupancy"},
                "samples": samples}
    write_json(output / "manifest.json", manifest)
    return manifest


def import_preprocessed(input_dir, split_file, output, input_scale=2.0, *, axis_lock=True,
                        legacy_uniform_start=None, frame_caches=None, storage="compressed"):
    """Adapt the research NPZ schema without redistributing any source data."""
    if not np.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if storage not in {"compressed", "uncompressed"}:
        raise ValueError("storage must be compressed or uncompressed")
    root, output = Path(input_dir), Path(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output must be outside the source dataset directory")
    splits = json.loads(Path(split_file).read_text())
    _validate_splits(splits)
    frames = {}
    frame_hashes = []
    for path in frame_caches or []:
        frame_hashes.append(file_sha256(path))
        with np.load(path, allow_pickle=False) as cache:
            identities, rotations = cache["rel_paths"].tolist(), cache["rotations"]
            if len(identities) != len(rotations) or rotations.shape[1:] != (3, 3):
                raise ValueError("Frame cache identity/rotation dimensions differ")
            for identity, rotation in zip(identities, rotations):
                identity = str(identity)
                if (not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                        or np.linalg.det(rotation) < 0):
                    raise ValueError("Frame cache contains an invalid rotation")
                if identity in frames and not np.array_equal(frames[identity], rotation):
                    raise ValueError("Conflicting rotations for the same cached identity")
                frames[identity] = rotation.astype(np.float32)
    samples, seen, seen_paths, seen_hashes, identifiers = [], set(), set(), {}, set()
    sources = []
    for split, entries in splits.items():
        for relative in entries:
            if relative in seen:
                raise ValueError(f"Split leakage: {relative}")
            seen.add(relative)
            source = (root / relative / "data.npz") if not str(relative).endswith(".npz") else root / relative
            source = source.resolve()
            if not source.is_relative_to(root.resolve()):
                raise ValueError("Input path escapes dataset root")
            sample_id = str(relative).replace("/", "__").replace(".npz", "")
            if sample_id in identifiers:
                raise ValueError("Distinct source identifiers collide after filename conversion")
            identifiers.add(sample_id)
            checksum = _check_source(source, split, seen_paths, seen_hashes)
            if frame_caches and relative not in frames:
                raise ValueError(f"Frame cache does not cover sample {relative}")
            sources.append((split, relative, source, sample_id, checksum))
    for split, relative, source, sample_id, checksum in sources:
        with np.load(source, allow_pickle=False) as item:
            point_key = "encoder_points" if "encoder_points" in item else "point_cloud"
            normal_key = "encoder_normals" if "encoder_normals" in item else "normals"
            points = item[point_key].astype(np.float32) / input_scale
            normals = item[normal_key].astype(np.float32)
            queries = item["query_points"].astype(np.float32) / input_scale
            sdf = item["query_sdf"].astype(np.float32).reshape(-1) / input_scale
            if "uniform_points" in item and "uniform_sdf" in item:
                uniform = item["uniform_points"].astype(np.float32) / input_scale
                uniform_sdf = item["uniform_sdf"].astype(np.float32).reshape(-1) / input_scale
            elif "query_segment_names" in item and "query_segment_counts" in item:
                names, counts = item["query_segment_names"].tolist(), item["query_segment_counts"].astype(int)
                offsets = np.r_[0, np.cumsum(counts)]
                bands = [np.arange(offsets[i], offsets[i + 1]) for i, name in enumerate(names) if str(name) == "uniform"]
                if not bands or offsets[-1] != len(queries):
                    raise ValueError("Missing or inconsistent uniform-field segment metadata")
                selected = np.concatenate(bands)
                uniform, uniform_sdf = queries[selected], sdf[selected]
            elif legacy_uniform_start is not None:
                start = int(legacy_uniform_start)
                if start != legacy_uniform_start or not 0 <= start < len(queries):
                    raise ValueError("legacy_uniform_start must select a nonempty query suffix")
                uniform, uniform_sdf = queries[start:], sdf[start:]
            else:
                raise ValueError("Uniform SDF supervision is missing; regenerate it from the validated mesh before importing")
        output.mkdir(parents=True, exist_ok=True)
        save = np.savez_compressed if storage == "compressed" else np.savez
        save(output / f"{sample_id}.npz", points=points, normals=normals,
                            query_points=queries, query_sdf=sdf,
                            uniform_points=uniform, uniform_sdf=uniform_sdf,
                            frame=frames[relative] if frame_caches else _frame(points, axis_lock))
        samples.append({"id": sample_id, "path": f"{sample_id}.npz", "split": split,
                        "source_sha256": checksum, "sha256": file_sha256(output / f"{sample_id}.npz"),
                        "category": str(relative).split("/")[0]})
    manifest = {"schema": "orbit-geometry-v1", "units": "model coordinates", "input_scale": input_scale,
                "frame_policy": {"method": "pca", "axis_lock": bool(axis_lock), "cache_sha256": frame_hashes},
                "preprocessing": {"version": 4, "adapter": "explicit SDF import",
                                  "storage": storage,
                                  "legacy_uniform_start": legacy_uniform_start}, "samples": samples}
    write_json(output / "manifest.json", manifest)
    return manifest
