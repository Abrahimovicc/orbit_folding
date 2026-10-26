# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Independent orientation, field reconstruction, and cached-token evaluations.

Coordinates and SDF are in the stored radius-0.5 model space, without another
InputScaleFactor. These evaluations do not establish raw-scan SDF reliability,
industrial deployment safety, grasp quality, or paper-level model quality.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from .meshing import MODEL_BOUNDS, validate_bounds

from .geometry import normalization_parameters, random_rotations, representative_frame
from .metrics import (aggregate_sdf, chamfer_distance, extract_mesh, grid_drift,
                      hidden_accuracy, point_tensor, sample_mesh_surface,
                      sdf_metrics, summary)


def _options(provided, defaults):
    unknown = set(provided) - set(defaults)
    if unknown:
        raise TypeError(f"Unknown evaluation options: {sorted(unknown)}")
    return {**defaults, **provided}


def _limit(length, maximum):
    if maximum is not None and (int(maximum) != maximum or maximum < 1):
        raise ValueError("max_samples must be a positive integer or None")
    return min(length, int(maximum)) if maximum is not None else length


def _positive(value, name, minimum=1):
    if int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _write_report(output, name, report):
    path = Path(output)
    if path.suffix.lower() != ".json":
        path = path / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report["output"] = str(path)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Sample path escapes the data root")
    return path


def observation_entries(data_root, split="test"):
    """Read processed splits or licensed raw assets, without inventing a holdout.

    Raw assets with no split are all returned, explicitly marked unassigned.
    A raw directory may contain mesh, NPY, or NPZ point observations.
    """
    root = Path(data_root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        processed = manifest.get("schema") == "orbit-geometry-v1"
        entries = manifest.get("samples", []) if processed else manifest.get("assets", [])
        result = []
        for entry in entries:
            if entry.get("split") and split != "all" and entry["split"] != split:
                continue
            result.append({**entry, "processed": processed})
    else:
        extensions = {".ply", ".obj", ".stl", ".off", ".npy", ".npz"}
        result = [{"id": path.stem, "path": str(path.relative_to(root)), "processed": False}
                  for path in sorted(root.iterdir()) if path.suffix.lower() in extensions]
    if not result:
        raise ValueError(f"No observations for split {split!r} at {root}")
    ids = [entry["id"] for entry in result]
    if len(set(ids)) != len(ids):
        raise ValueError("Observation identities must be unique")
    return result


def load_observation(data_root, entry, mesh_points=8192, seed=2026):
    """Load real observations; normalize raw assets only to model radius 0.5.

    Mesh sampling is area weighted and seeded. Stored processed point arrays
    are used in full and are never rescaled or replaced by generated geometry.
    """
    path = _safe_path(data_root, entry["path"])
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as record:
            cloud = np.asarray(record["points"], dtype=np.float64).copy()
    elif path.suffix.lower() == ".npy":
        cloud = np.load(path, allow_pickle=False)
    else:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        if mesh.has_triangles():
            cloud = sample_mesh_surface(np.asarray(mesh.vertices), np.asarray(mesh.triangles), mesh_points, seed)
        else:
            cloud = np.asarray(o3d.io.read_point_cloud(str(path)).points).copy()
    cloud = point_tensor(cloud, min_points=3).double()
    if cloud.ndim != 2:
        raise ValueError("An observation file must contain one (N,3) point cloud")
    if not entry.get("processed", False):
        center, scale = normalization_parameters(cloud.numpy())
        cloud = (cloud - torch.from_numpy(center)) / scale
    if (cloud - cloud.mean(0)).square().sum() <= 1e-20:
        raise ValueError("Degenerate observation has no spatial extent")
    return cloud


def _rotation_pair(reference, jitter, seed):
    # The public rotation sampler uses the global CPU generator. Restore it,
    # and do not seed or initialize CUDA just to evaluate CPU observations.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        rotations = random_rotations(2, device="cpu", dtype=reference.dtype)
        clean = reference.unsqueeze(0) @ rotations.transpose(1, 2)
        noise = (torch.randn_like(clean) * jitter).clamp(-3 * jitter, 3 * jitter) if jitter else 0
        observed = clean + noise
    return clean, observed, rotations


@torch.no_grad()
def eval_stability(data_root, split="test", output="evaluation/stability.json",
                   device="cpu", seed=2026, jitter=0.0, max_samples=None, **kwargs):
    """Compare independent observations using the SAME unperturbed reference.

    Options: rotation_pairs=8, mesh_points=8192, reference_points=1024,
    chunk_size=512. Jitter is per-coordinate Gaussian sigma, clipped at +/-3
    sigma, in model units:
    paper/unit-radius sigma .005 corresponds to model sigma .0025. Frames
    use the full observed cloud, not the metric's reference-point subset.
    """
    options = _options(kwargs, dict(rotation_pairs=8, mesh_points=8192,
                                    reference_points=1024, chunk_size=512))
    for name in options:
        options[name] = _positive(options[name], name)
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be a finite nonnegative model-space sigma")
    entries = observation_entries(data_root, split)
    records, failures = [], []
    for index, entry in enumerate(entries[:_limit(len(entries), max_samples)]):
        try:
            reference = load_observation(data_root, entry, options["mesh_points"], seed + index)
            rng = np.random.default_rng(seed + index)
            chosen = rng.choice(len(reference), min(len(reference), options["reference_points"]), replace=False)
            covariance = torch.cov(reference.T)
            values = torch.linalg.eigvalsh(covariance).flip(0)
            gaps = ((values[:-1] - values[1:]) / values[0].clamp_min(1e-30)).tolist()
            row = {"sample_id": entry["id"], "split": entry.get("split", "unassigned"),
                   "observed_points": len(reference), "reference_points": len(chosen),
                   "normalized_pca_eigengaps": gaps, "near_repeated_eigenvalue": min(gaps) < 0.001,
                   "pca_only_cd_x1e4": [], "locked_cd_x1e4": []}
            for trial in range(options["rotation_pairs"]):
                _, observed, rotations = _rotation_pair(reference, jitter, seed + index * 100003 + trial)
                observed = observed.to(device)
                clean = (reference[chosen].unsqueeze(0) @ rotations.transpose(1, 2)).to(device)
                for lock, key in ((False, "pca_only_cd_x1e4"), (True, "locked_cd_x1e4")):
                    frames = representative_frame(observed, lock=lock)
                    folded = clean @ frames
                    cd = chamfer_distance(folded[0], folded[1], options["chunk_size"])
                    row[key].append(float(cd) * 1e4)
            records.append(row)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            failures.append({"sample_id": entry["id"], "error": str(error)})
    result = {"evaluation": "stability", "seed": int(seed), "requested_split": split,
              "jitter_sigma_model_units": float(jitter), "jitter_sigma_unit_radius": float(jitter) * 2,
              "jitter_clip_bounds_model_units": [-3 * float(jitter), 3 * float(jitter)],
              "jitter_clip_bounds_unit_radius": [-6 * float(jitter), 6 * float(jitter)],
              "options": options, "attempted_samples": len(records) + len(failures),
              "successful_samples": len(records), "failed_samples": failures,
              "pca_only_cd_x1e4": summary(v for row in records for v in row["pca_only_cd_x1e4"]),
              "locked_cd_x1e4": summary(v for row in records for v in row["locked_cd_x1e4"]),
              "samples": records,
              "protocol": "Frames from independent rotations/noise, with per-coordinate Gaussian jitter clipped at +/-3 sigma; each frame acts on the same clean reference rotated into its own observation coordinates. No Chamfer between independently jittered geometries.",
              "chamfer": "0.5 * (mean squared nearest distance A->B + B->A), times 10000; model units squared.",
              "limitations": ["Repeated PCA eigenvalues, near-symmetry and probe-score ties can make orientation nonunique or noise-sensitive. Small set CD does not prove small orientation-angle error.",
                              "Raw assets with no declared split are unassigned illustrations, not a test benchmark. No partial-view, occlusion, translation or scale robustness is measured.",
                              "Orientation repeatability does not imply exact token identity, raw-scan signed-distance reliability, or grasp quality."]}
    return _write_report(output, "stability", result)


def _geometry_dataset(data_root, split, config, options):
    from .data import geometry_dataset

    dataset = geometry_dataset(data_root, config, split=split, seed=options["seed"],
                               pc_size=options["pc_size"] or int(config.get("PCsize", 8192)),
                               samples_per_mesh=options["samples_per_mesh"],
                               uniform_samples=options["uniform_samples"])
    options["sampling"] = dataset.sampling_settings()
    return dataset


def _field_options(kwargs):
    options = _options(kwargs, dict(pc_size=None, samples_per_mesh=2048, uniform_samples=1024,
                                    query_chunk_size=4096, chunk_size=512, mesh_samples=2048,
                                    seed=2026, sign_eps=0.0005, bounds=MODEL_BOUNDS, save_meshes=False))
    for name in ("samples_per_mesh", "uniform_samples", "query_chunk_size", "chunk_size", "mesh_samples"):
        options[name] = _positive(options[name], name)
    if options["pc_size"] is not None:
        options["pc_size"] = _positive(options["pc_size"], "pc_size", 3)
    if not np.isfinite(options["sign_eps"]) or options["sign_eps"] < 0:
        raise ValueError("sign_eps must be finite and nonnegative")
    lo, hi = options["bounds"]
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("bounds must be a finite increasing pair")
    return options


def _grid(resolution, bounds, device):
    resolution = _positive(resolution, "resolution", 2)
    bounds = validate_bounds(bounds)
    axis = torch.linspace(*bounds, resolution, device=device)
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(1, -1, 3)


def _decode(model, planes, queries, chunk_size):
    values = []
    for chunk in queries.split(chunk_size, dim=1):
        prediction = model.decode(planes, chunk)
        if prediction.shape not in (chunk.shape[:2], (*chunk.shape[:2], 1)):
            raise ValueError("decode must return (B,Q) or (B,Q,1) fields")
        values.append(prediction.reshape(chunk.shape[:2]).detach().cpu().numpy())
    return np.concatenate(values, axis=1)[0]


def _cached_frame(value, device):
    frame = torch.as_tensor(value).unsqueeze(0).to(device)
    if frame.shape != (1, 3, 3) or not torch.isfinite(frame).all():
        raise ValueError("Cached frame must be a finite (3,3) matrix")
    identity = torch.eye(3, device=frame.device, dtype=frame.dtype).unsqueeze(0)
    if not torch.allclose(frame.transpose(1, 2) @ frame, identity, atol=1e-4, rtol=1e-4) or torch.linalg.det(frame)[0] <= 0:
        raise ValueError("Cached frame must be an orthonormal proper rotation")
    return frame


def _reference_surface(dataset, index, frame, options):
    entry = dataset.items[index]
    reference = load_observation(dataset.root, {**entry, "processed": True})
    rng = np.random.default_rng(options["seed"] + index)
    selection = rng.choice(len(reference), min(len(reference), options["mesh_samples"]), replace=False)
    return (reference[selection] @ frame.detach().cpu().double()[0]).numpy()


def _mesh_result(field, reference, options, seed, output, filename):
    result, vertices, faces = extract_mesh(field, options["bounds"])
    result["cd_x1e4"] = None
    if result["valid"]:
        sampled = sample_mesh_surface(vertices, faces, options["mesh_samples"], seed)
        result["cd_x1e4"] = float(chamfer_distance(sampled, reference, options["chunk_size"])) * 1e4
        if options["save_meshes"]:
            import open3d as o3d

            root = Path(output).parent if Path(output).suffix.lower() == ".json" else Path(output)
            path = root / "meshes" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
            if not o3d.io.write_triangle_mesh(str(path), mesh):
                raise OSError(f"Mesh export failed: {path}")
            result["path"] = str(path)
    return result


def _evaluate_planes(model, planes, batch, frame, grid, reference, options, seed, output, filename):
    result = {}
    for prefix in ("query", "uniform"):
        queries = point_tensor(batch[f"{prefix}_points"], min_points=1).unsqueeze(0).to(frame.device) @ frame
        prediction = _decode(model, planes, queries, options["query_chunk_size"])
        result[prefix] = sdf_metrics(prediction, batch[f"{prefix}_sdf"].numpy(), options["sign_eps"])
    field = _decode(model, planes, grid, options["query_chunk_size"])
    resolution = round(grid.shape[1] ** (1 / 3))
    field = field.reshape(resolution, resolution, resolution)
    result["mesh"] = _mesh_result(field, reference, options, seed, output, filename)
    return result, field


def _path_summary(records, paths):
    results = {}
    for path in paths:
        rows = [row["paths"][path] for row in records]
        results[path] = {prefix: aggregate_sdf(row[prefix] for row in rows) for prefix in ("query", "uniform")}
        results[path].update(mesh_status_counts=dict(Counter(row["mesh"]["status"] for row in rows)),
                             mesh_valid_count=sum(row["mesh"]["valid"] for row in rows),
                             mesh_attempted_count=len(rows),
                             cd_x1e4_valid_only=summary(row["mesh"]["cd_x1e4"] for row in rows))
    return results


def _field_protocol():
    return ["All points and SDF values retain radius-0.5 model units; no additional input scaling is applied.",
            "Query scores use the dataset's balanced SDF samples; uniform scores are reported separately, with additive support counts. Sign ties within sign_eps are excluded.",
            "Mesh validity means a finite nonempty nondegenerate level-zero surface, not guaranteed watertightness, correct topology, or raw-scan SDF reliability. Topology is reported separately.",
            "Mesh CD compares area-sampled predicted triangles with stored reference surface points in the SAME representative coordinates. Invalid/missing surfaces receive null CD, never zero; valid-only averages are selection-biased.",
            "Tiny demonstrations, untrained checkpoints, and GPU overfit examples are not paper-quality evidence or industrial/grasp-quality validation."]


@torch.no_grad()
def eval_reconstruction(checkpoint, data_root, split="test", output="evaluation/reconstruction.json",
                        device="cpu", max_samples=None, resolution=32, **kwargs):
    """Evaluate raw, continuous, and quantized fields with one frozen decoder.

    Keyword options are documented in applications/README.md. Only explicit
    dataset splits are used. A cached full-cloud frame is supplied to encode.
    """
    from .checkpoints import load_model

    options = _field_options(kwargs)
    model, config = load_model(checkpoint, device=device)
    if config["Task"] != "tokenizer":
        raise ValueError("Reconstruction requires a tokenizer checkpoint")
    model.eval().requires_grad_(False)
    dataset = _geometry_dataset(data_root, split, config, options)
    grid = _grid(resolution, options["bounds"], device)
    paths = ("raw", "continuous", "quantized")
    records, failures = [], []
    for index in range(_limit(len(dataset), max_samples)):
        identity = dataset.items[index]["id"]
        try:
            batch = dataset[index]
            points = point_tensor(batch["points"], min_points=3).unsqueeze(0).to(device)
            frame = _cached_frame(batch["frame"], device)
            encoded = model.encode(points, frame=frame)
            reference = _reference_surface(dataset, index, frame, options)
            row = {"sample_id": identity, "category": batch["category"], "paths": {}}
            for path in paths:
                result, _ = _evaluate_planes(model, encoded[f"{path}_planes"], batch, frame, grid,
                                              reference, options, options["seed"] + index, output,
                                              f"sample_{index:06d}_{path}.ply")
                row["paths"][path] = result
            records.append(row)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            failures.append({"sample_id": identity, "error": str(error)})
    report = {"evaluation": "reconstruction", "split": split, "resolution": int(resolution),
              "options": options, "attempted_samples": len(records) + len(failures),
              "successful_samples": len(records), "failed_samples": failures,
              "paths": _path_summary(records, paths), "samples": records, "protocol": _field_protocol()}
    return _write_report(output, "reconstruction", report)


@torch.no_grad()
def eval_completion(prior_checkpoint, tokenizer_checkpoint, token_root, data_root,
                    split="test", output="evaluation/completion.json", device="cpu",
                    max_samples=None, resolution=32, ratio=0.5, seed=2026,
                    mask_ratio=None, allow_unverified_prior=False, allow_unverified_geometry=False, **kwargs):
    """Compare same-mask hard/relaxed/oracle fields using the frozen tokenizer.

    Hidden-bit accuracy is count pooled; a token is correct only when ALL its
    bits are correct. Drift uses the same representative grid and oracle cache.
    """
    from .checkpoints import load_model, verify_prior_binding
    from .provenance import dataset_fingerprint
    from .models.prior import masked_completion
    from .token_cache import TokenDataset, tokenizer_hash

    if mask_ratio is not None:
        if ratio != 0.5 and ratio != mask_ratio:
            raise ValueError("ratio and its mask_ratio alias disagree")
        ratio = mask_ratio
    options = _field_options({**kwargs, "seed": seed})
    if not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("ratio must be finite and in [0,1]")
    prior, prior_config = load_model(prior_checkpoint, device=device)
    tokenizer, config = load_model(tokenizer_checkpoint, device=device)
    if prior_config["Task"] != "prior" or config["Task"] != "tokenizer":
        raise ValueError("Completion needs a prior and a tokenizer checkpoint in that order")
    prior.eval().requires_grad_(False)
    tokenizer.eval().requires_grad_(False)
    cache = TokenDataset(token_root, split=split)
    if cache.tokenizer_hash != tokenizer_hash(tokenizer):
        raise ValueError("Cached tokens were produced by different tokenizer weights")
    categorical = prior.prior_type == "categorical"
    if cache.codebook_size != prior.codebook_size or cache.codebook_size != tokenizer.tokenizer.codebook_size:
        raise ValueError("Cache, prior and tokenizer vocabularies must agree")
    if not categorical and (cache.index_cache or cache.num_bits != prior.num_bits):
        raise ValueError("Bit-field completion requires matching LFQ bits, not VQ indices")
    binding_verified = verify_prior_binding(prior, cache.tokenizer_hash, allow_unverified=allow_unverified_prior)
    cache_fingerprint = dataset_fingerprint(token_root)
    geometry_fingerprint = dataset_fingerprint(data_root)
    cached_geometry = cache.manifest["provenance"].get("geometry_fingerprint")
    if cached_geometry is None and not allow_unverified_geometry:
        raise ValueError("Cache has no geometry fingerprint; re-export tokens or explicitly allow an unverified legacy cache")
    if cached_geometry is not None and cached_geometry != geometry_fingerprint:
        raise ValueError("Token cache and geometry have different content fingerprints; re-export tokens")
    dataset = _geometry_dataset(data_root, split, config, options)
    cached_axis_lock = cache.manifest["provenance"].get("encoder_sampling", {}).get("axis_lock")
    if cached_axis_lock is not None and cached_axis_lock != options["sampling"]["axis_lock"]:
        raise ValueError("Token cache and decoder evaluation use different frame policies; re-export tokens")
    identities = [entry["id"] for entry in dataset.items]
    if len(set(identities)) != len(identities):
        raise ValueError("Geometry identities must be unique")
    geometry_index = {identity: index for index, identity in enumerate(identities)}
    grid = _grid(resolution, options["bounds"], device)
    spacing = (options["bounds"][1] - options["bounds"][0]) / (int(resolution) - 1)
    records, failures = [], []
    paths = ("categorical", "oracle") if categorical else ("hard", "relaxed", "oracle")
    for index in range(_limit(len(cache), max_samples)):
        identity = cache.items[index]["id"]
        try:
            item = cache[index]
            if identity not in geometry_index:
                raise ValueError("Cached identity is absent from the requested geometry split")
            data_index = geometry_index[identity]
            batch = dataset[data_index]
            if batch["category"] != item["category"] or batch["split"] != item["split"]:
                raise ValueError("Cached and geometry metadata do not match")
            if categorical:
                from .models.categorical import batch_indices, categorical_completion
                target = batch_indices({key: value[None].to(device) for key, value in item.items()
                                        if isinstance(value, torch.Tensor)}, prior.codebook_size)
                completed = categorical_completion(prior, target, ratio, seed + index)
                hidden, accuracy = completed["masked"], completed["accuracy"]
                states = {"categorical": completed["indices"], "oracle": target}
                decode_states = tokenizer.decode_indices
            else:
                bits = item["bits"].unsqueeze(0).to(device)
                completed = masked_completion(prior, bits, ratio=ratio, seed=seed + index)
                hidden = completed["masked"]
                accuracy = hidden_accuracy(completed["hard_bits"], bits, hidden)
                probabilities = completed["probabilities"]
                if (probabilities.shape != bits.shape or not torch.isfinite(probabilities).all()
                        or not torch.all((probabilities >= 0) & (probabilities <= 1))):
                    raise ValueError("Completion probabilities must be finite BCHW values in [0,1]")
                if not accuracy["observed_bits_unchanged"] or not torch.all((probabilities == bits) | hidden):
                    raise ValueError("Completion changed observed tokens")
                states = {"hard": completed["hard_bits"], "relaxed": probabilities, "oracle": bits}
                decode_states = tokenizer.decode_probabilities
            frame = _cached_frame(batch["frame"], device)
            reference = _reference_surface(dataset, data_index, frame, options)
            row = {"sample_id": identity, "category": batch["category"], "mask_seed": seed + index,
                   "hidden_cells": int(hidden.sum()), "total_cells": int(hidden.numel()),
                   "accuracy": accuracy, "paths": {}}
            fields = {}
            for path in paths:
                planes = decode_states(states[path])
                result, field = _evaluate_planes(tokenizer, planes, batch, frame, grid, reference,
                                                 options, seed + index, output,
                                                 f"sample_{index:06d}_{path}.ply")
                row["paths"][path], fields[path] = result, field
            for path in paths:
                row["paths"][path]["drift"] = grid_drift(fields[path], fields["oracle"], spacing)
            records.append(row)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            failures.append({"sample_id": identity, "error": str(error)})
    count_keys = (("token_correct", "token_count") if categorical else
                  ("bit_correct", "bit_count", "token_correct", "token_count"))
    counts = {key: sum(row["accuracy"][key] for row in records) for key in count_keys}
    if not categorical:
        counts["hidden_bit_accuracy"] = counts["bit_correct"] / counts["bit_count"] if counts["bit_count"] else None
    counts["hidden_token_accuracy"] = counts["token_correct"] / counts["token_count"] if counts["token_count"] else None
    paths_summary = _path_summary(records, paths)
    for path in paths:
        paths_summary[path]["drift"] = {key: summary(row["paths"][path]["drift"][key] for row in records)
                                         for key in ("field_mae_vs_oracle", "grid_gradient_mae_vs_oracle")}
    report = {"evaluation": "completion", "split": split, "resolution": int(resolution),
              "prior_type": prior.prior_type, "quantizer_type": tokenizer.quantizer_type,
              "ratio": float(ratio), "seed": int(seed), "options": options,
              "tokenizer_hash": cache.tokenizer_hash,
              "prior_binding_verified": binding_verified, "cache_fingerprint": cache_fingerprint,
              "geometry_fingerprint_verified": cached_geometry == geometry_fingerprint,
              "frame_policy_verified": cached_axis_lock is not None,
              "attempted_samples": len(records) + len(failures), "successful_samples": len(records),
              "failed_samples": failures, "hidden_accuracy": counts, "paths": paths_summary,
              "samples": records, "protocol": _field_protocol() + [
                  "Exactly one seeded whole-cell mask is used per sample, shared by bit and categorical protocols. Predictions preserve observations; oracle uses all original cached tokens. All paths use the same frozen tokenizer decoder. Categorical priors predict hard indices only; no relaxed VQ bit states are defined.",
                  "Oracle is the unmasked tokenizer field, NOT ground-truth geometry. Gradient drift is a first-order finite-difference field diagnostic, NOT a geometric sharp-edge score.",
                  "Zero hidden cells yield null hidden accuracy, not 100 percent. Mesh failures remain explicit even when field metrics are finite.",
                  "Tokenizer/prior binding, cache content, source geometry fingerprint and sample/split/category identity are checked. Explicit legacy overrides are recorded. These checks do not establish prior training-set independence."]}
    return _write_report(output, "completion", report)
