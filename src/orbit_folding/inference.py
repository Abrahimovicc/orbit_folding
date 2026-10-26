# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""SDF-label-free observation encoding and frozen-field mesh decoding."""

import json
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_model, verify_prior_binding
from .data import spatial_indices
from .geometry import normalization_parameters, representative_frame
from .metrics import extract_mesh, hidden_accuracy, point_tensor, sample_mesh_surface
from .meshing import MODEL_BOUNDS, clean_components, mesh_topology, validate_bounds
from .provenance import file_sha256
from .token_cache import tokenizer_hash


ENCODED_FORMAT = "orbit_folding.encoded.v1"
INDEX_ENCODED_FORMAT = "orbit_folding.encoded_indices.v1"


def _destination(path):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("Inference output already exists; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _observation(path, count, seed):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        points = np.load(path, allow_pickle=False)
        extent = points
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as source:
            points = source["points"].copy()
        extent = points
    else:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        if mesh.has_triangles():
            extent = np.asarray(mesh.vertices)
            points = sample_mesh_surface(extent, np.asarray(mesh.triangles), count, seed)
        else:
            points = np.asarray(o3d.io.read_point_cloud(str(path)).points).copy()
            extent = points
    cloud = point_tensor(points, min_points=3).double()
    extent = point_tensor(extent, min_points=3).double()
    if cloud.ndim != 2 or extent.ndim != 2:
        raise ValueError("Inference expects a single N x 3 observation")
    return cloud.numpy(), extent.numpy()


@torch.no_grad()
def encode_observation(checkpoint, source, output, *, device="cpu", seed=2026,
                       observation_points=8192, model_space=False, normalization=None,
                       sampling_frame=None):
    if observation_points < 3:
        raise ValueError("observation_points must be at least three")
    if model_space and normalization is not None:
        raise ValueError("model_space and an explicit normalization are mutually exclusive")
    policy = "model-space" if model_space else (normalization or "centroid")
    sampling_frame = sampling_frame or ("observation" if policy == "legacy-bbox" else "representative")
    if sampling_frame not in {"observation", "representative"}:
        raise ValueError("sampling_frame must be observation or representative")
    destination = _destination(output)
    if destination.suffix.lower() != ".npz":
        raise ValueError("Encoded output must have .npz extension")
    model, config = load_model(checkpoint, device)
    if config["Task"] != "tokenizer":
        raise ValueError("Encoding needs a tokenizer checkpoint")
    points, extent = _observation(source, observation_points, seed)
    center, scale = (np.zeros(3), 1.0) if model_space else normalization_parameters(extent, policy)
    points = ((points - center) / scale).astype(np.float32)
    if np.square(points - points.mean(0)).sum() <= 1e-20:
        raise ValueError("Observation has no identifiable spatial extent")
    frame = representative_frame(torch.from_numpy(points),
                                 lock=bool(config.get("EnableAxisRepresentativeLock", True))).numpy()
    rng = np.random.default_rng(seed)
    count = int(config.get("PCsize", 8192))
    strategy = config.get("PointSamplingStrategy", "random")
    if strategy not in {"random", "spatial_mixed", "spatial_balanced"}:
        raise ValueError("Unsupported point sampling strategy")
    if strategy == "random":
        selected = rng.choice(len(points), count, replace=len(points) < count)
    else:
        random_count = round(count * config.get("PointRandomRatio", 0.25)) if strategy == "spatial_mixed" else 0
        sampling_points = points @ frame if sampling_frame == "representative" else points
        spatial = spatial_indices(sampling_points, np.arange(len(points)), count - random_count, rng,
                                  config.get("PointSpatialBins", 28))
        random = rng.choice(len(points), random_count, replace=len(points) < random_count)
        selected = rng.permutation(np.r_[spatial, random])
    encoded = model.encode_tokens(torch.from_numpy(points[selected])[None].to(device),
                                  frame=torch.from_numpy(frame)[None].to(device))
    index_values = model.quantizer_type == "standard_vq"
    field = "indices" if index_values else "bits"
    bits = encoded[field][0].cpu().numpy().astype(np.int64 if index_values else np.uint8)
    encoded_format = INDEX_ENCODED_FORMAT if index_values else ENCODED_FORMAT
    metadata = {"format": encoded_format, "source_name": Path(source).name,
                "source_sha256": file_sha256(source), "tokenizer_hash": tokenizer_hash(model),
                "observed_points": len(points), "encoder_points": len(selected), "seed": seed,
                "normalization": policy, "sampling_frame": sampling_frame,
                "model_space_input": bool(model_space), "quantizer_type": model.quantizer_type,
                ("token_shape" if index_values else "bit_shape"): list(bits.shape),
                "scope": "Encoding of an observed point set; no SDF labels or semantic pose are used"}
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **{field: bits}, frame=frame, center=center, scale=np.float64(scale),
                            tokenizer_hash=np.str_(metadata["tokenizer_hash"]),
                            format=np.str_(encoded_format), metadata=np.str_(json.dumps(metadata)))
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


@torch.no_grad()
def decode_observation(checkpoint, tokens, output, *, device="cpu", resolution=96,
                       space="input", prior_checkpoint=None, mask_ratio=0.5, state=None,
                       seed=2026, allow_unverified_prior=False, bounds=MODEL_BOUNDS,
                       component_min_fraction=0., component_min_triangles=0,
                       component_min_extent_fraction=0., query_chunk_size=8192):
    from .evaluation import _cached_frame, _decode, _grid, _positive

    if space not in {"input", "representative"}:
        raise ValueError("space must be input or representative")
    bounds = validate_bounds(bounds)
    query_chunk_size = _positive(query_chunk_size, "query_chunk_size")
    if (not 0 <= component_min_fraction <= 1 or not 0 <= component_min_extent_fraction <= 1
            or int(component_min_triangles) != component_min_triangles or component_min_triangles < 0):
        raise ValueError("Invalid component filtering parameters")
    destination = _destination(output)
    if destination.suffix.lower() != ".ply":
        raise ValueError("Decoded mesh output must have .ply extension")
    model, config = load_model(checkpoint, device)
    if config["Task"] != "tokenizer":
        raise ValueError("Decoding needs a tokenizer checkpoint")
    expected_hash = tokenizer_hash(model)
    index_values = model.quantizer_type == "standard_vq"
    with np.load(tokens, allow_pickle=False) as record:
        if str(record["format"].item()) != (INDEX_ENCODED_FORMAT if index_values else ENCODED_FORMAT):
            raise ValueError("Input must be produced by orbit encode")
        if str(record["tokenizer_hash"].item()) != expected_hash:
            raise ValueError("Encoded tokens use different tokenizer weights")
        bits = np.asarray(record["indices" if index_values else "bits"])
        frame = _cached_frame(record["frame"], "cpu")[0].numpy()
        center, scale = np.asarray(record["center"]), float(record["scale"])
    spatial_shape = (int(config.get("plane_res", 128)),) * 2
    if index_values:
        if (bits.dtype != np.int64 or bits.shape != spatial_shape
                or not np.all((bits >= 0) & (bits < model.tokenizer.codebook_size))):
            raise ValueError("Encoded VQ indices must be int64 HW within the checkpoint vocabulary")
    elif (bits.dtype != np.uint8 or bits.shape != (model.tokenizer.num_bits, *spatial_shape)
          or not np.all((bits == 0) | (bits == 1))):
        raise ValueError("Encoded bits must be binary uint8 CHW with the checkpoint's bit and spatial dimensions")
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Encoded coordinate recovery is invalid")
    bits = torch.from_numpy(bits.astype(np.int64 if index_values else np.float32))[None].to(device)
    selected_state = "oracle" if state is None and prior_checkpoint is None else (state or "relaxed")
    if selected_state not in {"oracle", "hard", "relaxed"}:
        raise ValueError("state must be oracle, hard or relaxed")
    accuracy, verified = None, None
    values = bits
    if prior_checkpoint is not None:
        from .models.prior import masked_completion

        prior, prior_config = load_model(prior_checkpoint, device)
        if prior_config["Task"] != "prior" or prior.codebook_size != model.tokenizer.codebook_size:
            raise ValueError("Prior task and vocabulary must match the tokenizer")
        verified = verify_prior_binding(prior, expected_hash, allow_unverified=allow_unverified_prior)
        if prior.prior_type == "categorical":
            from .models.categorical import batch_indices, categorical_completion
            selected_state = state or "hard"
            if selected_state == "relaxed":
                raise ValueError("Categorical priors support hard indices only, not relaxed LFQ states")
            indices = batch_indices({"indices" if index_values else "bits": bits}, prior.codebook_size)
            completed = categorical_completion(prior, indices, mask_ratio, seed)
            accuracy = completed["accuracy"]
            values = indices if selected_state == "oracle" else completed["indices"]
            index_values = True
        else:
            if index_values:
                raise ValueError("Bit-field prior requires LFQ tokens, not VQ indices")
            completed = masked_completion(prior, bits, mask_ratio, seed)
            accuracy = hidden_accuracy(completed["hard_bits"], bits, completed["masked"])
            values = {"oracle": bits, "hard": completed["hard_bits"], "relaxed": completed["probabilities"]}[selected_state]
    elif selected_state != "oracle":
        raise ValueError("Hard or relaxed completion requires --prior")
    planes = model.decode_indices(values) if index_values else model.decode_probabilities(values)
    grid = _grid(resolution, bounds, device)
    field = _decode(model, planes, grid, query_chunk_size).reshape((int(resolution),) * 3)
    mesh_report, vertices, faces = extract_mesh(field, bounds)
    report = {"state": selected_state, "space": space, "resolution": int(resolution),
              "tokenizer_hash": expected_hash, "prior_binding_verified": verified,
              "mask_ratio": float(mask_ratio) if prior_checkpoint else None,
              "hidden_accuracy": accuracy, "mesh": mesh_report,
              "protocol": {"bounds": list(bounds), "level": 0., "query_chunk_size": query_chunk_size,
                  "component_min_fraction": component_min_fraction,
                  "component_min_triangles": component_min_triangles,
                  "component_min_extent_fraction": component_min_extent_fraction,
                  "smoothing": False}}
    if mesh_report["valid"]:
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals()
        if component_min_fraction or component_min_triangles or component_min_extent_fraction:
            report["mesh_before_cleanup"] = dict(mesh_report)
            mesh = clean_components(mesh, component_min_fraction, component_min_triangles,
                                    component_min_extent_fraction)
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            mesh_report.update(mesh_topology(vertices, faces))
        if space == "input":
            mesh.vertices = o3d.utility.Vector3dVector((vertices @ frame.T) * scale + center)
            mesh.compute_vertex_normals()
        if not o3d.io.write_triangle_mesh(str(destination), mesh):
            raise OSError("Mesh export failed")
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
