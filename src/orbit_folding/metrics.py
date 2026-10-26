# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Explicit model-unit metrics, bounded-memory Chamfer, and mesh diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from .meshing import MODEL_BOUNDS, mesh_topology, validate_bounds


def point_tensor(points, *, min_points=1):
    """Validate a finite floating point (N,3) or (B,N,3) cloud, preserving rank."""
    value = torch.as_tensor(points)
    if value.ndim not in (2, 3) or value.shape[-1] != 3:
        raise ValueError("points must have shape (N,3) or (B,N,3)")
    if value.shape[-2] < min_points or (value.ndim == 3 and value.shape[0] == 0):
        raise ValueError(f"point clouds need at least {min_points} points")
    if not value.is_floating_point():
        value = value.to(torch.float64)
    if not torch.isfinite(value).all():
        raise ValueError("point clouds must contain only finite coordinates")
    return value


@torch.no_grad()
def chamfer_distance(first, second, chunk_size=1024):
    """Symmetric squared CD = .5*(mean_a min_b ||a-b||^2 + reverse).

    Return one value per batch, or a scalar for two unbatched clouds. Both
    point axes are chunked, bounding pairwise storage by B*chunk_size**2.
    This is a distance between point sets, not an orientation-angle metric.
    """
    first, second = point_tensor(first), point_tensor(second)
    unbatched = first.ndim == second.ndim == 2
    first = first.unsqueeze(0) if first.ndim == 2 else first
    second = second.unsqueeze(0) if second.ndim == 2 else second
    if first.shape[0] != second.shape[0]:
        raise ValueError("Chamfer batch sizes must match")
    if first.device != second.device:
        raise ValueError("Chamfer clouds must be on the same device")
    if int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)
    dtype = torch.promote_types(first.dtype, second.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    first, second = first.to(dtype), second.to(dtype)

    def directed(source, reference):
        total = torch.zeros(source.shape[0], device=source.device, dtype=dtype)
        for queries in source.split(chunk_size, dim=1):
            best = torch.full(queries.shape[:2], float("inf"), device=source.device, dtype=dtype)
            for candidates in reference.split(chunk_size, dim=1):
                distances = torch.cdist(queries, candidates, compute_mode="donot_use_mm_for_euclid_dist")
                best = torch.minimum(best, distances.square().amin(dim=-1))
            total += best.sum(dim=1)
        return total / source.shape[1]

    result = (directed(first, second) + directed(second, first)) / 2
    return result[0] if unbatched else result


def summary(values):
    """Summarize available finite scalars; missing values are never zeros."""
    values = list(values)
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return {"count": int(len(finite)), "missing": len(values) - len(finite),
            "mean": float(finite.mean()) if len(finite) else None,
            "median": float(np.median(finite)) if len(finite) else None,
            "p95": float(np.quantile(finite, 0.95)) if len(finite) else None}


def field_support(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return {"count": int(values.size), "finite_count": int(finite.size),
            "nonfinite_count": int(values.size - finite.size),
            "negative_count": int((finite < 0).sum()), "zero_count": int((finite == 0).sum()),
            "positive_count": int((finite > 0).sum()),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "strict_zero_crossing": bool(finite.size and finite.min() < 0 < finite.max())}


def sdf_metrics(prediction, target, sign_eps=0.0005):
    """Count-weighted MAE and sign agreement, excluding target surface ties.

    Nonfinite predictions invalidate the corresponding headline metric. Finite
    support and additive sums remain available for auditing partial failures.
    """
    if not np.isfinite(sign_eps) or sign_eps < 0:
        raise ValueError("sign_eps must be finite and nonnegative")
    prediction, target = np.asarray(prediction).reshape(-1), np.asarray(target).reshape(-1)
    if prediction.shape != target.shape or not target.size:
        raise ValueError("SDF prediction and target must have matching, nonempty shapes")
    finite = np.isfinite(prediction) & np.isfinite(target)
    eligible = np.isfinite(target) & (np.abs(target) > sign_eps)
    signed = eligible & finite
    error = float(np.abs(prediction[finite].astype(float) - target[finite]).sum())
    correct = int(((prediction[signed] < 0) == (target[signed] < 0)).sum())
    count, finite_count = int(target.size), int(finite.sum())
    sign_count, sign_finite_count = int(eligible.sum()), int(signed.sum())
    return {"sdf_mae": error / count if finite_count == count else None,
            "absolute_error_sum": error, "query_count": count, "finite_count": finite_count,
            "nonfinite_count": count - finite_count,
            "sign_accuracy": correct / sign_count if sign_count and sign_count == sign_finite_count else None,
            "sign_correct": correct, "sign_count": sign_count, "sign_finite_count": sign_finite_count,
            "sign_excluded_surface_count": int((np.isfinite(target) & ~eligible).sum()),
            "sign_eps": float(sign_eps), "target_support": field_support(target),
            "prediction_support": field_support(prediction)}


def aggregate_sdf(rows):
    """Pool query counts instead of averaging differently supported samples."""
    rows = list(rows)
    names = ("absolute_error_sum", "query_count", "finite_count", "nonfinite_count",
             "sign_correct", "sign_count", "sign_finite_count", "sign_excluded_surface_count")
    result = {name: sum(row[name] for row in rows) for name in names}
    count, signed = result["query_count"], result["sign_count"]
    result["sdf_mae"] = (result["absolute_error_sum"] / count
                         if count and result["finite_count"] == count else None)
    result["sign_accuracy"] = (result["sign_correct"] / signed
                               if signed and result["sign_finite_count"] == signed else None)
    return result


def hidden_accuracy(hard_bits, target_bits, hidden):
    """Evaluate hidden bits and whole spatial tokens only (never observations)."""
    hard, target, hidden = map(torch.as_tensor, (hard_bits, target_bits, hidden))
    if hard.ndim != 4 or hard.shape != target.shape or min(hard.shape) < 1:
        raise ValueError("bits must be matching nonempty BCHW tensors")
    if hidden.dtype != torch.bool or hidden.shape != (hard.shape[0], 1, *hard.shape[2:]):
        raise ValueError("hidden must be a boolean B1HW whole-cell mask")
    if not all(torch.all((value == 0) | (value == 1)) for value in (hard, target)):
        raise ValueError("hard and target bits must be binary")
    correct = hard == target
    bit_count, token_count = int(hidden.expand_as(hard).sum()), int(hidden.sum())
    bit_correct = int((correct & hidden).sum())
    token_correct = int((correct.all(dim=1, keepdim=True) & hidden).sum())
    return {"bit_correct": bit_correct, "bit_count": bit_count,
            "token_correct": token_correct, "token_count": token_count,
            "hidden_bit_accuracy": bit_correct / bit_count if bit_count else None,
            "hidden_token_accuracy": token_correct / token_count if token_count else None,
            "observed_bits_unchanged": bool(torch.all(correct | hidden))}


def grid_drift(prediction, oracle, spacing):
    """Field drift and finite-difference gradient drift on the identical grid.

    Gradient drift is not a sharp-edge, grasp, or physical-surface quality score.
    """
    prediction, oracle = np.asarray(prediction), np.asarray(oracle)
    if prediction.shape != oracle.shape or prediction.ndim != 3 or min(prediction.shape) < 2:
        raise ValueError("drift requires matching three-dimensional grids with side lengths >= 2")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("grid spacing must be finite and positive")
    if not np.isfinite(prediction).all() or not np.isfinite(oracle).all():
        return {"field_mae_vs_oracle": None, "grid_gradient_mae_vs_oracle": None}
    differences = [np.abs(np.diff(prediction, axis=axis) - np.diff(oracle, axis=axis)) / spacing
                   for axis in range(3)]
    return {"field_mae_vs_oracle": float(np.abs(prediction - oracle).mean()),
            "grid_gradient_mae_vs_oracle": float(sum(value.sum() for value in differences)
                                                  / sum(value.size for value in differences))}


def sample_mesh_surface(vertices, faces, count, seed=2026):
    """Deterministically area-sample actual triangles; no synthetic substitute."""
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("mesh vertices must be finite (V,3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or not faces.size or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("mesh faces must contain valid triangle indices")
    if int(count) != count or count < 1:
        raise ValueError("surface sample count must be a positive integer")
    triangles = vertices[faces]
    weights = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                      triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("mesh has no positive finite surface area")
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(faces), int(count), p=weights / weights.sum())]
    u, v = rng.random((2, int(count)))
    root = np.sqrt(u)[:, None]
    return (1 - root) * chosen[:, 0] + root * (1 - v[:, None]) * chosen[:, 1] + root * v[:, None] * chosen[:, 2]


def extract_mesh(field, bounds=MODEL_BOUNDS):
    """Extract level zero with skimage; retain failure reasons and topology.

    ``valid`` means finite nonempty nondegenerate triangles, not a watertight
    solid or a reliable scan SDF. Boundary and nonmanifold edges are separate.
    Return (diagnostics, vertices, faces); absent geometry uses None arrays.
    """
    field = np.asarray(field)
    if field.ndim != 3 or min(field.shape) < 2:
        raise ValueError("mesh extraction needs a 3D field with side lengths >= 2")
    lo, hi = validate_bounds(bounds)
    result = {"valid": False, "status": "nonfinite_field", "support": field_support(field),
              "vertices": 0, "triangles": 0, "watertight": None,
              "boundary_edges": None, "nonmanifold_edges": None}
    if not np.isfinite(field).all():
        return result, None, None
    if not result["support"]["strict_zero_crossing"]:
        result["status"] = "no_isosurface"
        return result, None, None
    from skimage.measure import marching_cubes

    try:
        spacing = tuple((hi - lo) / (size - 1) for size in field.shape)
        vertices, faces, _, _ = marching_cubes(field.astype(np.float32), level=0,
                                              spacing=spacing, allow_degenerate=False)
    except (ValueError, RuntimeError) as error:
        result.update(status="extraction_failed", error=str(error))
        return result, None, None
    vertices = vertices.astype(np.float64) + lo
    faces = faces.astype(np.int64)
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        result["status"] = "invalid_geometry"
        return result, None, None
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(area).all() or np.any(area <= 0):
        result["status"] = "degenerate_triangles"
        return result, None, None
    boundary = np.concatenate([field[0].ravel(), field[-1].ravel(), field[:, 0].ravel(),
                               field[:, -1].ravel(), field[:, :, 0].ravel(), field[:, :, -1].ravel()])
    result.update(valid=True, status="ok", **mesh_topology(vertices, faces),
                  grid_boundary_negative_fraction=float((boundary < 0).mean()))
    return result, vertices, faces
