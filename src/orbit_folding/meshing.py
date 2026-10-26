# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Shared model-space meshing and extent-aware component filtering."""

import numpy as np


MODEL_BOUNDS = (-0.5, 0.5)


def validate_bounds(bounds):
    if len(bounds) != 2:
        raise ValueError("bounds must contain two finite increasing values")
    lo, hi = map(float, bounds)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("bounds must contain two finite increasing values")
    return lo, hi


def mesh_topology(vertices, faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {"vertices": len(vertices), "triangles": len(faces),
            "boundary_edges": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum()),
            "watertight": bool(len(counts) and np.all(counts == 2))}


def clean_components(mesh, min_fraction=.02, min_triangles=64, min_extent_fraction=0.):
    """Filter small components while optionally retaining long, thin supports.

    Extent and triangle-count tests are alternatives, matching the original
    paper export. No vertex smoothing or largest-component-only filter is used.
    """
    import open3d as o3d

    if (not 0 <= min_fraction <= 1 or not 0 <= min_extent_fraction <= 1
            or int(min_triangles) != min_triangles or min_triangles < 0):
        raise ValueError("Component fractions must lie in [0,1] and triangle count must be nonnegative")
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    if len(counts) <= 1:
        return mesh
    keep = counts >= max(int(np.ceil(counts.max() * min_fraction)), min_triangles)
    if min_extent_fraction > 0:
        points = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)].reshape(-1, 3)
        components = np.repeat(labels, 3)
        lower, upper = np.full((len(counts), 3), np.inf), np.full((len(counts), 3), -np.inf)
        np.minimum.at(lower, components, points)
        np.maximum.at(upper, components, points)
        extent = np.linalg.norm(upper - lower, axis=1)
        keep |= extent >= extent.max() * min_extent_fraction
    if not keep.any():
        keep[counts.argmax()] = True
    result = o3d.geometry.TriangleMesh(mesh)
    result.remove_triangles_by_mask((~keep[labels]).tolist())
    result.remove_unreferenced_vertices().compute_vertex_normals()
    return result
