# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Geometry-only frames and rotations for row-vector point clouds."""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch


def _axis_rotations() -> torch.Tensor:
    matrices = []
    for perm in itertools.permutations(range(3)):
        permutation = torch.eye(3, dtype=torch.float64)[:, list(perm)]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = permutation @ torch.diag(torch.tensor(signs, dtype=torch.float64))
            if round(torch.linalg.det(matrix).item()) == 1:
                matrices.append(matrix)
    return torch.stack(matrices)


AXIS_ROTATIONS = _axis_rotations()
# Preserve the published probe precision before float64 scoring.
_PROBES = torch.tensor(
    [[1, 2, 3], [-2, 1, 0.5], [0.5, -1.5, 2.5],
     [1.75, 0.25, -1.25], [-1.25, -2.25, 0.75], [2.5, -0.75, -1.5]],
    dtype=torch.float64,
)
PROBE_DIRECTIONS = (_PROBES / _PROBES.norm(dim=-1, keepdim=True)).float().double()


def normalization_parameters(points, policy="centroid"):
    """Return center and diameter; centroid normalization commutes with SO(3)."""
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud) or not np.isfinite(cloud).all():
        raise ValueError("Normalization needs finite nonempty N x 3 points")
    if policy == "centroid":
        center = cloud.mean(0)
    elif policy == "legacy-bbox":
        center = (cloud.min(0) + cloud.max(0)) / 2
    else:
        raise ValueError("Normalization must be centroid or legacy-bbox")
    scale = float(2 * np.linalg.norm(cloud - center, axis=1).max())
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Observation has no spatial extent")
    return center, scale


@torch.no_grad()
def representative_frame(points: torch.Tensor, lock: bool = True) -> torch.Tensor:
    """Return proper PCA frames, with a vectorized 24-orientation axis lock.

    Accept ``(N,3)`` or ``(B,N,3)`` and return ``(3,3)`` or ``(B,3,3)``.
    Use the FULL observed cloud before sampling encoder points. Covariance and
    probe scoring use float64; frames return in the input dtype and device.
    Centering is used only to select the frame: representatives are points @ frame.
    """
    if not isinstance(points, torch.Tensor) or not points.is_floating_point():
        raise ValueError("points must be a floating-point tensor")
    if not torch.isfinite(points).all():
        raise ValueError("points must contain only finite values")
    unbatched = points.ndim == 2
    cloud = points.unsqueeze(0) if unbatched else points
    if cloud.ndim != 3 or cloud.shape[-1] != 3 or cloud.shape[1] == 0:
        raise ValueError("points must have shape (N,3) or (B,N,3), with N > 0")
    cloud = cloud.double()
    centered = cloud - cloud.mean(dim=1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered / max(cloud.shape[1] - 1, 1)
    values, vectors = torch.linalg.eigh(covariance)
    order = values.argsort(dim=-1, descending=True)
    frame = vectors.gather(2, order[:, None].expand(-1, 3, -1)).clone()
    frame[:, :, 2] *= torch.where(torch.linalg.det(frame) < 0, -1.0, 1.0)[:, None]
    if lock:
        representative = cloud @ frame
        representative = representative - representative.mean(dim=1, keepdim=True)
        rotations = AXIS_ROTATIONS.to(device=cloud.device)
        candidates = torch.einsum("bnc,kdc->bknd", representative, rotations)
        probes = PROBE_DIRECTIONS.to(device=cloud.device)
        projections = torch.einsum("bknd,md->bknm", candidates, probes)
        count = min(16, cloud.shape[1])
        top = projections.topk(count, dim=2).values.mean(dim=2)
        bottom = projections.topk(count, dim=2, largest=False).values.mean(dim=2)
        terms = (top - bottom / 8 + projections.pow(3).mean(dim=2) / 32
                 + projections.pow(5).mean(dim=2) / 128)
        scores = torch.zeros_like(terms[..., 0])
        for index in range(6):
            scores = scores * 16 + terms[..., index]
        best = rotations[scores.argmax(dim=1)]
        frame = frame @ best.transpose(1, 2)
    frame = frame.to(dtype=points.dtype)
    return frame[0] if unbatched else frame


def random_rotations(batch: int, device, dtype=torch.float32) -> torch.Tensor:
    """Sample Haar-uniform SO(3) matrices R; rotate row points with points @ R.T."""
    u1 = torch.rand(batch, device=device, dtype=dtype)
    u2 = torch.rand(batch, device=device, dtype=dtype)
    u3 = torch.rand(batch, device=device, dtype=dtype)
    x = (1 - u1).sqrt() * (2 * math.pi * u2).sin()
    y = (1 - u1).sqrt() * (2 * math.pi * u2).cos()
    z = u1.sqrt() * (2 * math.pi * u3).sin()
    w = u1.sqrt() * (2 * math.pi * u3).cos()
    return torch.stack([
        w*w + x*x - y*y - z*z, 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), w*w - x*x + y*y - z*z, 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), w*w - x*x - y*y + z*z,
    ], dim=-1).reshape(batch, 3, 3)


def yaw_rotations(order: int, *, device, dtype) -> torch.Tensor:
    """Finite representative-y-axis rotations used only for auxiliary matching."""
    angles = torch.arange(order, device=device, dtype=dtype) * (2 * math.pi / order)
    cos, sin = angles.cos(), angles.sin()
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    return torch.stack([cos, zero, sin, zero, one, zero, -sin, zero, cos], -1).reshape(-1, 3, 3)
