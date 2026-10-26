# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Local point messages followed by bilinear tri-plane splatting."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def group_count(channels: int) -> int:
    return next(groups for groups in (32, 16, 8, 4, 2, 1) if channels % groups == 0)


class PointNeXtLocalBlock(nn.Module):
    """Relative-feature/position messages with max aggregation and a residual FFN."""

    def __init__(self, channels: int):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(channels + 4, channels), nn.ReLU(inplace=True),
                                     nn.Linear(channels, channels))
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU(),
                                 nn.Linear(2 * channels, channels))

    def forward(self, points, features, knn_idx):
        batch = torch.arange(points.shape[0], device=points.device)[:, None, None]
        relative = points[batch, knn_idx] - points[:, :, None]
        messages = torch.cat([features[batch, knn_idx] - features[:, :, None],
                              relative, relative.norm(dim=-1, keepdim=True)], dim=-1)
        output = features + self.message(messages).max(dim=2).values
        return output + self.ffn(self.norm(output))


class PlaneResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(group_count(channels), channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(group_count(channels), channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, inputs):
        return inputs + self.net(inputs)


class PlaneRefiner(nn.Module):
    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.Sequential(*[PlaneResidualBlock(channels) for _ in range(num_blocks)])

    def forward(self, inputs):
        return self.blocks(inputs)


class PointNeXtTriPlaneEncoder(nn.Module):
    """Compact PointNeXt-like encoder; no external point operators are needed."""

    def __init__(self, c_dim=256, hidden_dim=128, plane_resolution=128,
                 padding=0.1, num_blocks=3, knn_k=12, knn_chunk_size=1024,
                 knn_reference_size=2048, plane_refine_blocks=3,
                 grid_sample_align_corners=False, output_scale=0.025):
        super().__init__()
        self.c_dim = int(c_dim)
        self.reso_plane = int(plane_resolution)
        self.padding = float(padding)
        self.knn_k = int(knn_k)
        self.knn_chunk_size = int(knn_chunk_size)
        self.knn_reference_size = knn_reference_size
        self.grid_sample_align_corners = bool(grid_sample_align_corners)
        self.output_scale = float(output_scale)
        self.input = nn.Sequential(nn.Linear(7, hidden_dim), nn.LayerNorm(hidden_dim),
                                   nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([PointNeXtLocalBlock(hidden_dim) for _ in range(num_blocks)])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, c_dim))
        self.refiner = PlaneRefiner(c_dim, plane_refine_blocks)

    @torch.no_grad()
    def _knn_indices(self, points):
        batch_size, count, _ = points.shape
        if count < 2:
            raise ValueError("The local encoder needs at least two points")
        k = min(max(self.knn_k, 1), count - 1)
        reference = points
        reference_indices = None
        if self.knn_reference_size and 0 < self.knn_reference_size < count:
            reference_indices = torch.linspace(0, count - 1, int(self.knn_reference_size),
                                               device=points.device).round().long().unique()
            reference = points[:, reference_indices]
            k = min(k, max(reference_indices.numel() - 1, 1))
            positions = torch.full((count,), -1, device=points.device, dtype=torch.long)
            positions[reference_indices] = torch.arange(reference_indices.numel(), device=points.device)
        chunks = []
        for start in range(0, count, max(self.knn_chunk_size, 1)):
            end = min(start + max(self.knn_chunk_size, 1), count)
            distances = torch.cdist(points[:, start:end], reference)
            local = torch.arange(start, end, device=points.device)
            if reference_indices is None:
                distances[:, torch.arange(end - start, device=points.device), local] = float("inf")
            else:
                local_positions = positions[local]
                valid = local_positions >= 0
                distances[:, torch.arange(end - start, device=points.device)[valid],
                          local_positions[valid]] = float("inf")
            indices = distances.topk(k=k, dim=-1, largest=False).indices
            chunks.append(indices if reference_indices is None else reference_indices[indices])
        return torch.cat(chunks, dim=1).reshape(batch_size, count, k)

    def get_point_cloud_features(self, points):
        inputs = torch.cat([points, points.square(), points.norm(dim=-1, keepdim=True)], -1)
        features = self.input(inputs)
        neighbors = self._knn_indices(points)
        for block in self.blocks:
            features = block(points, features, neighbors)
        return self.output(features)

    def normalize_coordinate(self, points, plane="xz"):
        axes = {"xz": [0, 2], "xy": [0, 1], "yz": [1, 2]}[plane]
        return (points[..., axes] / (1 + self.padding + 1e-5) + 0.5).clamp(0, 1 - 1e-6)

    def _bilinear_splat(self, coordinates, features):
        batch, _, channels = features.shape
        res = self.reso_plane
        coordinates = (coordinates * (res - 1) if self.grid_sample_align_corners
                       else coordinates * res - 0.5)
        base = coordinates.floor().long()
        fraction = coordinates - base.to(coordinates.dtype)
        plane = features.new_zeros(batch, channels, res * res)
        denominator = features.new_zeros(batch, 1, res * res)
        for ox, wx in ((0, 1 - fraction[..., 0]), (1, fraction[..., 0])):
            ix = (base[..., 0] + ox).clamp(0, res - 1)
            for oy, wy in ((0, 1 - fraction[..., 1]), (1, fraction[..., 1])):
                iy = (base[..., 1] + oy).clamp(0, res - 1)
                weight = (wx * wy).clamp_min(0).to(features.dtype)
                index = ix + res * iy
                source = (features * weight.unsqueeze(-1)).transpose(1, 2)
                plane.scatter_add_(2, index[:, None].expand(-1, channels, -1), source)
                denominator.scatter_add_(2, index[:, None], weight[:, None])
        return (plane / denominator.clamp_min(1e-6)).view(batch, channels, res, res)

    def get_plane_features(self, points):
        features = self.get_point_cloud_features(points)
        return tuple(self.refiner(self._bilinear_splat(self.normalize_coordinate(points, name), features))
                     * self.output_scale for name in ("xz", "xy", "yz"))

    def forward(self, points):
        return torch.cat(self.get_plane_features(points), dim=1)

    def sample_plane_feature(self, queries, plane_feature, plane):
        coordinates = self.normalize_coordinate(queries, plane)
        grid = (2 * coordinates[:, :, None] - 1).to(plane_feature.dtype)
        return F.grid_sample(plane_feature, grid, mode="bilinear", padding_mode="border",
                             align_corners=self.grid_sample_align_corners).squeeze(-1)

    def sample_features(self, planes, queries):
        return torch.cat([self.sample_plane_feature(queries, plane, name).transpose(1, 2)
                          for plane, name in zip(planes.chunk(3, dim=1), ("xz", "xy", "yz"))], -1)
