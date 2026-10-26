# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""The shared representative-frame, plane-aware SDF decoder."""

from __future__ import annotations

import torch
from torch import nn


class PlaneAwareSDFFieldHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim=384, plane_state_dim=128,
                 num_layers=4, beta=20.0, output_scale=0.1, output_bias=0.02,
                 residual_concat=True):
        super().__init__()
        if feature_dim % 3:
            raise ValueError("feature_dim must contain three concatenated plane states")
        self.plane_dim = feature_dim // 3
        self.output_scale = float(output_scale)
        self.residual_concat = bool(residual_concat)
        self.plane_norm = nn.LayerNorm(self.plane_dim)
        self.plane_encoder = nn.Sequential(nn.Linear(self.plane_dim, hidden_dim),
                                           nn.Softplus(beta=beta), nn.Linear(hidden_dim, plane_state_dim))
        self.gate = nn.Sequential(nn.LayerNorm(plane_state_dim + 3),
                                  nn.Linear(plane_state_dim + 3, hidden_dim // 2),
                                  nn.Softplus(beta=beta), nn.Linear(hidden_dim // 2, 1))
        if self.residual_concat:
            self.concat_residual = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, plane_state_dim))
        layers = []
        width = plane_state_dim + 7
        for _ in range(max(int(num_layers), 1)):
            layers.extend([nn.Linear(width, hidden_dim), nn.Softplus(beta=beta)])
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.sdf_head = nn.Sequential(*layers)
        nn.init.normal_(self.sdf_head[-1].weight, mean=0, std=1e-4)
        nn.init.constant_(self.sdf_head[-1].bias, float(output_bias))

    def forward(self, xyz, point_features):
        batch, queries, _ = point_features.shape
        planes = point_features.reshape(batch, queries, 3, self.plane_dim)
        states = self.plane_encoder(self.plane_norm(planes))
        coordinates = xyz.unsqueeze(2).expand(-1, -1, 3, -1)
        weights = self.gate(torch.cat([states, coordinates], -1)).squeeze(-1).softmax(dim=-1)
        fused = (states * weights.unsqueeze(-1)).sum(dim=2)
        if self.residual_concat:
            fused = fused + self.concat_residual(point_features)
        coordinate_features = torch.cat([xyz, xyz.square(), xyz.norm(dim=-1, keepdim=True)], -1)
        return self.sdf_head(torch.cat([coordinate_features, fused], -1)).squeeze(-1) * self.output_scale
