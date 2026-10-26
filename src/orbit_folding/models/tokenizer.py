# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Geometric representatives, spatial LFQ/VQ, and shared SDF decoding.

All public inputs are already in model space. No InputScaleFactor is applied.
For historical checkpoints retain pointnet.*, tokenizer.*, and field_head.*.
Drop only unused pose_predictor.* and token_field_head.* tensors before strict
loading: the selected configuration never used those modules. The architecture
and all active tensor names/shapes are unchanged.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from ..geometry import representative_frame
from .encoder import PointNeXtTriPlaneEncoder
from .field import PlaneAwareSDFFieldHead
from .quantization import TriPlaneLFQTokenizer, TriPlaneVQTokenizer


class ReconstructiveTokenizer(nn.Module):
    """The active paper model, configured with the original internal key schema."""

    def __init__(self, specs: dict):
        super().__init__()
        self.specs = copy.deepcopy(specs)
        latent_dim = int(specs["SdfModelSpecs"].get("latent_dim", 256))
        encoder = specs.get("Encoder", {})
        quantization = specs.get("Tokenizer", {})
        field = specs.get("FieldDecoder", {})
        if specs.get("Canonicalizer", "pca_representative") not in ("pca", "pca_representative"):
            raise ValueError("Only the geometric PCA representative is supported")
        if encoder.get("type", "pointnext_triplane") not in ("pointnext_triplane", "pointnext", "pointnext_lite"):
            raise ValueError("Only the local PointNeXt-like encoder is supported")
        self.quantizer_type = quantization.get("type", "lfq")
        if self.quantizer_type not in {"lfq", "standard_vq"}:
            raise ValueError("Tokenizer.type must be lfq or standard_vq")
        if field.get("type", "pointnext_plane_aware") not in ("pointnext_plane_aware", "plane_aware"):
            raise ValueError("Only the plane-aware field head is supported")
        if field.get("plane_feature_aggregation", "concat") != "concat":
            raise ValueError("The active field head needs concatenated plane features")
        if not specs.get("ShareTokenFieldHead", True):
            raise ValueError("The active model shares one field head across all paths")
        self.pointnet = PointNeXtTriPlaneEncoder(
            c_dim=latent_dim, hidden_dim=int(encoder.get("hidden_dim", specs["SdfModelSpecs"].get("pn_hidden_dim", 128))),
            plane_resolution=int(specs.get("plane_res", 128)),
            **{key: encoder[key] for key in (
                "padding", "num_blocks", "knn_k", "knn_chunk_size", "knn_reference_size",
                "plane_refine_blocks", "grid_sample_align_corners", "output_scale") if key in encoder},
        )
        quantizer_class = TriPlaneLFQTokenizer if self.quantizer_type == "lfq" else TriPlaneVQTokenizer
        optional = (("entropy_loss_weight", "diversity_gamma", "frac_per_sample_entropy", "soft_clamp_input_value")
                    if self.quantizer_type == "lfq" else ("use_cosine_sim", "decay"))
        self.tokenizer = quantizer_class(
            in_channels=latent_dim * 3,
            **{key: quantization[key] for key in (
                "token_dim", "hidden_channels", "codebook_size", "num_res_blocks", "commitment_loss_weight",
                *optional) if key in quantization},
        )
        self.field_head = PlaneAwareSDFFieldHead(
            feature_dim=latent_dim * 3,
            **{key: field[key] for key in ("hidden_dim", "plane_state_dim", "num_layers", "beta",
                                          "output_scale", "output_bias", "residual_concat") if key in field},
        )

    def representative_features(self, points, frame=None):
        """Use a cached full-cloud frame, or estimate one from the supplied cloud."""
        if frame is None:
            frame = representative_frame(points, lock=bool(self.specs.get("EnableAxisRepresentativeLock", True)))
        frame = frame.to(device=points.device, dtype=points.dtype)
        if frame.ndim == 2:
            frame = frame.unsqueeze(0).expand(points.shape[0], -1, -1)
        representative = points @ frame
        return {"rotation": frame, "representative": representative,
                "raw_planes": self.pointnet(representative)}

    def encode(self, points, frame=None) -> dict:
        """Return the frame, representative, three plane paths, indices and auxiliary loss.

        points: (B,N,3); frame: (B,3,3). Planes: (B,3C,H,W), indices:
        (B,H,W). LFQ additionally exposes bits: (B,D,H,W), with D=log2(K).
        Only the tokenizer input
        is detached when configured; raw planes keep their encoder gradients.
        """
        result = self.representative_features(points, frame)
        raw = result["raw_planes"]
        token_input = raw.detach() if self.specs.get("Tokenizer", {}).get("detach_input", True) else raw
        result.update(self.tokenizer(token_input))
        return result

    def decode(self, planes, queries_in_representative):
        """Evaluate the shared SDF field at (B,Q,3) representative-space queries."""
        features = self.pointnet.sample_features(planes, queries_in_representative)
        return self.field_head(queries_in_representative, features)

    @torch.no_grad()
    def encode_tokens(self, points, frame=None):
        """Export tokens without running either reconstruction-plane decoder."""
        if self.training:
            raise ValueError("Token-only export requires eval mode; use encode for training")
        result = self.representative_features(points, frame)
        raw = result.pop("raw_planes")
        result.update(self.tokenizer.encode_tokens(raw))
        return result

    def decode_indices(self, indices):
        """Restore tri-planes from integer spatial LFQ or VQ indices."""
        return self.tokenizer.decode_indices(indices)

    def decode_probabilities(self, probabilities):
        """Restore tri-planes from (B,D,H,W) relaxed LFQ bit states."""
        if self.quantizer_type != "lfq":
            raise ValueError("Relaxed bit states are LFQ-specific; VQ requires decode_indices")
        return self.tokenizer.decode_probabilities(probabilities)

    def forward(self, batch, step=0):
        """Return (scalar loss, detached metrics) for an already-model-space batch."""
        from ..objectives import tokenizer_objective

        return tokenizer_objective(self, batch, step)
