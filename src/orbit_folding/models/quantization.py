# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Spatial LFQ states and matched learned-centroid VQ hard-index quantization.

LFQ is an installed vector-quantize-pytorch dependency, not vendored code.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from vector_quantize_pytorch import LFQ, VectorQuantize
from vector_quantize_pytorch.lookup_free_quantization import is_distributed

from .encoder import group_count


def spatial_indices(indices, codebook_size):
    if (not isinstance(indices, torch.Tensor) or indices.dtype == torch.bool
            or indices.is_floating_point() or indices.is_complex() or indices.is_quantized):
        raise ValueError("indices must be an integer tensor")
    indices = indices.long()
    if indices.numel() == 0 or not torch.all((indices >= 0) & (indices < codebook_size)):
        raise ValueError("indices must be nonempty and in [0, codebook_size)")
    if indices.ndim == 2:
        side = math.isqrt(indices.shape[1])
        if side * side != indices.shape[1]:
            raise ValueError("Flat index sequences must have square spatial size")
        indices = indices.reshape(indices.shape[0], side, side)
    if indices.ndim != 3:
        raise ValueError("indices must have shape (B,H,W) or (B,H*W)")
    return indices


class ConvGNAct(nn.Module):
    def __init__(self, inputs, outputs, kernel_size=3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(inputs, outputs, kernel_size, padding=kernel_size // 2),
                                  nn.GroupNorm(group_count(outputs), outputs), nn.SiLU(inplace=True))

    def forward(self, inputs):
        return self.net(inputs)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(ConvGNAct(channels, channels),
                                  nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.GroupNorm(group_count(channels), channels))
        self.act = nn.SiLU(inplace=True)

    def forward(self, inputs):
        return self.act(inputs + self.net(inputs))


class TriPlaneLFQTokenizer(nn.Module):
    """Same encoder/quantizer/decoder parameter names as the trained tokenizer."""

    def __init__(self, in_channels, token_dim=256, hidden_channels=512,
                 codebook_size=4096, num_res_blocks=2, entropy_loss_weight=0.5,
                 commitment_loss_weight=0.1, diversity_gamma=2.0,
                 frac_per_sample_entropy=1.0, soft_clamp_input_value=5.0):
        super().__init__()
        self.encoder = nn.Sequential(ConvGNAct(in_channels, hidden_channels, 1),
                                     *[ResidualConvBlock(hidden_channels) for _ in range(num_res_blocks)],
                                     nn.Conv2d(hidden_channels, token_dim, 1))
        self.pre_quant_norm = nn.GroupNorm(group_count(token_dim), token_dim)
        self.quantizer = LFQ(dim=token_dim, codebook_size=codebook_size,
                             entropy_loss_weight=entropy_loss_weight,
                             commitment_loss_weight=commitment_loss_weight,
                             diversity_gamma=diversity_gamma,
                             frac_per_sample_entropy=frac_per_sample_entropy,
                             soft_clamp_input_value=soft_clamp_input_value)
        self.decoder = nn.Sequential(ConvGNAct(token_dim, hidden_channels, 1),
                                     *[ResidualConvBlock(hidden_channels) for _ in range(num_res_blocks)],
                                     nn.Conv2d(hidden_channels, in_channels, 1))
        self.codebook_size = int(codebook_size)
        self.num_bits = int(math.log2(codebook_size))
        self._distributed_context = None

    def _refresh_distributed_context(self):
        # LFQ caches this predicate globally. A model may exist before DDP init,
        # or survive teardown/reinitialization; never reuse a stale False/True.
        initialized = torch.distributed.is_initialized()
        group = torch.distributed.group.WORLD if initialized else None
        context = (initialized, id(group), torch.distributed.get_world_size() if initialized else 1)
        if context != self._distributed_context:
            is_distributed.cache_clear()
            self._distributed_context = context

    def _quantize(self, features):
        self._refresh_distributed_context()
        z_pre = self.pre_quant_norm(self.encoder(features))
        batch, channels, height, width = z_pre.shape
        flat = z_pre.permute(0, 2, 3, 1).contiguous().view(batch, height * width, channels)
        quantized, indices, auxiliary = self.quantizer(flat)
        z_q = quantized.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return z_pre, z_q, indices.reshape(batch, height, width), auxiliary

    def forward(self, features):
        z_pre, z_q, indices, auxiliary = self._quantize(features)
        return {
            "continuous_planes": self.decoder(z_pre),
            "quantized_planes": self.decoder(z_q),
            "indices": indices,
            "bits": self.indices_to_bits(indices),
            "aux_loss": auxiliary,
        }

    def encode_tokens(self, features):
        _, _, indices, _ = self._quantize(features)
        return {"indices": indices, "bits": self.indices_to_bits(indices)}

    def _spatial_indices(self, indices):
        return spatial_indices(indices, self.codebook_size)

    def indices_to_bits(self, indices):
        """MSB-first bits as (B,D,H,W), consistent with LFQ's persistent mask."""
        indices = self._spatial_indices(indices)
        bits = ((indices[..., None] & self.quantizer.mask) != 0).to(self.quantizer.dtype)
        return bits.permute(0, 3, 1, 2).contiguous()

    def decode_indices(self, indices):
        indices = self._spatial_indices(indices)
        return self.decoder(self.quantizer.indices_to_codes(indices, project_out=True).contiguous())

    def decode_probabilities(self, probabilities):
        """Map channel-first bit probabilities through the exact LFQ output projection."""
        if (not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 4
                or probabilities.shape[1] != self.num_bits or min(probabilities.shape) == 0):
            raise ValueError(f"probabilities must have shape (B,{self.num_bits},H,W)")
        if not torch.isfinite(probabilities).all() or not torch.all((probabilities >= 0) & (probabilities <= 1)):
            raise ValueError("probabilities must be finite values in [0,1]")
        values = probabilities.permute(0, 2, 3, 1).to(self.quantizer.dtype)
        codes = self.quantizer.maybe_l2norm(self.quantizer.bits_to_codes(values))
        if getattr(self.quantizer, "orthogonal_rotation", False):
            codes = codes @ self.quantizer.orthogonal_rot.t()
        projected = self.quantizer.project_out(codes.contiguous()).permute(0, 3, 1, 2).contiguous()
        return self.decoder(projected)


class TriPlaneVQTokenizer(nn.Module):
    """Matched learned-centroid VQ; no LFQ bit-probability interpretation."""

    def __init__(self, in_channels, token_dim=256, hidden_channels=512,
                 codebook_size=4096, num_res_blocks=2, commitment_loss_weight=0.25,
                 use_cosine_sim=False, decay=None):
        super().__init__()
        if int(codebook_size) != codebook_size or codebook_size < 2:
            raise ValueError("VQ codebook_size must be an integer >= 2")
        self.codebook_size = int(codebook_size)
        self.encoder = nn.Sequential(ConvGNAct(in_channels, hidden_channels, 1),
            *[ResidualConvBlock(hidden_channels) for _ in range(num_res_blocks)],
            nn.Conv2d(hidden_channels, token_dim, 1))
        self.pre_quant_norm = nn.GroupNorm(group_count(token_dim), token_dim)
        options = dict(dim=token_dim, codebook_size=self.codebook_size,
                       commitment_weight=commitment_loss_weight, use_cosine_sim=use_cosine_sim)
        if decay is not None:
            options["decay"] = float(decay)
        self.quantizer = VectorQuantize(**options)
        self.decoder = nn.Sequential(ConvGNAct(token_dim, hidden_channels, 1),
            *[ResidualConvBlock(hidden_channels) for _ in range(num_res_blocks)],
            nn.Conv2d(hidden_channels, in_channels, 1))

    def _quantize(self, features):
        z_pre = self.pre_quant_norm(self.encoder(features))
        batch, channels, height, width = z_pre.shape
        flat = z_pre.permute(0, 2, 3, 1).contiguous().view(batch, height * width, channels)
        quantized, indices, auxiliary = self.quantizer(flat)
        z_q = quantized.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return z_pre, z_q, indices.reshape(batch, height, width), auxiliary.sum()

    def forward(self, features):
        z_pre, z_q, indices, auxiliary = self._quantize(features)
        return {"continuous_planes": self.decoder(z_pre), "quantized_planes": self.decoder(z_q),
                "indices": indices, "aux_loss": auxiliary}

    def encode_tokens(self, features):
        _, _, indices, _ = self._quantize(features)
        return {"indices": indices}

    def decode_indices(self, indices):
        indices = spatial_indices(indices, self.codebook_size)
        codes = self.quantizer.get_output_from_indices(indices)
        return self.decoder(codes.permute(0, 3, 1, 2).contiguous())
