# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Spatial LFQ bit-field completion, without an unconditional sampling path."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation,
                               dilation=dilation)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.silu(self.norm1(x)))
        y = self.dropout(y)
        y = self.conv2(F.silu(self.norm2(y)))
        return x + y


class BitFieldPrior(nn.Module):
    """The original OrbitBitFieldPrior architecture and state-dict names."""

    def __init__(self, num_bits: int = 12, hidden_channels: int = 128,
                 num_blocks: int = 8, dilations=(1, 2, 4, 8, 16, 32),
                 dropout: float = 0.1):
        super().__init__()
        if num_bits < 1 or hidden_channels < 1 or num_blocks < 1:
            raise ValueError("Bit, channel, and block counts must be positive.")
        dilation_cycle = tuple(int(d) for d in dilations)
        if not dilation_cycle or min(dilation_cycle) < 1:
            raise ValueError("dilations must contain positive integers.")
        self.num_bits = int(num_bits)
        self.stem = nn.Conv2d(self.num_bits * 2 + 2, hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(*[
            DilatedResidualBlock(hidden_channels,
                                 dilation_cycle[i % len(dilation_cycle)], dropout)
            for i in range(num_blocks)
        ])
        self.norm = nn.GroupNorm(_group_count(hidden_channels), hidden_channels)
        self.head = nn.Conv2d(hidden_channels, self.num_bits, 1)

    @staticmethod
    def coord_channels(batch: int, height: int, width: int,
                       device: torch.device) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device)
        xs = torch.linspace(-1.0, 1.0, width, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy, xx], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, observed_bits: torch.Tensor,
                observed_mask: torch.Tensor) -> torch.Tensor:
        if observed_bits.ndim != 4 or observed_bits.shape[1] != self.num_bits:
            raise ValueError(f"Expected observed bits shaped (B,{self.num_bits},H,W).")
        if observed_mask.ndim == 3:
            observed_mask = observed_mask[:, None]
        if observed_mask.ndim == 4 and observed_mask.shape[1] == 1:
            observed_mask = observed_mask.expand(-1, self.num_bits, -1, -1)
        if observed_mask.shape != observed_bits.shape:
            raise ValueError("observed_mask must broadcast to observed_bits.")
        batch, _, height, width = observed_bits.shape
        values = (observed_bits.float() * 2.0 - 1.0) * observed_mask.float()
        coords = self.coord_channels(batch, height, width, observed_bits.device)
        x = torch.cat([values, observed_mask.float(), coords], dim=1)
        return self.head(F.silu(self.norm(self.blocks(self.stem(x)))))


def _validate_bits(bits: torch.Tensor, num_bits: int) -> torch.Tensor:
    if not isinstance(bits, torch.Tensor) or bits.ndim != 4:
        raise ValueError("bits must be a BCHW tensor.")
    if bits.shape[1] != num_bits or min(bits.shape) < 1:
        raise ValueError(f"Expected nonempty BCHW bits with {num_bits} channels.")
    if not torch.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain binary values, not logits/probabilities.")
    return bits.float()


def spatial_mask(bits: torch.Tensor, ratio_min: float, ratio_max: float,
                 uncond_prob: float = 0.0) -> torch.Tensor:
    """Return B1HW hidden cells; every sample hides at least one whole cell.

    Context removal is a training regularizer, not a generation interface.
    """
    if not 0.0 <= ratio_min <= ratio_max <= 1.0:
        raise ValueError("Mask ratios must satisfy 0 <= min <= max <= 1.")
    if not 0.0 <= uncond_prob <= 1.0:
        raise ValueError("uncond_prob must be in [0, 1].")
    batch, _, height, width = bits.shape
    ratios = torch.empty(batch, 1, device=bits.device).uniform_(ratio_min, ratio_max)
    noise = torch.rand(batch, height * width, device=bits.device)
    hidden = noise < ratios
    # scatter only fills empty rows, and operates on cells rather than bit planes.
    first = noise.argmin(dim=1, keepdim=True)
    hidden.scatter_(1, first, hidden.gather(1, first) | ~hidden.any(1, keepdim=True))
    hidden = hidden | (torch.rand(batch, 1, device=bits.device) < uncond_prob)
    return hidden.reshape(batch, 1, height, width)


class CompletionPrior(nn.Module):
    """Engine adapter returning a scalar objective and additive metric counts.

    Sum bit_correct/bit_count and token_correct/token_count across batches/ranks
    before dividing. loss_sum/bit_count is the reporting BCE. Under DDP the
    returned objective accounts for different masked-bit counts on each rank.
    """

    def __init__(self, specs: dict):
        super().__init__()
        if not isinstance(specs, dict):
            raise TypeError("specs must be a configuration dictionary.")
        if specs.get("Task", "prior") != "prior":
            raise ValueError("CompletionPrior requires Task='prior'.")
        self.specs = specs
        self.prior_type = "bit_field"
        self.num_bits = int(specs.get("num_bits", 12))
        self.codebook_size = 1 << self.num_bits
        codebook_size = specs.get("Tokenizer", {}).get("codebook_size", 1 << self.num_bits)
        if int(codebook_size) != 1 << self.num_bits:
            raise ValueError("Tokenizer.codebook_size must equal 2**num_bits.")
        cfg = specs.get("model", {})
        if not cfg.get("spatial_mask", True):
            raise ValueError("The release prior supports whole-cell spatial masks only.")
        self.ratio_min = float(cfg.get("mask_ratio_min", 0.15))
        self.ratio_max = float(cfg.get("mask_ratio_max", 0.85))
        self.uncond_prob = float(cfg.get("uncond_prob", 0.15))
        if not 0 <= self.ratio_min <= self.ratio_max <= 1:
            raise ValueError("Invalid model mask-ratio interval.")
        if not 0 <= self.uncond_prob <= 1:
            raise ValueError("Invalid model uncond_prob.")
        self.model = BitFieldPrior(
            num_bits=self.num_bits,
            hidden_channels=int(cfg.get("hidden_channels", 128)),
            num_blocks=int(cfg.get("num_blocks", 8)),
            dilations=cfg.get("dilations", (1, 2, 4, 8, 16, 32)),
            dropout=float(cfg.get("dropout", 0.1)),
        )

    def forward(self, batch: Mapping, step: int = 0):
        bits = _validate_bits(batch["bits"], self.num_bits)
        hidden = spatial_mask(bits, self.ratio_min, self.ratio_max, self.uncond_prob)
        observed_bits = torch.where(hidden, 0.5, bits)
        logits = self.model(observed_bits, ~hidden)
        weights = hidden.expand_as(bits)
        loss_sum = (F.binary_cross_entropy_with_logits(logits, bits, reduction="none")
                    * weights).sum()
        bit_count = weights.sum()
        denominator = bit_count.detach().clone()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        loss = loss_sum * world_size / denominator.clamp_min(1)
        with torch.no_grad():
            correct = (logits >= 0) == bits.bool()
            bit_correct = (correct & weights).sum()
            token_count = hidden.sum()
            token_correct = (correct.all(dim=1, keepdim=True) & hidden).sum()
            metrics = {
                "loss": loss_sum.detach() / bit_count.clamp_min(1),
                "loss_sum": loss_sum.detach(),
                "bit_correct": bit_correct,
                "bit_count": bit_count,
                "token_correct": token_correct,
                "token_count": token_count,
                "bit_accuracy": bit_correct.float() / bit_count.clamp_min(1),
                "bit_acc": bit_correct.float() / bit_count.clamp_min(1),
                "token_exact": token_correct.float() / token_count.clamp_min(1),
                "mask_ratio": hidden.float().mean(),
            }
        return loss, metrics


def fixed_spatial_mask(batch, height, width, ratio, seed, device):
    """A local CPU generator gives both prior types the same exact-count mask."""
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be finite and in [0, 1].")
    cells = height * width
    count = max(int(ratio > 0), int(round(ratio * cells)))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.rand(batch, cells, generator=generator)
    indices = noise.topk(count, dim=1).indices
    masked = torch.zeros(batch, cells, dtype=torch.bool)
    masked.scatter_(1, indices, True)
    return masked.reshape(batch, 1, height, width).to(device)


@torch.no_grad()
def masked_completion(model: nn.Module, bits: torch.Tensor, ratio: float,
                      seed: int) -> dict[str, torch.Tensor]:
    """Complete exactly round(ratio*H*W) cells, preserving observed LFQ bits."""
    prior = model.model if isinstance(model, CompletionPrior) else model
    bits = _validate_bits(bits, prior.num_bits)
    batch, _, height, width = bits.shape
    masked = fixed_spatial_mask(batch, height, width, ratio, seed, bits.device)
    training = prior.training
    try:
        prior.eval()
        probabilities = torch.sigmoid(prior(torch.where(masked, 0.5, bits), ~masked))
    finally:
        prior.train(training)
    return {
        "probabilities": torch.where(masked, probabilities, bits),
        "hard_bits": torch.where(masked, (probabilities >= 0.5).float(), bits),
        "masked": masked,
    }


def build_completion_prior(specs):
    kind = specs.get("Prior", {}).get("type", "bit_field")
    if kind == "bit_field":
        return CompletionPrior(specs)
    if kind == "categorical":
        from .categorical import CategoricalCompletionPrior
        return CategoricalCompletionPrior(specs)
    raise ValueError("Prior.type must be bit_field or categorical")
