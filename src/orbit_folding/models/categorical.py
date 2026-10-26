# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Unconditioned masked-index prior, matching the original categorical backbone."""

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from .prior import DilatedResidualBlock, _group_count, _validate_bits, fixed_spatial_mask, spatial_mask
from .quantization import spatial_indices


def batch_indices(batch, codebook_size):
    """LFQ has true MSB-first bits; VQ supplies integer indices directly."""
    if "indices" in batch:
        return spatial_indices(batch["indices"], codebook_size)
    if codebook_size < 2 or codebook_size & (codebook_size - 1):
        raise ValueError("Binary LFQ fields require a power-of-two vocabulary")
    width = codebook_size.bit_length() - 1
    bits = _validate_bits(batch["bits"], width)
    weights = 1 << torch.arange(width - 1, -1, -1, device=bits.device)
    return (bits.long() * weights[None, :, None, None]).sum(1)


class MaskedTokenMapPrior(nn.Module):
    """Original model.* parameter names, including learned row/column embeddings."""

    def __init__(self, vocab_size=4096, hidden_channels=256, max_resolution=128,
                 num_blocks=12, dilations=(1, 2, 4, 8, 16, 32), dropout=0.1):
        super().__init__()
        if min(vocab_size, hidden_channels, max_resolution, num_blocks) < 1 or vocab_size < 2:
            raise ValueError("Positive categorical model dimensions and vocabulary >= 2 required")
        if not dilations or min(dilations) < 1:
            raise ValueError("dilations must contain positive integers")
        self.vocab_size, self.mask_token_id = int(vocab_size), int(vocab_size)
        self.max_resolution = int(max_resolution)
        self.token_embedding = nn.Embedding(vocab_size + 1, hidden_channels)
        self.row_embedding = nn.Embedding(max_resolution, hidden_channels)
        self.col_embedding = nn.Embedding(max_resolution, hidden_channels)
        self.input_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.blocks = nn.Sequential(*[DilatedResidualBlock(hidden_channels, dilations[i % len(dilations)], dropout)
                                      for i in range(num_blocks)])
        self.final_norm = nn.GroupNorm(_group_count(hidden_channels), hidden_channels)
        self.head = nn.Conv2d(hidden_channels, vocab_size, 1)

    def forward(self, tokens):
        tokens = spatial_indices(tokens, self.vocab_size + 1)
        batch, height, width = tokens.shape
        if max(height, width) > self.max_resolution:
            raise ValueError("Token map exceeds categorical max_resolution")
        rows = torch.arange(height, device=tokens.device)
        cols = torch.arange(width, device=tokens.device)
        x = (self.token_embedding(tokens) + self.row_embedding(rows)[None, :, None, :]
             + self.col_embedding(cols)[None, None, :, :])
        x = self.input_proj(x.permute(0, 3, 1, 2).contiguous())
        return self.head(F.silu(self.final_norm(self.blocks(x)))).permute(0, 2, 3, 1).contiguous()


class CategoricalCompletionPrior(nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.specs, self.prior_type = specs, "categorical"
        if specs.get("Task", "prior") != "prior" or specs.get("num_classes", 0) != 0:
            raise ValueError("Categorical baseline requires Task=prior and no category conditioning")
        self.codebook_size = int(specs.get("vocab_size", specs.get("Tokenizer", {}).get("codebook_size", 4096)))
        if specs.get("Tokenizer", {}).get("codebook_size", self.codebook_size) != self.codebook_size:
            raise ValueError("Categorical vocabulary and Tokenizer.codebook_size disagree")
        cfg = specs.get("model", {})
        self.ratio_min = float(cfg.get("mask_ratio_min", .15))
        self.ratio_max = float(cfg.get("mask_ratio_max", .85))
        if not 0 <= self.ratio_min <= self.ratio_max <= 1:
            raise ValueError("Invalid categorical mask ratio interval")
        if not cfg.get("spatial_mask", True) or cfg.get("uncond_prob", 0) != 0:
            raise ValueError("Categorical baseline uses whole-cell masking without unconditional training")
        self.model = MaskedTokenMapPrior(self.codebook_size, int(cfg.get("hidden_channels", 256)),
            int(specs.get("max_resolution", specs.get("plane_res", 128))), int(cfg.get("num_blocks", 12)),
            tuple(cfg.get("dilations", (1, 2, 4, 8, 16, 32))), float(cfg.get("dropout", .1)))

    def forward(self, batch, step=0):
        indices = batch_indices(batch, self.codebook_size)
        hidden = spatial_mask(indices[:, None], self.ratio_min, self.ratio_max)[:, 0]
        logits = self.model(torch.where(hidden, self.model.mask_token_id, indices))
        loss_sum = F.cross_entropy(logits[hidden], indices[hidden], reduction="sum")
        count = hidden.sum()
        denominator, world = count.detach().clone(), 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(denominator)
            world = dist.get_world_size()
        with torch.no_grad():
            correct = ((logits.argmax(-1) == indices) & hidden).sum()
            metrics = {"loss": loss_sum.detach() / count, "loss_sum": loss_sum.detach(),
                "loss_count": count, "token_count": count, "token_correct": correct,
                "token_exact": correct.float() / count, "mask_ratio": hidden.float().mean()}
        return loss_sum * world / denominator.clamp_min(1), metrics


@torch.no_grad()
def categorical_completion(prior, indices, ratio, seed):
    indices = spatial_indices(indices, prior.codebook_size)
    batch, height, width = indices.shape
    hidden = fixed_spatial_mask(batch, height, width, ratio, seed, indices.device)[:, 0]
    training = prior.training
    try:
        prior.eval()
        logits = prior.model(torch.where(hidden, prior.model.mask_token_id, indices))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite categorical logits")
        predictions = torch.where(hidden, logits.argmax(-1), indices)
        loss_sum = float(F.cross_entropy(logits[hidden], indices[hidden], reduction="sum")) if hidden.any() else 0.0
    finally:
        prior.train(training)
    count = int(hidden.sum())
    correct = int(((predictions == indices) & hidden).sum())
    return {"indices": predictions, "masked": hidden[:, None],
            "accuracy": {"token_correct": correct, "token_count": count,
                         "hidden_token_accuracy": correct / count if count else None,
                         "observed_tokens_unchanged": bool(torch.all((predictions == indices) | hidden))},
            "loss_sum": loss_sum}
