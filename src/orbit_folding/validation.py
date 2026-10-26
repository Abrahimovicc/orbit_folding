# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Held-out field, masked-bit and masked-index validation without sampler padding."""

import math

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def validate(model, dataset, *, task, device, rank=0, world=1, batch_size=1, seed=2026,
             mask_ratios=None, selection_metric=None):
    from .models.prior import fixed_spatial_mask

    model = model.module if hasattr(model, "module") else model
    categorical = task == "prior" and model.prior_type == "categorical"
    training = model.training
    generator = torch.Generator().manual_seed(seed)
    indices = list(range(rank, len(dataset), world))
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=0, generator=generator)
    ratios = [.5] if mask_ratios is None else list(mask_ratios)
    if (not ratios or len(set(ratios)) != len(ratios)
            or any(not math.isfinite(ratio) or not 0 < ratio <= 1 for ratio in ratios)):
        raise ValueError("Validation mask ratios must be distinct values in (0, 1]")
    if task == "prior" and .5 not in ratios:
        raise ValueError("Include 0.5 for the primary validation metric")
    keys = (["absolute_error_sum", "query_count", "sign_correct", "sign_count", "truncated_absolute_error_sum"] if task == "tokenizer"
            else (["loss_sum", "loss_count", "token_count", "token_correct"] if categorical
                  else ["loss_sum", "bit_count", "bit_correct", "token_count", "token_correct"]))
    totals = torch.zeros((1 if task == "tokenizer" else len(ratios), len(keys)), dtype=torch.float64, device=device)
    histogram = (torch.zeros(model.tokenizer.codebook_size, dtype=torch.int64, device=device)
                 if task == "tokenizer" else None)
    try:
        model.eval()
        # Prior masks are keyed by sample position, not batch size or rank.
        position = 0
        for raw in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in raw.items()}
            if task == "tokenizer":
                encoded = model.encode(batch["points"], frame=batch["frame"])
                prediction = model.decode(encoded["quantized_planes"], batch["uniform_points"] @ batch["frame"])
                target = batch["uniform_sdf"]
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Non-finite validation SDF field")
                eligible = target.abs() > 0.0005
                truncation = float(model.specs.get("SdfTruncation", .1))
                values = [(prediction.double() - target).abs().sum(), target.numel(),
                          (((prediction < 0) == (target < 0)) & eligible).sum(), eligible.sum(),
                          (prediction.double() - target.clamp(-truncation, truncation)).abs().sum()]
                totals[0] += torch.stack([torch.as_tensor(value, device=device, dtype=torch.float64) for value in values])
                histogram += torch.bincount(encoded["indices"].reshape(-1), minlength=len(histogram))
            else:
                from .models.categorical import batch_indices
                values = batch_indices(batch, model.codebook_size) if categorical else batch["bits"]
                count, height, width = len(values), *values.shape[-2:]
                for row, ratio in enumerate(ratios):
                    hidden = torch.cat([fixed_spatial_mask(1,height,width,ratio,seed+identity,"cpu")
                                        for identity in indices[position:position+count]]).to(device)
                    if categorical:
                        cell_mask = hidden[:,0]
                        logits = model.model(torch.where(cell_mask,model.model.mask_token_id,values))
                        if not torch.isfinite(logits).all():
                            raise FloatingPointError("Non-finite validation index logits")
                        loss_sum = F.cross_entropy(logits[cell_mask],values[cell_mask],reduction="none").double().sum()
                        correct = (logits.argmax(-1)==values) & cell_mask
                        current = [loss_sum,cell_mask.sum(),cell_mask.sum(),correct.sum()]
                    else:
                        probabilities = model.model(torch.where(hidden,.5,values),~hidden).sigmoid()
                        if not torch.isfinite(probabilities).all():
                            raise FloatingPointError("Non-finite validation bit probabilities")
                        weights = hidden.expand_as(values)
                        correct = (probabilities>=.5)==values.bool()
                        current = [F.binary_cross_entropy(probabilities,values,reduction="none")[weights].double().sum(),
                                   weights.sum(),(correct & weights).sum(),hidden.sum(),
                                   (correct.all(1,keepdim=True) & hidden).sum()]
                    totals[row] += torch.stack([value.to(dtype=torch.float64) for value in current])
                position += count
    finally:
        model.train(training)
    if world > 1:
        dist.all_reduce(totals)
        if histogram is not None:
            dist.all_reduce(histogram)
    measured = [dict(zip(keys,row)) for row in totals.cpu().tolist()]
    result = measured[0] if task == "tokenizer" else measured[ratios.index(.5)]
    if task == "tokenizer":
        result.update(uniform_sdf_mae=result["absolute_error_sum"] / result["query_count"],
                      truncated_uniform_sdf_mae=result["truncated_absolute_error_sum"] / result["query_count"],
                      uniform_sign_accuracy=result["sign_correct"] / result["sign_count"] if result["sign_count"] else None)
        from .token_support import summarize_counts
        support = summarize_counts(histogram.cpu().numpy())
        result.update(token_nominal_usage=support["nominal_usage"],token_effective_support=support["effective_support"],
                      token_codes_for_50=support["codes_for_50"])
        metric = "uniform_sdf_mae"
    else:
        metric = "hidden_ce" if categorical else "hidden_bce"
        for ratio,row in zip(ratios,measured):
            row.update(mask_ratio=ratio,token_exact=row["token_correct"]/row["token_count"])
            row[metric] = row["loss_sum"]/row["loss_count" if categorical else "bit_count"]
            if not categorical:
                row["bit_accuracy"] = row["bit_correct"]/row["bit_count"]
        # Copy: result must not recursively contain itself through by_mask_ratio.
        result = dict(result)
        result["by_mask_ratio"] = measured
        result["mean_"+metric] = sum(row[metric] for row in measured)/len(measured)
    if selection_metric is not None:
        allowed = ({"uniform_sdf_mae","truncated_uniform_sdf_mae"} if task == "tokenizer"
                   else {metric,"mean_"+metric})
        if selection_metric not in allowed:
            raise ValueError("Unsupported validation selection metric")
        metric = selection_metric
    result.update(samples=len(dataset), selection_metric=metric, selection_score=result[metric])
    return result
