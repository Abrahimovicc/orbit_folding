# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Optional distance and posterior calibration without representation retraining."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import torch
from torch.nn import functional as F

from .checkpoints import load_checkpoint, load_model, save_checkpoint, verify_prior_binding
from .data import geometry_dataset
from .evaluation import _positive
from .geometry import representative_frame
from .models.prior import fixed_spatial_mask
from .objectives import sdf_weights
from .provenance import dataset_fingerprint, file_sha256
from .token_cache import TokenDataset


def fit_temperature(logits, targets, bounds=(.5, 3.)):
    """Fit one positive logit temperature by hidden-bit BCE, including identity."""
    lo, hi = map(float, bounds)
    if not np.isfinite([lo, hi]).all() or not 0 < lo <= 1 <= hi or lo == hi:
        raise ValueError("Temperature bounds must be finite, increasing, positive and contain one")
    if (logits.shape != targets.shape or not logits.numel()
            or not torch.isfinite(logits).all() or not torch.isfinite(targets).all()
            or not torch.all((targets == 0) | (targets == 1))):
        raise ValueError("Expected matching nonempty finite logits and binary targets")
    logits, targets = logits.detach().float(), targets.detach().float()
    if not torch.isfinite(logits).all():
        raise ValueError("Logits must remain finite in float32")
    before = float(F.binary_cross_entropy_with_logits(logits, targets))
    result = minimize_scalar(lambda log_t: float(F.binary_cross_entropy_with_logits(logits / np.exp(log_t), targets)),
        bounds=(np.log(lo), np.log(hi)), method="bounded", options={"xatol": 1e-6})
    fitted = float(np.exp(result.x))
    after = float(F.binary_cross_entropy_with_logits(logits / fitted, targets))
    return {"fitted_temperature": fitted, "selected_temperature": fitted if after < before else 1.,
            "validation_bce_before": before, "validation_bce_after": min(before, after),
            "accepted_by_validation_bce": after < before, "temperature_bounds": [lo, hi]}


@torch.random.fork_rng(devices=[])
@torch.no_grad()
def calibrate_prior(checkpoint, tokens, output, *, device="cpu", mask_ratio=.75,
                    seed=20260511, batch_size=4):
    """Use only the explicit val split; preserve source weights and binding."""
    batch_size = _positive(batch_size, "batch_size")
    if not np.isfinite(mask_ratio) or not 0 < mask_ratio <= 1:
        raise ValueError("mask_ratio must lie in (0,1]")
    output = Path(output)
    if output.suffix != ".pt":
        raise ValueError("Calibrated checkpoint must have .pt extension")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Calibration output already exists")
    source = load_checkpoint(checkpoint)
    if source["config"]["Task"] != "prior":
        raise ValueError("Calibration requires a bit-field prior")
    model, config = load_model(checkpoint, device)
    dataset = TokenDataset(tokens, "val")
    if model.prior_type != "bit_field" or dataset.index_cache or dataset.num_bits != model.num_bits:
        raise ValueError("Calibration requires matching binary LFQ fields and a bit-field prior")
    verify_prior_binding(model, dataset.tokenizer_hash)
    logits, targets = [], []
    for start in range(0, len(dataset), batch_size):
        selected = range(start, min(start + batch_size, len(dataset)))
        values = torch.stack([dataset[index]["bits"] for index in selected]).to(device)
        height, width = values.shape[-2:]
        hidden = torch.cat([fixed_spatial_mask(1, height, width, mask_ratio, seed + index, "cpu")
                            for index in selected]).to(device)
        prediction = model.model(torch.where(hidden, .5, values), ~hidden)
        logits.append(prediction[hidden.expand_as(values)])
        targets.append(values[hidden.expand_as(values)])
    logits, targets = torch.cat(logits), torch.cat(targets)
    fitted = fit_temperature(logits, targets)
    temperature = fitted["selected_temperature"]
    model.model.head.weight.div_(temperature)
    model.model.head.bias.div_(temperature)
    report = {"format": "orbit_folding.prior_calibration.v1", **fitted,
        "source_sha256": file_sha256(checkpoint), "tokenizer_hash": dataset.tokenizer_hash,
        "data_fingerprint": dataset_fingerprint(tokens), "split": "val", "samples": len(dataset),
        "mask_ratio": mask_ratio, "seed": seed, "hidden_bits": logits.numel(),
        "mask_rule": "Exact-count mask keyed by validation sample index.",
        "selection": "Minimum validation hidden-bit BCE including identity temperature; no test selection.",
        "scope": "Optional post-training probability calibration, not unchanged paper reproduction. Positive rescaling preserves logit signs in exact arithmetic; finite-precision threshold ties can differ."}
    save_checkpoint(output, model, config, epoch=source["epoch"], step=source["step"],
        extra={**source["extra"], "calibration": report})
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


@torch.random.fork_rng(devices=[])
@torch.no_grad()
def calibrate_field(checkpoint, data, output, *, device="cpu", weighting="uniform",
                    train_samples=128, observation_points=2048, seed=20260513):
    """Fit the existing shared-head bias on train, with explicit val guards."""
    from .evaluation import _decode

    train_samples = _positive(train_samples, "train_samples")
    observation_points = _positive(observation_points, "observation_points", 3)
    if weighting not in {"uniform", "balanced"}:
        raise ValueError("weighting must be uniform or balanced")
    output = Path(output)
    if output.suffix != ".pt":
        raise ValueError("Calibrated checkpoint must have .pt extension")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Calibration output already exists")
    source = load_checkpoint(checkpoint)
    if source["config"]["Task"] != "tokenizer":
        raise ValueError("Field calibration requires a tokenizer")
    model, config = load_model(checkpoint, device)

    def collect(split, limit=None):
        queries = geometry_dataset(data, config, split, seed=seed)
        clouds = geometry_dataset(data, config, split, seed=seed, points_only=True,
                                  sampling_protocol="reference")
        indices = np.arange(len(queries))
        if limit is not None:
            indices = np.sort(np.random.default_rng(20261026).choice(indices, min(limit, len(indices)), replace=False))
        generator = torch.Generator().manual_seed(seed)
        predictions, targets, uniform_predictions, uniform_targets, ids = [], [], [], [], []
        for index in indices:
            batch, cloud = queries[int(index)], clouds[int(index)]["points"]
            if len(cloud) > observation_points:
                cloud = cloud[torch.randperm(len(cloud), generator=generator)[:observation_points]]
            points = cloud[None].to(device)
            frame = representative_frame(points, lock=bool(config.get("EnableAxisRepresentativeLock", True)))
            planes = model.encode(points, frame=frame)["quantized_planes"]
            predictions.append(_decode(model, planes, batch["query_points"][None].to(device) @ frame, 16384))
            uniform_predictions.append(_decode(model, planes, batch["uniform_points"][None].to(device) @ frame, 16384))
            if not np.isfinite(predictions[-1]).all() or not np.isfinite(uniform_predictions[-1]).all():
                raise FloatingPointError("Nonfinite field during distance calibration")
            targets.append(batch["query_sdf"].numpy())
            uniform_targets.append(batch["uniform_sdf"].numpy())
            ids.append(batch["sample_id"])
        return (*[torch.from_numpy(np.stack(rows)).to(device) for rows in
                  (predictions, targets, uniform_predictions, uniform_targets)], ids)

    predicted, target, uniform, truth, train_ids = collect("train", train_samples)
    truncation, beta = float(config.get("SdfTruncation", .1)), float(config.get("SdfLossBeta", .003))
    weights = sdf_weights(target, config) if weighting == "balanced" else 1.
    uniform_weights = sdf_weights(truth, config) if weighting == "balanced" else 1.

    def objective(offset):
        return float((F.smooth_l1_loss(predicted + offset, target.clamp(-truncation, truncation),
                            beta=beta, reduction="none") * weights).mean()
            + .2 * (F.smooth_l1_loss(uniform + offset, truth.clamp(-truncation, truncation),
                            beta=beta, reduction="none") * uniform_weights).mean())

    result = minimize_scalar(objective, bounds=(-.003, .003), method="bounded", options={"xatol": 1e-8})
    fitted = float(result.x)
    val, labels, uval, ulabels, validation_ids = collect("val")

    def scores(offset):
        current = val + offset
        surface = labels.abs() <= .0005
        return {"truncated_sdf_mae": float((current - labels.clamp(-truncation, truncation)).abs().mean()),
            "uniform_truncated_sdf_mae": float((uval + offset - ulabels.clamp(-truncation, truncation)).abs().mean()),
            "inside_gap_pp": float(((current < 0).float().mean(1) - (labels < 0).float().mean(1)).abs().mean() * 100),
            "sign_accuracy": float(((current < 0) == (labels < 0)).float().mean()),
            "surface_sdf_mae": float(current[surface].abs().mean()) if surface.any() else None}

    before, after = scores(0.), scores(fitted)
    accepted = (after["truncated_sdf_mae"] <= before["truncated_sdf_mae"] * 1.01
        and after["uniform_truncated_sdf_mae"] <= before["uniform_truncated_sdf_mae"] * 1.05
        and after["inside_gap_pp"] <= before["inside_gap_pp"]
        and after["sign_accuracy"] >= before["sign_accuracy"]
        and (before["surface_sdf_mae"] is None or after["surface_sdf_mae"] <= before["surface_sdf_mae"]))
    selected = fitted if accepted else 0.
    model.field_head.sdf_head[-1].bias.add_(selected / model.field_head.output_scale)
    report = {"format": "orbit_folding.field_calibration.v1", "source_sha256": file_sha256(checkpoint),
        "data_fingerprint": dataset_fingerprint(data), "train_ids": train_ids, "validation_ids": validation_ids,
        "train_samples": len(train_ids), "validation_samples": len(validation_ids), "seed": seed,
        "train_subset_seed": 20261026, "observation_points": observation_points, "weighting": weighting,
        "fit_bounds_model_units": [-.003, .003], "fitted_sdf_offset": fitted, "selected_sdf_offset": selected,
        "train_objective_before": objective(0.), "train_objective_after": objective(fitted),
        "validation_before": before, "validation_fitted": after, "accepted_by_validation_guards": accepted,
        "encoder_and_quantizer_unchanged": True, "mesh_smoothing": False,
        "selection": "Train-only Huber fit plus validation guards: query MAE within1%, uniform MAE within5%, no worse sign, inside-gap or surface residual.",
        "scope": "Optional distance calibration of one existing shared-head bias; no test-based selection. Calibrated weights require a new binding and must not be substituted for a prior's original tokenizer."}
    save_checkpoint(output, model, config, epoch=source["epoch"], step=source["step"],
        extra={**source["extra"], "calibration": report})
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
