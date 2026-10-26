# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""The active reconstructive objective, split into sampling and supervision helpers.

Inputs, SDF targets, shell offsets, and thresholds are all in model-space units.
All three reconstructive paths remain in the autograd graph at zero curriculum
weight, so the model is suitable for DDP with find_unused_parameters=False.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import random_rotations, yaw_rotations


BRANCHES = ("raw", "continuous", "token")
PLANE_KEYS = ("raw_planes", "continuous_planes", "quantized_planes")


def linear_ramp(step: int, start: int, duration: int) -> float:
    if step < start:
        return 0.0
    return 1.0 if duration <= 0 else min(1.0, max(0.0, (step - start) / duration))


def loss_scales(specs: dict, step: int) -> dict[str, float]:
    """Reproduce optimizer-step curricula rather than counting data batches."""
    schedule = specs.get("TrainingSchedule", {})
    defaults = {"continuous_sdf": (200, 800), "feature": (200, 800),
                "token_sdf": (500, 1500), "field_distill": (500, 1500), "aux": (200, 800)}
    return {name: linear_ramp(int(step), int(schedule.get(name + "_start_step", start)),
                              int(schedule.get(name + "_ramp_steps", duration)))
            for name, (start, duration) in defaults.items()}


def sdf_weights(target: torch.Tensor, specs: dict) -> torch.Tensor:
    """Near-surface weighting with per-shape inside/outside mass balancing."""
    sigma = max(float(specs.get("NearSurfaceWeightSigma", 0.01)), 1e-6)
    weights = 1 + (float(specs.get("NearSurfaceWeight", 2)) - 1) * torch.exp(-(target.abs() / sigma).square())
    if not specs.get("BalanceInsideOutsideSdfWeights", True):
        return weights
    epsilon = max(float(specs.get("SurfaceSignEps", 0.00025)), 0)
    inside, outside = target < -epsilon, target > epsilon
    signed_count = (inside.float() + outside.float()).sum(1, keepdim=True).clamp_min(1)
    balance = torch.ones_like(target)
    balance = torch.where(inside, signed_count / (2 * inside.float().sum(1, keepdim=True).clamp_min(1)), balance)
    balance = torch.where(outside, signed_count / (2 * outside.float().sum(1, keepdim=True).clamp_min(1)), balance)
    balance = torch.where(~(inside | outside), torch.full_like(balance, float(specs.get("SurfaceSdfWeight", 0.35))), balance)
    return weights * balance


def weighted_sdf_loss(prediction, target, weights, beta=0.003):
    return (F.smooth_l1_loss(prediction, target, reduction="none", beta=beta) * weights).mean()


def sdf_sign_loss(prediction, target, specs):
    """Balanced BCE on inside logits, ignoring the numerical surface band."""
    valid = target.abs() > float(specs.get("SurfaceSignEps", 0.00025))
    if not valid.any():
        return prediction.sum() * 0
    inside = (target[valid] < 0).to(prediction.dtype)
    logits = -prediction[valid] / max(float(specs.get("SignLossTemperature", 0.005)), 1e-6)
    losses = F.binary_cross_entropy_with_logits(logits, inside, reduction="none")
    count = float(inside.numel())
    balance = torch.where(inside > 0.5, count / (2 * inside.sum().clamp_min(1)),
                          count / (2 * (1 - inside).sum().clamp_min(1)))
    return (losses * balance).mean()


def free_space_margin_loss(prediction, target, specs):
    """Squared positive-distance margin at exterior queries only."""
    settings = specs.get("FreeSpace", {})
    outside = target > float(settings.get("eps", 0.001))
    if not settings.get("enabled", False) or not outside.any():
        return prediction.sum() * 0
    losses = F.relu(float(settings.get("margin", 0.004)) - prediction[outside]).square()
    near = target[outside] < float(settings.get("near_band", 0.04))
    weights = torch.where(near, torch.full_like(losses, float(settings.get("near_weight", 2))),
                          torch.ones_like(losses))
    return (losses * weights).sum() / weights.sum().clamp_min(1)


def build_surface_shell(points, normals, specs):
    """Surface anchors plus signed offsets along normalized, repaired normals."""
    settings = specs.get("SurfaceShell", {})
    count = min(int(settings.get("points", 0)), points.shape[1])
    epsilon = float(settings.get("eps", 0.004))
    if not settings.get("enabled", False) or count <= 0 or epsilon <= 0 or normals is None:
        return None
    lengths = normals.norm(dim=-1, keepdim=True)
    valid = lengths.squeeze(-1) > 1e-6
    if not valid.any():
        return None
    normals = normals / lengths.clamp_min(1e-6)
    if points.shape[1] > count:
        indices = torch.randperm(points.shape[1], device=points.device)[:count]
        points, normals, valid = points[:, indices], normals[:, indices], valid[:, indices]
    queries = torch.cat([points, points - epsilon * normals, points + epsilon * normals], dim=1)
    shape = points.shape[:2]
    target = torch.cat([points.new_zeros(shape), points.new_full(shape, -epsilon),
                        points.new_full(shape, epsilon)], dim=1)
    weights = torch.cat([points.new_full(shape, float(settings.get("zero_weight", 1))),
                         points.new_full(shape, float(settings.get("signed_weight", 1))),
                         points.new_full(shape, float(settings.get("signed_weight", 1)))], dim=1)
    return queries, target, torch.where(torch.cat([valid] * 3, dim=1), weights, torch.zeros_like(weights))


def surface_shell_loss(prediction, target, weights):
    loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.002)
    return (loss * weights).sum() / weights.sum().clamp_min(1)


def field_supervision(prediction, sdf, specs):
    """The signed field losses shared by regular and uniform-field queries."""
    truncation = float(specs.get("SdfTruncation", 0.1))
    return {
        "sdf": weighted_sdf_loss(prediction, sdf.clamp(-truncation, truncation),
                                 sdf_weights(sdf, specs), float(specs.get("SdfLossBeta", 0.003))),
        "sign": sdf_sign_loss(prediction, sdf, specs),
        "free_space": free_space_margin_loss(prediction, sdf, specs),
    }


def reconstructive_terms(model, encoded, query_points, query_sdf,
                         normals=None, uniform_points=None, uniform_sdf=None):
    """Evaluate raw, continuous, and quantized fields before applying any weights."""
    specs = model.specs
    rotation = encoded["rotation"]
    queries = query_points @ rotation
    planes = [encoded[name] for name in PLANE_KEYS]
    predictions = [model.decode(features, queries) for features in planes]
    terms = {}
    metrics = {}
    for branch, prediction in zip(BRANCHES, predictions):
        terms.update({branch + "_" + name: value for name, value in field_supervision(prediction, query_sdf, specs).items()})
        metrics[branch + "_sign_acc"] = ((prediction < 0) == (query_sdf < 0)).float().mean()
        metrics[branch + "_inside_ratio"] = (prediction < 0).float().mean()
        metrics[branch + "_mae"] = (prediction - query_sdf.clamp(-float(specs.get("SdfTruncation", 0.1)),
                                                                  float(specs.get("SdfTruncation", 0.1)))).abs().mean()
    reference = encoded["raw_planes"].detach()
    terms["feature_continuous"] = F.smooth_l1_loss(planes[1], reference)
    terms["feature_token"] = F.smooth_l1_loss(planes[2], reference)
    terms["field_distill"] = (F.smooth_l1_loss(predictions[1], predictions[0].detach())
                              + F.smooth_l1_loss(predictions[2], predictions[0].detach()))
    terms["aux"] = encoded["aux_loss"]
    if uniform_points is not None and uniform_sdf is not None:
        uniform_queries = uniform_points @ rotation
        for branch, features in zip(BRANCHES, planes):
            prediction = model.decode(features, uniform_queries)
            terms.update({branch + "_uniform_" + name: value
                          for name, value in field_supervision(prediction, uniform_sdf, specs).items()})
    shell = build_surface_shell(encoded["representative"],
                                normals @ rotation if normals is not None else None, specs)
    if shell is not None:
        shell_queries, target, weights = shell
        for branch, features in zip(BRANCHES, planes):
            terms[branch + "_surface_shell"] = surface_shell_loss(model.decode(features, shell_queries), target, weights)
    return terms, metrics


def weighted_reconstruction(terms: dict, specs: dict, scales: dict):
    """Apply exact configured weights and branch-specific curriculum factors."""
    defaults = {"raw_sdf": 150, "continuous_sdf": 75, "token_sdf": 150,
                "feature_continuous": 5, "feature_token": 5, "field_distill": 2, "aux": 1}
    weights = specs.get("LossWeights", {})
    total = next(iter(terms.values())).new_zeros(())
    for name, value in terms.items():
        scale = 1.0
        if name.startswith("continuous_"):
            scale = scales["continuous_sdf"]
        elif name.startswith("token_"):
            scale = scales["token_sdf"]
        elif name.startswith("feature_"):
            scale = scales["feature"]
        elif name in ("field_distill", "aux"):
            scale = scales[name]
        total = total + float(weights.get(name, defaults.get(name, 0))) * scale * value
    return total


def symmetry_align(model, anchor, rotated, categories):
    """Feature-closest finite yaw match, only for Table and Lamp auxiliaries."""
    specs = model.specs
    if not specs.get("EnableSymmetryAwarePose", True) or categories is None:
        return rotated
    order_map = {"04379243": 4, "03636649": 8}
    overrides = specs.get("SymmetryOrders", {})
    for category in order_map:
        order_map[category] = int(overrides.get(category, order_map[category]))
    aliases = {"table": "04379243", "lamp": "03636649"}
    orders = [order_map.get(aliases.get(str(name).lower(), str(name)), 1) for name in categories]
    if len(orders) != anchor["raw_planes"].shape[0]:
        raise ValueError("category must contain one string per batch item")
    result = {key: value.clone() for key, value in rotated.items()}
    for order in sorted(set(orders)):
        if order <= 1:
            continue
        mask = torch.tensor([value == order for value in orders], device=rotated["representative"].device)
        cloud = rotated["representative"][mask]
        rotations = yaw_rotations(order, device=cloud.device, dtype=cloud.dtype)
        candidates = cloud[:, None] @ rotations[None]
        features = model.pointnet(candidates.reshape(-1, cloud.shape[1], 3))
        features = features.reshape(cloud.shape[0], order, *features.shape[1:])
        differences = features - anchor["raw_planes"][mask, None]
        best = differences.flatten(2).square().mean(-1).argmin(1)
        rows = torch.arange(cloud.shape[0], device=cloud.device)
        result["raw_planes"][mask] = features[rows, best]
        result["rotation"][mask] = rotated["rotation"][mask] @ rotations[best]
        result["representative"][mask] = cloud @ rotations[best]
    return result


def pca_loss(points):
    centered = points - points.mean(dim=1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered / max(points.shape[1] - 1, 1)
    diagonal = covariance.diagonal(dim1=-2, dim2=-1)
    return (covariance.square().sum(dim=(1, 2)) - diagonal.square().sum(dim=1)).mean()


def canonical_geometry_loss(anchor, other, max_points=1024):
    """Unsquared symmetric Chamfer auxiliary; unrelated to evaluation CD scaling."""
    if anchor.shape[1] > max_points:
        indices = torch.randperm(anchor.shape[1], device=anchor.device)[:max_points]
        anchor, other = anchor[:, indices], other[:, indices]
    distances = torch.cdist(anchor, other)
    forward, backward = distances.min(dim=2).values, distances.min(dim=1).values
    loss = forward.mean() + backward.mean()
    return loss, {"canon_cd_l1": loss,
                  "canon_nn_p95": torch.maximum(forward.detach().quantile(0.95, dim=1).mean(),
                                                  backward.detach().quantile(0.95, dim=1).mean())}


def tokenizer_objective(model, batch: dict, step: int = 0):
    """Reconstruct the anchor and propagate its cached geometric frame to a Siamese view.

    batch keys: points, query_points, query_sdf, frame; optional normals,
    uniform_points/uniform_sdf, category. The frame MUST precede point sampling.
    LFQ eval mode supplies zero auxiliary loss without entropy collectives.
    """
    specs = model.specs
    if "frame" not in batch:
        raise ValueError("Training requires batch['frame'] computed from the full cloud before sampling")
    if str(specs.get("SignLossType", "bce")).lower() != "bce":
        raise ValueError("The selected objective uses balanced BCE sign supervision")
    points = batch["points"]
    clean = model.encode(points, batch["frame"])
    random = random_rotations(points.shape[0], points.device, points.dtype)
    rotated = model.representative_features(points @ random.transpose(1, 2), random @ clean["rotation"])
    aligned = symmetry_align(model, clean, rotated, batch.get("category"))
    normals = batch.get("normals")
    if normals is not None:
        normals = F.normalize(normals.to(points.device), dim=-1, eps=1e-6)
    scales = loss_scales(specs, step)
    terms, debug = reconstructive_terms(model, clean, batch["query_points"], batch["query_sdf"], normals,
                                       batch.get("uniform_points"), batch.get("uniform_sdf"))
    clean_loss = weighted_reconstruction(terms, specs, scales)
    if not specs.get("TrainCleanBranchReconstruction", True):
        raise ValueError("The active objective reconstructs the clean representative")
    loss = float(specs.get("CleanBranchLossWeight", 1)) * clean_loss
    metrics = {"loss/clean_" + name: value for name, value in terms.items()}
    metrics.update({"debug/" + name: value for name, value in debug.items()})
    if specs.get("TrainTargetBranchReconstruction", False):
        token_input = aligned["raw_planes"]
        if specs.get("Tokenizer", {}).get("detach_input", True):
            token_input = token_input.detach()
        target = {**aligned, **model.tokenizer(token_input)}
        rotated_normals = normals @ random.transpose(1, 2) if normals is not None else None
        uniform = batch.get("uniform_points")
        target_terms, _ = reconstructive_terms(
            model, target, batch["query_points"] @ random.transpose(1, 2), batch["query_sdf"],
            rotated_normals, uniform @ random.transpose(1, 2) if uniform is not None else None,
            batch.get("uniform_sdf"))
        loss = loss + weighted_reconstruction(target_terms, specs, scales)
        metrics.update({"loss/target_" + name: value for name, value in target_terms.items()})
    geometry, geometry_metrics = canonical_geometry_loss(clean["representative"], aligned["representative"])
    auxiliaries = {
        "pose": F.mse_loss(clean["raw_planes"], aligned["raw_planes"]),
        "canonical_pc": geometry,
        "pca": pca_loss(clean["representative"]) + pca_loss(aligned["representative"]),
        "raw_feature_l2": aligned["raw_planes"].square().mean(),
    }
    defaults = {"pose": 10, "canonical_pc": 20, "pca": 0.02, "raw_feature_l2": 0.0001}
    for name, value in auxiliaries.items():
        loss = loss + float(specs.get("LossWeights", {}).get(name, defaults[name])) * value
        metrics["loss/" + name] = value
    metrics.update({"debug/" + name: value for name, value in geometry_metrics.items()})
    metrics.update({"schedule/" + name: points.new_tensor(value) for name, value in scales.items()})
    metrics["loss"] = loss
    metrics["debug/code_count"] = points.new_tensor(clean["indices"].unique().numel())
    metrics["debug/code_fraction"] = metrics["debug/code_count"] / clean["indices"].numel()
    metrics["debug/gt_inside_ratio"] = (batch["query_sdf"] < 0).float().mean()
    return loss, {key: value.detach() for key, value in metrics.items()}
