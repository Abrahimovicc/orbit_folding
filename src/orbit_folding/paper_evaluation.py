# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Fixed Bernoulli-mask token-prior and LFQ-state evaluation used by the paper."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoints import load_model, verify_prior_binding
from .provenance import dataset_fingerprint, file_sha256
from .token_cache import TokenDataset, tokenizer_hash


def _token_source(token_root, indices, metadata_path, split, num_bits, expected_hash, allow_unverified,
                  *, categorical=False, codebook_size=None):
    codebook_size = codebook_size if codebook_size is not None else 1 << num_bits
    if (token_root is None) == (indices is None):
        raise ValueError("Choose exactly one release token cache or external index array")
    if token_root is not None:
        dataset = TokenDataset(token_root, split)
        if (dataset.codebook_size != codebook_size or dataset.tokenizer_hash != expected_hash
                or (not categorical and dataset.index_cache)):
            raise ValueError("Token cache and prior use different tokenizer identities or bit dimensions")
        provenance = {"kind": "release_cache", "fingerprint": dataset_fingerprint(token_root),
                      "tokenizer_binding_verified": True}

        def batch(selected):
            key = "indices" if dataset.index_cache else "bits"
            values = torch.stack([dataset[int(index)][key] for index in selected])
            if categorical:
                from .models.categorical import batch_indices
                return batch_indices({key: values}, codebook_size)
            return values

        return len(dataset), batch, provenance
    if metadata_path is None:
        raise ValueError("External indices require their metadata JSON")
    metadata = json.loads(Path(metadata_path).read_text())
    array = np.load(indices, mmap_mode="r", allow_pickle=False)
    if array.ndim == 2 and math.isqrt(array.shape[1]) ** 2 == array.shape[1]:
        array = array.reshape(len(array), math.isqrt(array.shape[1]), math.isqrt(array.shape[1]))
    if (array.ndim != 3 or min(array.shape) < 1 or array.dtype.kind not in "iu"
            or array.min() < 0 or array.max() >= codebook_size):
        raise ValueError("External token indices must be nonempty integer NHW arrays in vocabulary range")
    if (metadata.get("num_samples") != len(array) or metadata.get("codebook_size") != codebook_size
            or tuple(metadata.get("token_shape", ())) != array.shape[1:]):
        raise ValueError("External token metadata does not match index dimensions")
    if split != "all" and metadata.get("split") != split:
        raise ValueError("External token metadata does not identify the requested split")
    identities = metadata.get("rel_paths", metadata.get("sample_ids"))
    if identities is not None and (len(identities) != len(array) or len(set(identities)) != len(array)):
        raise ValueError("External sample identities must be unique and match array length")
    declared_hash = metadata.get("tokenizer_hash")
    if declared_hash is not None and declared_hash != expected_hash:
        raise ValueError("External token metadata identifies another tokenizer")
    if declared_hash is None and not allow_unverified:
        raise ValueError("External tokens have no tokenizer hash; explicitly allow unverified indices only for a trusted legacy source")
    shifts = None if categorical else np.arange(num_bits - 1, -1, -1, dtype=np.int64)

    def batch(selected):
        values = np.asarray(array[selected], dtype=np.int64)
        if categorical:
            return torch.from_numpy(values.copy())
        bits = ((values[:, None] >> shifts[None, :, None, None]) & 1).astype(np.float32)
        return torch.from_numpy(bits)

    provenance = {"kind": "external_indices", "indices_sha256": file_sha256(indices),
                  "metadata_sha256": file_sha256(metadata_path), "tokenizer_binding_verified": declared_hash is not None,
                  "categories": dict(Counter(metadata.get("synset_ids", [])))}
    return len(array), batch, provenance


def _state_sums(probabilities, target, hidden, threshold, *, as_tensors=False):
    soft, truth = 2 * probabilities - 1, 2 * target - 1
    hard = 2 * (probabilities >= 0.5).float() - 1
    squared_soft, squared_hard = (soft - truth).square(), (hard - truth).square()
    transitions = torch.zeros_like(hidden)
    transitions[..., 1:] |= truth[..., 1:] != truth[..., :-1]
    transitions[..., :-1] |= truth[..., 1:] != truth[..., :-1]
    transitions[..., 1:, :] |= truth[..., 1:, :] != truth[..., :-1, :]
    transitions[..., :-1, :] |= truth[..., 1:, :] != truth[..., :-1, :]
    selections = {"uncertain": hidden & (soft.abs() <= threshold),
                  "hard_wrong": hidden & (hard != truth), "thin_proxy": hidden & transitions}
    result = {}
    for name, selected in selections.items():
        result.update({name + "_count": selected.sum(),
                       name + "_soft_mse_sum": squared_soft[selected].sum(),
                       name + "_hard_mse_sum": squared_hard[selected].sum()})
    soft_full, hard_full = torch.where(hidden, soft, truth), torch.where(hidden, hard, truth)
    cell_mask = hidden.any(1, keepdim=True)
    edge_soft, edge_hard, edge_count = 0.0, 0.0, 0
    for dimension in (2, 3):
        before, after = [slice(None)] * 4, [slice(None)] * 4
        before[dimension], after[dimension] = slice(None, -1), slice(1, None)
        before, after = tuple(before), tuple(after)
        selected = (cell_mask[before] | cell_mask[after]).expand_as(truth[before])
        difference = truth[after] - truth[before]
        edge_soft = edge_soft + ((soft_full[after] - soft_full[before] - difference).square())[selected].sum()
        edge_hard = edge_hard + ((hard_full[after] - hard_full[before] - difference).square())[selected].sum()
        edge_count = edge_count + selected.sum()
    result.update(edge_soft_mse_sum=edge_soft, edge_hard_mse_sum=edge_hard, edge_count=edge_count)
    return result if as_tensors else {key: int(value) if key.endswith("_count") else float(value)
                                      for key, value in result.items()}


def _host_totals(totals):
    keys = list(totals)
    values = torch.stack([value.double() for value in totals.values()]).cpu().tolist()
    return {key: int(value) if key.endswith(("_count", "_correct")) else value for key, value in zip(keys, values)}


@torch.random.fork_rng(devices=[])
@torch.no_grad()
def evaluate_token_prior(prior_checkpoint, *, token_root=None, indices=None, indices_metadata=None,
                         tokenizer_checkpoint=None, split="test", output="outputs/token_prior",
                         device="cpu", mask_device="cpu", seed=20260511, batch_size=None,
                         ratios=(0.1, 0.25, 0.5, 0.75, 0.9), max_samples=None,
                         uncertain_threshold=0.9, allow_unverified_indices=False):
    from .evaluation import _limit, _positive, _write_report

    ratios = tuple(float(ratio) for ratio in ratios)
    if not ratios or len(set(ratios)) != len(ratios) or any(not math.isfinite(ratio) or not 0 <= ratio <= 1 for ratio in ratios):
        raise ValueError("Mask ratios must be distinct finite values in [0,1]")
    if not math.isfinite(uncertain_threshold) or not 0 <= uncertain_threshold <= 1:
        raise ValueError("uncertain_threshold must be in [0,1]")
    prior, config = load_model(prior_checkpoint, device)
    if config["Task"] != "prior":
        raise ValueError("Token-prior evaluation requires a prior checkpoint")
    expected_hash = prior.checkpoint_metadata.get("tokenizer_hash")
    if tokenizer_checkpoint is not None:
        tokenizer, tokenizer_config = load_model(tokenizer_checkpoint)
        if tokenizer_config["Task"] != "tokenizer":
            raise ValueError("Expected a tokenizer checkpoint")
        expected_hash = tokenizer_hash(tokenizer)
    if expected_hash is None:
        raise ValueError("The prior must have a verified tokenizer binding")
    verify_prior_binding(prior, expected_hash)
    categorical = prior.prior_type == "categorical"
    batch_size = _positive((1 if categorical else 16) if batch_size is None else batch_size, "batch_size")
    count, load_batch, source = _token_source(token_root, indices, indices_metadata, split,
        getattr(prior, "num_bits", None), expected_hash, allow_unverified_indices,
        categorical=categorical, codebook_size=prior.codebook_size)
    order = np.arange(count)
    np.random.default_rng(seed).shuffle(order)
    order = order[:_limit(count, max_samples)]
    mask_device = torch.device(mask_device)
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    rows = []
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        for ratio in ratios:
            generator = torch.Generator(device=mask_device).manual_seed(seed + round(ratio * 10000))
            totals = {key: torch.zeros((), dtype=torch.float64, device=device) for key in
                      ("loss_sum", "bit_correct", "bit_count", "token_correct", "token_count", "confidence_sum")}
            mask_digest = hashlib.sha256()
            for start in range(0, len(order), batch_size):
                bits = load_batch(order[start:start + batch_size]).to(device)
                batch, height, width = bits.shape[0], *bits.shape[-2:]
                cells = (torch.rand((batch, height, width), generator=generator, device=mask_device) < ratio)[:, None]
                mask_digest.update(cells.cpu().numpy().tobytes())
                cells = cells.to(device)
                if categorical:
                    selected = cells[:, 0]
                    logits = prior.model(torch.where(selected, prior.model.mask_token_id, bits))
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Nonfinite categorical logits in token evaluation")
                    totals["loss_sum"] += F.cross_entropy(logits[selected], bits[selected], reduction="sum")
                    totals["token_correct"] += ((logits.argmax(-1) == bits) & selected).sum()
                    totals["token_count"] += selected.sum()
                    confidence = torch.exp(logits.amax(-1) - logits.logsumexp(-1))
                    totals["confidence_sum"] += confidence[selected].sum()
                    continue
                hidden = cells.expand_as(bits)
                logits = prior.model(torch.where(hidden, 0.5, bits), (~hidden).float())
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite prior logits in token evaluation")
                probabilities = logits.sigmoid()
                correct = (probabilities >= 0.5) == bits.bool()
                totals["loss_sum"] += F.binary_cross_entropy_with_logits(logits[hidden], bits[hidden], reduction="sum")
                totals["bit_correct"] += (correct & hidden).sum()
                totals["bit_count"] += hidden.sum()
                totals["token_correct"] += (correct.all(1, keepdim=True) & cells).sum()
                totals["token_count"] += cells.sum()
                totals["confidence_sum"] += (2 * (probabilities - 0.5).abs())[hidden].sum()
                for key, value in _state_sums(probabilities, bits, hidden, uncertain_threshold, as_tensors=True).items():
                    if key not in totals:
                        totals[key] = torch.zeros((), dtype=torch.float64, device=device)
                    totals[key] += value
            totals = _host_totals(totals)
            bits_count, cells_count = totals["bit_count"], totals["token_count"]
            if categorical:
                rows.append({"mask_ratio": ratio, "loss_sum": totals["loss_sum"],
                    "token_correct": totals["token_correct"], "token_count": cells_count,
                    "mask_sha256": mask_digest.hexdigest(),
                    "ce": totals["loss_sum"] / cells_count if cells_count else None,
                    "token_exact": totals["token_correct"] / cells_count if cells_count else None,
                    "confidence": totals["confidence_sum"] / cells_count if cells_count else None})
                continue
            row = {"mask_ratio": ratio, **totals, "mask_sha256": mask_digest.hexdigest(),
                   "bce": totals["loss_sum"] / bits_count if bits_count else None,
                   "bit_accuracy": totals["bit_correct"] / bits_count if bits_count else None,
                   "token_exact": totals["token_correct"] / cells_count if cells_count else None,
                   "confidence": totals["confidence_sum"] / bits_count if bits_count else None}
            for name in ("uncertain", "hard_wrong", "thin_proxy", "edge"):
                soft, hard = totals[name + "_soft_mse_sum"], totals[name + "_hard_mse_sum"]
                if name != "edge":
                    row[name + "_fraction"] = totals[name + "_count"] / bits_count if bits_count else None
                row[name + "_mse_reduction"] = 1 - soft / hard if hard > 1e-12 else None
            rows.append(row)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
    report = {"evaluation": "token_prior", "protocol": "paper-fixed-bernoulli-v1",
              "prior_type": prior.prior_type,
              "source": source, "split": split, "samples": len(order), "seed": seed,
              "batch_size": batch_size, "device": str(device), "mask_device": str(mask_device),
              "torch": str(torch.__version__), "uncertain_threshold": uncertain_threshold,
              "prior_sha256": file_sha256(prior_checkpoint), "tokenizer_hash": expected_hash,
              "selection_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(), "rows": rows,
              "scope": ["Inference with a supplied prior; not training from scratch or a new generation benchmark.",
                        "The selected explicit split is shuffled before masks, matching the paper's full-token-array protocol. No implicit train/validation split is created.",
                        "Bernoulli masks use a local generator seeded by seed + round(ratio*10000). RNG device and batch size are recorded; CPU masks reproduce the original fixed-mask run.",
                        "Thin/edge values are LFQ-state transition diagnostics, not decoded mesh edge distances. Missing error support yields null reduction."]}
    if categorical:
        report["scope"].append("Categorical rows report hidden-index CE, exact-token accuracy and maximum class probability; no LFQ bit or relaxed-state metrics are defined.")
    return _write_report(output, "token_prior", report)
