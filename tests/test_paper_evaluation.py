# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Paper-protocol masks, pooled counts, legacy provenance and state-space errors."""

import json
import math
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from orbit_folding.checkpoints import save_checkpoint
from orbit_folding.data import geometry_dataset
from orbit_folding.paper_evaluation import _state_sums, evaluate_token_prior
from orbit_folding.token_cache import export_tokens, tokenizer_hash
from orbit_folding.training import build_model


@pytest.fixture
def paper_setup(tmp_path, geometry_root, tokenizer_config, prior_config):
    tokenizer = build_model(tokenizer_config)
    cache = tmp_path / "tokens"
    export_tokens(tokenizer, DataLoader(geometry_dataset(geometry_root, tokenizer_config, "all")), cache, "cpu")
    prior = build_model(prior_config)
    for parameter in prior.parameters():
        torch.nn.init.zeros_(parameter)
    path = tmp_path / "prior.pt"
    save_checkpoint(path, prior, prior_config, extra={"tokenizer_hash": tokenizer_hash(tokenizer)})
    indices = tmp_path / "indices.npy"
    np.save(indices, np.random.default_rng(2).integers(0, 16, (5, 8, 8), dtype=np.uint16))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"num_samples": 5, "codebook_size": 16, "token_shape": [8, 8],
        "sample_ids": [f"sample_{index}" for index in range(5)], "split": "test"}))
    return {"prior_checkpoint": path, "indices": indices, "indices_metadata": metadata,
            "allow_unverified_indices": True, "output": tmp_path / "report.json"}, cache


def test_paper_eval_matches_analytical_zero_logits_and_local_rng(paper_setup):
    options, _ = paper_setup
    state = torch.get_rng_state().clone()
    report = evaluate_token_prior(**options, ratios=[0.5], batch_size=2, seed=12)
    assert torch.equal(state, torch.get_rng_state())
    row = report["rows"][0]
    assert row["bce"] == pytest.approx(math.log(2), rel=1e-6)
    assert row["confidence"] == 0 and row["uncertain_fraction"] == 1
    assert row["hard_wrong_mse_reduction"] == pytest.approx(0.75)
    assert row["bit_count"] == row["token_count"] * 4
    assert row["bit_accuracy"] == row["bit_correct"] / row["bit_count"]
    assert report["source"]["tokenizer_binding_verified"] is False
    assert "edge_fraction" not in row


def test_cache_source_verifies_binding_and_preserves_explicit_split(paper_setup):
    options, cache = paper_setup
    report = evaluate_token_prior(options["prior_checkpoint"], token_root=cache, output=options["output"], ratios=[0.5])
    assert report["samples"] == 1
    assert report["source"]["tokenizer_binding_verified"] is True


def test_cpu_paper_masks_match_across_batch_sizes(paper_setup):
    options, _ = paper_setup
    first = evaluate_token_prior(**options, ratios=[0.5], batch_size=1)
    second = evaluate_token_prior(**options, ratios=[0.5], batch_size=4)
    assert first["selection_sha256"] == second["selection_sha256"]
    assert first["rows"][0]["mask_sha256"] == second["rows"][0]["mask_sha256"]
    assert first["rows"][0]["bit_correct"] == second["rows"][0]["bit_correct"]


@pytest.mark.parametrize("ratio", [0, 1])
def test_extreme_mask_counts_are_explicit(paper_setup, ratio):
    options, _ = paper_setup
    report = evaluate_token_prior(**options, ratios=[ratio])
    row = report["rows"][0]
    assert row["token_count"] == ratio * 5 * 8 * 8
    if ratio == 0:
        assert row["bit_accuracy"] is None and row["uncertain_mse_reduction"] is None


@pytest.mark.parametrize("ratios", [[], [0.5, 0.5], [-0.1], [float("nan")], [1.1]])
def test_invalid_ratios_are_rejected(paper_setup, ratios):
    with pytest.raises(ValueError, match="ratios"):
        evaluate_token_prior(**paper_setup[0], ratios=ratios)


def test_legacy_indices_require_an_explicit_override(paper_setup):
    options, _ = paper_setup
    options["allow_unverified_indices"] = False
    with pytest.raises(ValueError, match="no tokenizer hash"):
        evaluate_token_prior(**options)


@pytest.mark.parametrize("field,value", [("codebook_size", 32), ("num_samples", 4),
    ("split", "train"), ("tokenizer_hash", "a" * 64), ("sample_ids", ["duplicate"] * 5)])
def test_legacy_metadata_mismatches_cannot_be_overridden(paper_setup, field, value):
    options, _ = paper_setup
    path = options["indices_metadata"]
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        evaluate_token_prior(**options)


@pytest.mark.parametrize("value,dtype", [(-1, np.int16), (16, np.uint16), (1.9, np.float32)])
def test_invalid_external_indices_are_never_coerced(paper_setup, value, dtype):
    options, _ = paper_setup
    np.save(options["indices"], np.full((5, 8, 8), value, dtype=dtype))
    with pytest.raises(ValueError, match="integer NHW"):
        evaluate_token_prior(**options)


def test_state_sums_wrong_bits_and_observed_edges():
    probabilities = torch.full((1, 1, 2, 2), 0.5)
    target = torch.zeros_like(probabilities)
    hidden = torch.ones_like(target, dtype=torch.bool)
    result = _state_sums(probabilities, target, hidden, .9)
    assert result["hard_wrong_count"] == 4
    assert result["hard_wrong_soft_mse_sum"] == 4
    assert result["hard_wrong_hard_mse_sum"] == 16
    assert result["edge_soft_mse_sum"] == result["edge_hard_mse_sum"] == 0
    assert result["edge_count"] == 4


def test_paper_token_cli(paper_setup, run_cli):
    options, _ = paper_setup
    result = run_cli("evaluate", "tokens", "--prior", options["prior_checkpoint"],
        "--indices", options["indices"], "--indices-metadata", options["indices_metadata"],
        "--allow-unverified-indices", "--output", options["output"], "--ratios", .5, "--threads", 1)
    assert json.loads(result.stdout)["samples"] == 5


def _paper_checker():
    path = Path(__file__).resolve().parents[1] / "tools/check_paper_reproduction.py"
    spec = importlib.util.spec_from_file_location("paper_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_report


def test_paper_checker_accepts_rounded_metrics_but_requires_exact_counts():
    reference = {"format": "orbit_folding.paper_reference.v1", "protocol": "fixed",
        "samples": 512, "seed": 1, "batch_size": 16, "mask_device": "cpu",
        "rows": [{"mask_ratio": .5, "masked_cells": 10, "bit_accuracy_percent": "97.23",
                  "uncertain_mse_reduction_percent": "30.3"}]}
    report = {key: value for key, value in reference.items() if key not in {"format", "rows"}}
    report["rows"] = [{"mask_ratio": .5, "token_count": 10, "bit_accuracy": .97229124,
                       "uncertain_mse_reduction": .3034}]
    check = _paper_checker()
    assert check(report, reference)["matches_reported_precision"]
    report["rows"][0]["token_count"] = 11
    assert not check(report, reference)["matches_reported_precision"]


@pytest.mark.parametrize("rows", [[], [{"mask_ratio": .5, "bit_accuracy": float("nan")}],
    [{"mask_ratio": .5, "bit_accuracy": .5}]])
def test_paper_checker_rejects_missing_nonfinite_or_incorrect_metrics(rows):
    reference = {"format": "orbit_folding.paper_reference.v1", "protocol": "fixed", "samples": 512,
        "seed": 1, "batch_size": 16, "mask_device": "cpu",
        "rows": [{"mask_ratio": .5, "bit_accuracy_percent": "97.23"}]}
    report = {key: value for key, value in reference.items() if key not in {"format", "rows"}}
    report["rows"] = rows
    assert not _paper_checker()(report, reference)["matches_reported_precision"]
