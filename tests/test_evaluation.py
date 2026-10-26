# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""CPU-only isolated evaluation contracts; analytic fixtures are not quality evidence."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from orbit_folding import checkpoints, evaluation
from orbit_folding.data import GeometryDataset
from orbit_folding.metrics import (aggregate_sdf, chamfer_distance, extract_mesh,
                                  grid_drift, hidden_accuracy, sdf_metrics)
from orbit_folding.models.prior import masked_completion
from orbit_folding.token_cache import export_tokens


class _Tokenizer(nn.Module):
    """Constant positive fields make missing isosurfaces deterministic."""

    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.tensor(0.125))
        self.tokenizer = SimpleNamespace(num_bits=2, codebook_size=4)
        self.quantizer_type = "lfq"
        self.encoded_inputs, self.query_inputs, self.decoded_states = [], [], []

    def encode(self, points, frame=None):
        self.encoded_inputs.append((points.clone(), frame.clone()))
        planes = torch.zeros(points.shape[0], 1, 2, 2, device=points.device)
        bits = torch.tensor([[[[0, 0], [1, 1]], [[0, 1], [0, 1]]]], device=points.device).float()
        return {"rotation": frame, "representative": points @ frame,
                "raw_planes": planes, "continuous_planes": planes + 0.1,
                "quantized_planes": planes + 0.2,
                "bits": bits.expand(points.shape[0], -1, -1, -1)}

    def decode(self, planes, queries):
        self.query_inputs.append(queries.clone())
        return torch.ones_like(queries[..., 0]) * (self.offset + planes.mean())

    def decode_probabilities(self, probabilities):
        self.decoded_states.append(probabilities.clone())
        return probabilities.mean(dim=1, keepdim=True)


class _Prior(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_bits = 2
        self.prior_type, self.codebook_size = "bit_field", 4
        self.observed = []

    def forward(self, bits, observed_mask):
        self.observed.append((bits.clone(), observed_mask.clone()))
        return torch.zeros_like(bits)


@pytest.fixture
def field_options():
    return dict(pc_size=24, samples_per_mesh=12, uniform_samples=7,
                query_chunk_size=4, chunk_size=8, mesh_samples=16)


@pytest.fixture
def completion_setup(geometry_root, tmp_path, monkeypatch):
    tokenizer, prior = _Tokenizer(), _Prior()
    dataset = GeometryDataset(geometry_root, split="all", pc_size=24,
                              samples_per_mesh=1, uniform_samples=1)
    tokens = tmp_path / "tokens"
    export_tokens(tokenizer, DataLoader(dataset, batch_size=1), tokens, "cpu")
    from orbit_folding.token_cache import tokenizer_hash
    prior.checkpoint_metadata = {"tokenizer_hash": tokenizer_hash(tokenizer)}

    def load_model(path, device="cpu"):
        return (prior, {"Task": "prior"}) if path == "prior" else (tokenizer, {"Task": "tokenizer", "PCsize": 24})

    monkeypatch.setattr(checkpoints, "load_model", load_model)
    return tokenizer, prior, tokens, geometry_root


def test_completion_rejects_a_known_cached_frame_policy_mismatch(completion_setup, tmp_path):
    _, _, tokens, geometry = completion_setup
    path = tokens / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["provenance"]["encoder_sampling"]["axis_lock"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frame policies"):
        evaluation.eval_completion("prior", "tokenizer", tokens, geometry, output=tmp_path / "report.json")


def test_chunked_chamfer_matches_paper_half_sum_and_batch_counts():
    first = torch.tensor([[0., 0, 0], [1, 0, 0]], dtype=torch.float64)
    second = torch.tensor([[0., 1, 0], [2, 0, 0], [1, 0, 0]], dtype=torch.float64)
    distances = torch.cdist(first, second).square()
    expected = (distances.amin(1).mean() + distances.amin(0).mean()) / 2
    torch.testing.assert_close(chamfer_distance(first, second, chunk_size=1), expected)
    torch.testing.assert_close(chamfer_distance(first[None].repeat(2, 1, 1),
                                               second[None].repeat(2, 1, 1), 1), expected.repeat(2))


def test_sdf_counts_are_pooled_and_surface_ties_are_excluded():
    first = sdf_metrics([0., -0.1, -0.1], [0., -0.2, 0.1])
    second = sdf_metrics([-0.1], [-0.1])
    result = aggregate_sdf([first, second])
    assert result["query_count"] == 4
    assert result["sdf_mae"] == pytest.approx(0.3 / 4)
    assert result["sign_count"] == 3
    assert result["sign_accuracy"] == pytest.approx(2 / 3)
    assert result["sign_excluded_surface_count"] == 1
    assert first["prediction_support"]["negative_count"] == 2


def test_nonfinite_predictions_never_become_perfect_or_zero_scores():
    result = sdf_metrics([np.nan, 1.], [-1., 1.])
    assert result["sdf_mae"] is None and result["sign_accuracy"] is None
    assert result["query_count"] == 2 and result["nonfinite_count"] == 1
    assert aggregate_sdf([result])["sdf_mae"] is None


def test_hidden_mask_counts_exclude_observations_and_require_whole_tokens():
    target = torch.zeros(1, 2, 2, 2)
    predicted = target.clone()
    predicted[0, 0, 0, 0] = 1
    hidden = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    hidden[0, 0, 0, 0] = True
    result = hidden_accuracy(predicted, target, hidden)
    assert result["bit_count"] == 2 and result["token_count"] == 1
    assert result["hidden_bit_accuracy"] == 0.5
    assert result["hidden_token_accuracy"] == 0
    assert result["observed_bits_unchanged"]
    predicted[0, :, 1, 1] = 1
    assert not hidden_accuracy(predicted, target, hidden)["observed_bits_unchanged"]
    assert hidden_accuracy(target, target, torch.zeros_like(hidden))["hidden_bit_accuracy"] is None
    with pytest.raises(ValueError, match="B1HW"):
        hidden_accuracy(target, target, hidden.expand_as(target))


def test_stability_isolates_frames_using_the_same_clean_reference(geometry_root, tmp_path, monkeypatch):
    rotations = torch.stack([torch.eye(3), torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]])]).double()
    observed_inputs = []

    def rotated_pair(reference, jitter, seed):
        clean = reference[None] @ rotations.transpose(1, 2)
        noise = torch.zeros_like(clean)
        noise[0, :, 0], noise[1, :, 2] = 0.03, -0.02
        return clean, clean + noise, rotations

    def frames(observed, lock=True):
        observed_inputs.append(observed.clone())
        return rotations

    monkeypatch.setattr(evaluation, "_rotation_pair", rotated_pair)
    monkeypatch.setattr(evaluation, "representative_frame", frames)
    result = evaluation.eval_stability(geometry_root, output=tmp_path / "stability.json",
                                        jitter=0.0025, rotation_pairs=2, reference_points=24, chunk_size=8)
    assert result["successful_samples"] == 1 and not result["failed_samples"]
    assert result["samples"][0]["sample_id"] == "generated_5"
    assert result["pca_only_cd_x1e4"]["count"] == 2
    assert result["jitter_clip_bounds_model_units"] == pytest.approx([-0.0075, 0.0075])
    assert result["jitter_clip_bounds_unit_radius"] == pytest.approx([-0.015, 0.015])
    assert result["locked_cd_x1e4"]["p95"] < 1e-20
    assert result["pca_only_cd_x1e4"]["p95"] < 1e-20
    assert len(observed_inputs) == 4
    assert observed_inputs[0].shape == (2, 96, 3)
    folded_noisy = observed_inputs[0] @ rotations
    assert chamfer_distance(folded_noisy[0], folded_noisy[1], 16) > 1e-5


def test_jitter_is_independent_bounded_reproducible_and_restores_rng(monkeypatch):
    reference = torch.arange(96, dtype=torch.float64).reshape(32, 3) / 100
    before = torch.random.get_rng_state().clone()
    clean, observed, _ = evaluation._rotation_pair(reference, 0.0025, 2026)
    noise = observed - clean
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(observed, evaluation._rotation_pair(reference, 0.0025, 2026)[1])
    assert not torch.equal(noise[0], noise[1])
    assert noise.abs().max() <= 0.0075 + 1e-15
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.full_like(value, 100.))
    clean, observed, _ = evaluation._rotation_pair(reference, 0.0025, 2026)
    torch.testing.assert_close(observed - clean, torch.full_like(observed, 0.0075), rtol=0, atol=1e-15)


@pytest.mark.parametrize("invalid", [np.empty((0, 3)), np.zeros((2, 4)), np.full((4, 3), np.nan)])
def test_stability_records_invalid_samples_without_hiding_successes(
        geometry_root, tmp_path, rewrite_npz, invalid):
    rewrite_npz(geometry_root / "generated_5.npz", points=invalid)
    report = evaluation.eval_stability(geometry_root, split="all", output=tmp_path / "stability",
                                        rotation_pairs=1, reference_points=16, chunk_size=8)
    assert report["attempted_samples"] == 6 and report["successful_samples"] == 5
    assert [row["sample_id"] for row in report["failed_samples"]] == ["generated_5"]
    assert report["locked_cd_x1e4"]["count"] == 5
    assert report["jitter_clip_bounds_model_units"] == [0, 0]
    json.loads((tmp_path / "stability" / "stability.json").read_text())


def test_reconstruction_reports_support_and_missing_isosurfaces(
        geometry_root, tmp_path, monkeypatch, field_options):
    tokenizer = _Tokenizer()
    monkeypatch.setattr(checkpoints, "load_model", lambda *args, **kwargs: (tokenizer, {"Task": "tokenizer"}))
    report = evaluation.eval_reconstruction("tokenizer", geometry_root, output=tmp_path / "reconstruction",
                                             resolution=4, **field_options)
    assert report["successful_samples"] == 1 and not report["failed_samples"]
    assert not tokenizer.training
    for path in ("raw", "continuous", "quantized"):
        row = report["paths"][path]
        assert row["query"]["query_count"] == 12 and row["uniform"]["query_count"] == 7
        assert row["mesh_status_counts"] == {"no_isosurface": 1}
        assert row["mesh_valid_count"] == 0
        assert row["cd_x1e4_valid_only"]["count"] == 0
        assert row["cd_x1e4_valid_only"]["mean"] is None
        assert report["samples"][0]["paths"][path]["mesh"]["cd_x1e4"] is None
    batch = GeometryDataset(geometry_root, split="test", pc_size=24,
                             samples_per_mesh=12, uniform_samples=7)[0]
    torch.testing.assert_close(tokenizer.encoded_inputs[0][0], batch["points"][None])
    expected = batch["query_points"][None] @ batch["frame"][None]
    torch.testing.assert_close(torch.cat(tokenizer.query_inputs[:3], dim=1), expected)


def test_reconstruction_rejects_a_prior_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "load_model", lambda *args, **kwargs: (_Prior(), {"Task": "prior"}))
    with pytest.raises(ValueError, match="tokenizer checkpoint"):
        evaluation.eval_reconstruction("prior", tmp_path / "missing", output=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_reconstruction_records_invalid_cached_frame(
        geometry_root, tmp_path, monkeypatch, rewrite_npz, field_options):
    rewrite_npz(geometry_root / "generated_5.npz", frame=np.eye(3, dtype=np.float32) * 2)
    monkeypatch.setattr(checkpoints, "load_model", lambda *args, **kwargs: (_Tokenizer(), {"Task": "tokenizer"}))
    result = evaluation.eval_reconstruction("tokenizer", geometry_root, output=tmp_path / "out",
                                            resolution=4, **field_options)
    assert result["successful_samples"] == 0
    error = result["failed_samples"][0]["error"]
    assert "proper" in error and "rotation" in error


def test_completion_uses_one_mask_and_preserves_observed_bits(completion_setup, tmp_path, field_options):
    tokenizer, prior, tokens, geometry = completion_setup
    report = evaluation.eval_completion("prior", "tokenizer", tokens, geometry,
                                         output=tmp_path / "completion", resolution=4,
                                         mask_ratio=0.5, **field_options)
    assert report["successful_samples"] == 1, report["failed_samples"]
    assert report["hidden_accuracy"]["bit_count"] == 4
    assert report["hidden_accuracy"]["token_count"] == 2
    assert len(prior.observed) == 1 and len(tokenizer.decoded_states) == 3
    hard, relaxed, oracle = tokenizer.decoded_states
    observed = prior.observed[0][1].expand_as(oracle)
    assert torch.equal(hard[observed], oracle[observed])
    assert torch.equal(relaxed[observed], oracle[observed])
    correct = hidden_accuracy(hard, oracle, prior.observed[0][1].logical_not())
    assert report["hidden_accuracy"]["bit_correct"] == correct["bit_correct"]
    assert report["hidden_accuracy"]["token_correct"] == correct["token_correct"]
    for path in ("hard", "relaxed", "oracle"):
        assert report["paths"][path]["mesh_status_counts"] == {"no_isosurface": 1}
        assert report["paths"][path]["cd_x1e4_valid_only"]["mean"] is None
    assert report["paths"]["oracle"]["drift"]["field_mae_vs_oracle"]["mean"] == 0


def test_completion_zero_mask_has_no_accuracy_support(completion_setup, tmp_path, field_options):
    _, _, tokens, geometry = completion_setup
    report = evaluation.eval_completion("prior", "tokenizer", tokens, geometry,
                                         output=tmp_path / "zero", resolution=4, ratio=0, **field_options)
    assert report["successful_samples"] == 1
    assert report["hidden_accuracy"]["bit_count"] == 0
    assert report["hidden_accuracy"]["hidden_bit_accuracy"] is None
    assert report["hidden_accuracy"]["hidden_token_accuracy"] is None
    for path in ("hard", "relaxed"):
        assert report["paths"][path]["drift"]["field_mae_vs_oracle"]["mean"] == 0


def test_completion_rejects_a_different_tokenizer_cache(completion_setup, tmp_path, field_options):
    tokenizer, _, tokens, geometry = completion_setup
    tokenizer.offset.add_(1)
    with pytest.raises(ValueError, match="different tokenizer weights"):
        evaluation.eval_completion("prior", "tokenizer", tokens, geometry,
                                     output=tmp_path / "wrong", resolution=4, **field_options)
    assert not (tmp_path / "wrong").exists()


def test_completion_rejects_prior_bound_to_other_weights(completion_setup, tmp_path, field_options):
    _, prior, tokens, geometry = completion_setup
    prior.checkpoint_metadata["tokenizer_hash"] = "0" * 64
    with pytest.raises(ValueError, match="different tokenizer"):
        evaluation.eval_completion("prior", "tokenizer", tokens, geometry,
                                   output=tmp_path / "wrong", resolution=4, **field_options)


def test_completion_rejects_stale_geometry(completion_setup, tmp_path, field_options, rewrite_npz):
    _, _, tokens, geometry = completion_setup
    with np.load(geometry / "generated_5.npz") as sample:
        target = sample["query_sdf"] + 0.01
    rewrite_npz(geometry / "generated_5.npz", query_sdf=target)
    with pytest.raises(ValueError, match="different content fingerprints"):
        evaluation.eval_completion("prior", "tokenizer", tokens, geometry,
                                   output=tmp_path / "wrong", resolution=4, **field_options)


def test_completion_rejects_changes_overlapping_observed_cells(
        completion_setup, tmp_path, field_options, monkeypatch):
    _, _, tokens, geometry = completion_setup

    def invalid_completion(model, bits, ratio, seed):
        result = masked_completion(model, bits, ratio, seed)
        observed = ~result["masked"].expand_as(bits)
        result["hard_bits"] = torch.where(observed, 1 - bits, result["hard_bits"])
        return result

    monkeypatch.setattr("orbit_folding.models.prior.masked_completion", invalid_completion)
    report = evaluation.eval_completion("prior", "tokenizer", tokens, geometry,
                                         output=tmp_path / "overlap", resolution=4, **field_options)
    assert report["successful_samples"] == 0
    assert "changed observed tokens" in report["failed_samples"][0]["error"]


def test_mesh_and_grid_drift_failure_semantics():
    assert extract_mesh(np.ones((4, 4, 4)))[0]["status"] == "no_isosurface"
    assert extract_mesh(np.full((4, 4, 4), np.nan))[0]["status"] == "nonfinite_field"
    oracle = np.arange(64, dtype=float).reshape(4, 4, 4)
    result = grid_drift(oracle + 0.125, oracle, spacing=0.1)
    assert result["field_mae_vs_oracle"] == 0.125
    assert result["grid_gradient_mae_vs_oracle"] == 0
