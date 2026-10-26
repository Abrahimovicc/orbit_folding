# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Learned-codebook and categorical-prior contracts, CLI workflows and DDP."""

import json
import math

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from orbit_folding.checkpoints import load_checkpoint, load_model, save_checkpoint
from orbit_folding.data import geometry_dataset
from orbit_folding.models.categorical import batch_indices, categorical_completion
from orbit_folding.models.prior import masked_completion
from orbit_folding.models.quantization import TriPlaneVQTokenizer
from orbit_folding.paper_evaluation import evaluate_token_prior
from orbit_folding.token_cache import TokenDataset, export_tokens, tokenizer_hash
from orbit_folding.training import build_model, train
from orbit_folding.validation import validate


@pytest.fixture
def vq_config(project_root):
    return yaml.safe_load((project_root / "configs/tokenizer_vq_demo.yaml").read_text())


@pytest.fixture
def categorical_config(project_root):
    return yaml.safe_load((project_root / "configs/prior_categorical_demo.yaml").read_text())


@pytest.mark.parametrize("size", [16, 17, 8192])
def test_vq_hard_lookup_and_finite_backward_without_binary_states(size):
    model = TriPlaneVQTokenizer(24, 16, 16, size, 1)
    features = torch.randn(2, 24, 8, 8)
    result = model(features)
    (result["quantized_planes"].square().mean() + result["aux_loss"]).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert "bits" not in result and not hasattr(model, "decode_probabilities")
    model.eval()
    result = model(features)
    assert torch.allclose(model.decode_indices(result["indices"]), result["quantized_planes"], atol=1e-6)


def test_vq_checkpoint_and_ema_round_trip(vq_config, tmp_path):
    model = build_model(vq_config)
    before = model.tokenizer.quantizer.codebook.clone()
    model.tokenizer(torch.randn(1, 24, 8, 8))
    assert not torch.equal(before, model.tokenizer.quantizer.codebook)
    path = tmp_path / "vq.pt"
    save_checkpoint(path, model, vq_config)
    restored, _ = load_model(path)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match="LFQ-specific"):
        restored.decode_probabilities(torch.rand(1, 4, 8, 8))


@pytest.fixture
def vq_cache(geometry_root, vq_config, tmp_path):
    vq_config["Tokenizer"]["codebook_size"] = 17
    model = build_model(vq_config)
    cache = tmp_path / "indices"
    export_tokens(model, DataLoader(geometry_dataset(geometry_root, vq_config, "all"), batch_size=2), cache, "cpu")
    return cache, model


def test_vq_cache_preserves_actual_vocabulary_and_no_fake_bits(vq_cache):
    cache, _ = vq_cache
    dataset = TokenDataset(cache, "all")
    assert dataset.num_bits is None and dataset.codebook_size == 17
    assert dataset.token_shape == (8, 8) and len(dataset) == 6
    assert "bits" not in dataset[0] and dataset[0]["indices"].dtype == torch.int64
    from orbit_folding.token_support import cache_histogram
    counts = cache_histogram(cache, "test")["series"][0]["counts"]
    assert len(counts) == 17 and sum(counts) == 64


@pytest.mark.parametrize("value,dtype", [(-1, np.int64), (17, np.int64), (1.2, np.float32)])
def test_invalid_vq_cached_indices_are_rejected(vq_cache, value, dtype):
    cache, _ = vq_cache
    dataset = TokenDataset(cache, "test")
    path = cache / dataset.items[0]["path"]
    with np.load(path) as saved:
        values = {key: saved[key].copy() for key in saved.files}
    values["indices"] = np.full((8, 8), value, dtype=dtype)
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="Cached indices"):
        dataset[0]


def test_bit_field_prior_refuses_vq_indices(vq_cache, prior_config, tmp_path):
    cache, _ = vq_cache
    with pytest.raises(ValueError, match="LFQ bits"):
        train(prior_config, cache, tmp_path / "bad", device="cpu", max_steps=1)


def test_categorical_ce_and_counts_with_zero_logits(categorical_config):
    model = build_model(categorical_config)
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    loss, metrics = model({"indices": target})
    assert float(loss.detach()) == pytest.approx(math.log(16), rel=1e-6)
    assert metrics["token_correct"] == metrics["token_count"] > 0
    assert "bit_accuracy" not in metrics and metrics["loss_count"] == metrics["token_count"]
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("ratio", [0, .5, 1])
def test_categorical_and_bit_priors_share_masks_and_preserve_observations(categorical_config, prior_config, ratio):
    categorical, bit = build_model(categorical_config), build_model(prior_config)
    bits = torch.randint(2, (2, 4, 8, 8)).float()
    indices = batch_indices({"bits": bits}, 16)
    state = torch.get_rng_state().clone()
    result = categorical_completion(categorical, indices, ratio, 7)
    assert torch.equal(state, torch.get_rng_state()) and categorical.training
    binary = masked_completion(bit, bits, ratio, 7)
    assert torch.equal(result["masked"], binary["masked"])
    assert result["accuracy"]["observed_tokens_unchanged"]
    if ratio == 0:
        assert result["accuracy"]["hidden_token_accuracy"] is None and result["loss_sum"] == 0


@pytest.mark.parametrize("value,dtype", [(-1, torch.int64), (16, torch.int64), (.5, torch.float32)])
def test_categorical_rejects_bad_labels_instead_of_clamping(categorical_config, value, dtype):
    model = build_model(categorical_config)
    with pytest.raises(ValueError, match="indices"):
        model({"indices": torch.full((1, 8, 8), value, dtype=dtype)})


def test_categorical_token_eval_and_validation_with_vq(vq_cache, categorical_config, tmp_path):
    cache, tokenizer = vq_cache
    categorical_config["vocab_size"] = categorical_config["Tokenizer"]["codebook_size"] = 17
    model = build_model(categorical_config)
    path = tmp_path / "prior.pt"
    save_checkpoint(path, model, categorical_config, extra={"tokenizer_hash": tokenizer_hash(tokenizer)})
    report = evaluate_token_prior(path, token_root=cache, output=tmp_path / "eval", ratios=[0, .5, 1])
    assert report["prior_type"] == "categorical" and report["samples"] == 1
    assert report["rows"][0]["ce"] is None and report["rows"][2]["token_count"] == 64
    assert "bit_accuracy" not in report["rows"][1]
    result = validate(model, TokenDataset(cache, "val"), task="prior", device="cpu")
    assert result["selection_metric"] == "hidden_ce" and math.isfinite(result["selection_score"])


@pytest.fixture
def baseline_pipeline(geometry_root, project_root, tmp_path, run_cli):
    output = tmp_path / "pipeline"
    tokenizer, cache, prior = (output / name for name in ("tokenizer", "tokens", "prior"))
    run_cli("train-tokenizer", "--config", "configs/tokenizer_vq_demo.yaml", "--data", geometry_root,
        "--output", tokenizer, "--device", "cpu", "--epochs", 2, "--max-steps", 2, "--threads", 1)
    run_cli("cache-tokens", "--checkpoint", tokenizer / "last.pt", "--data", geometry_root,
        "--output", cache, "--device", "cpu")
    run_cli("train-prior", "--config", "configs/prior_categorical_demo.yaml", "--data", cache,
        "--output", prior, "--device", "cpu", "--epochs", 2, "--max-steps", 2, "--threads", 1)
    return {"geometry": geometry_root, "tokenizer": tokenizer, "cache": cache, "prior": prior}


@pytest.mark.integration
def test_vq_categorical_cli_training_validation_evaluation_and_previews(baseline_pipeline, tmp_path, run_cli):
    paths = baseline_pipeline
    for task in ("tokenizer", "prior"):
        run = paths[task]
        assert load_checkpoint(run / "last.pt")["step"] == 2
        assert (run / "best.pt").exists() and (run / "training.png").exists()
        run_cli("plot-training", "--run", run, "--data", paths["geometry" if task == "tokenizer" else "cache"])
        assert (run / "preview.png").exists()
    run_cli("evaluate", "tokens", "--tokens", paths["cache"], "--prior", paths["prior"] / "last.pt",
        "--output", tmp_path / "token_eval", "--threads", 1)
    run_cli("evaluate", "completion", "--data", paths["geometry"], "--tokens", paths["cache"],
        "--prior", paths["prior"] / "last.pt", "--tokenizer", paths["tokenizer"] / "last.pt",
        "--resolution", 8, "--output", tmp_path / "geometry_eval")
    report = json.loads((tmp_path / "geometry_eval/completion.json").read_text())
    assert report["successful_samples"] == 1 and set(report["paths"]) == {"categorical", "oracle"}
    assert report["quantizer_type"] == "standard_vq" and "hidden_bit_accuracy" not in report["hidden_accuracy"]


@pytest.mark.parametrize("quantizer", ["lfq", "standard_vq"])
def test_categorical_training_accepts_either_real_cache(geometry_root, tokenizer_config, categorical_config, quantizer, tmp_path):
    tokenizer_config["Tokenizer"]["type"] = quantizer
    tokenizer = build_model(tokenizer_config)
    cache = tmp_path / "tokens"
    export_tokens(tokenizer, DataLoader(geometry_dataset(geometry_root, tokenizer_config, "all")), cache, "cpu")
    result = train(categorical_config, cache, tmp_path / "train", device="cpu", epochs=1, max_steps=2)
    assert result["step"] == 2


@pytest.mark.parametrize("kind", ["tokenizer", "prior"])
def test_baseline_resume_is_exact(kind, geometry_root, vq_cache, vq_config, categorical_config, tmp_path):
    config = vq_config if kind == "tokenizer" else categorical_config
    data = geometry_root if kind == "tokenizer" else vq_cache[0]
    if kind == "prior":
        config["vocab_size"] = config["Tokenizer"]["codebook_size"] = 17
    partial, complete = tmp_path / "partial", tmp_path / "complete"
    train(config, data, partial, device="cpu", epochs=2, max_steps=1)
    train(config, data, partial, device="cpu", epochs=2, max_steps=2, resume=partial / "last.pt")
    train(config, data, complete, device="cpu", epochs=2, max_steps=2)
    a, b = load_checkpoint(partial / "last.pt"), load_checkpoint(complete / "last.pt")
    assert all(torch.equal(v, b["state_dict"][k]) for k, v in a["state_dict"].items())


def test_vq_raw_inference_and_categorical_only_hard_decoding(vq_config, categorical_config, tmp_path):
    from orbit_folding.inference import encode_observation, decode_observation
    tokenizer = build_model(vq_config)
    path, prior_path = tmp_path / "tokenizer.pt", tmp_path / "prior.pt"
    save_checkpoint(path, tokenizer, vq_config)
    prior = build_model(categorical_config)
    save_checkpoint(prior_path, prior, categorical_config, extra={"tokenizer_hash": tokenizer_hash(tokenizer)})
    source, encoded = tmp_path / "points.npy", tmp_path / "encoded.npz"
    np.save(source, np.random.default_rng(1).normal(size=(96, 3)).astype(np.float32) * [3, 2, 1])
    encode_observation(path, source, encoded)
    with np.load(encoded) as values:
        assert "bits" not in values and values["indices"].shape == (8, 8)
    result = decode_observation(path, encoded, tmp_path / "completed.ply", resolution=8, prior_checkpoint=prior_path)
    assert result["state"] == "hard" and result["hidden_accuracy"]["observed_tokens_unchanged"]
    with pytest.raises(ValueError, match="hard indices only"):
        decode_observation(path, encoded, tmp_path / "relaxed.ply", resolution=8,
                           prior_checkpoint=prior_path, state="relaxed")


def test_lfq_categorical_completion_uses_same_frozen_decoder(geometry_root, tokenizer_config, categorical_config, tmp_path):
    from orbit_folding.evaluation import eval_completion
    tokenizer = build_model(tokenizer_config)
    cache, tokenizer_path, prior_path = tmp_path / "tokens", tmp_path / "tokenizer.pt", tmp_path / "prior.pt"
    export_tokens(tokenizer, DataLoader(geometry_dataset(geometry_root, tokenizer_config, "all")), cache, "cpu")
    save_checkpoint(tokenizer_path, tokenizer, tokenizer_config)
    save_checkpoint(prior_path, build_model(categorical_config), categorical_config,
                    extra={"tokenizer_hash": tokenizer_hash(tokenizer)})
    report = eval_completion(prior_path, tokenizer_path, cache, geometry_root, output=tmp_path / "eval", resolution=8)
    assert report["successful_samples"] == 1 and report["prior_type"] == "categorical"
    assert report["samples"][0]["accuracy"]["observed_tokens_unchanged"]


@pytest.mark.parametrize("kind", ["tokenizer", "prior"])
@pytest.mark.distributed
def test_baselines_two_rank_gloo_training(kind, geometry_root, vq_cache, tmp_path, run_python):
    data = geometry_root if kind == "tokenizer" else vq_cache[0]
    config = "configs/tokenizer_vq_demo.yaml" if kind == "tokenizer" else "configs/prior_categorical_demo.yaml"
    if kind == "prior":
        values = yaml.safe_load(open(config))
        values["vocab_size"] = values["Tokenizer"]["codebook_size"] = 17
        config = tmp_path / "prior.yaml"
        config.write_text(yaml.safe_dump(values))
    output = tmp_path / "distributed"
    run_python("-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", "-m", "orbit_folding",
        f"train-{kind}", "--config", config, "--data", data, "--output", output,
        "--device", "cpu", "--epochs", 1, "--max-steps", 1, "--batch-size", 1, "--threads", 1)
    assert load_checkpoint(output / "last.pt")["step"] == 1
