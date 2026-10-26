# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""CPU subprocess smoke tests: execution and integrity, never visual quality."""

import json
import math
import shutil

import numpy as np
import pytest
import torch

from orbit_folding.checkpoints import CHECKPOINT_FORMAT, load_checkpoint, load_model
from orbit_folding.token_cache import TokenDataset, tokenizer_hash


def _train_args(kind, data, output, *, epochs=4, max_steps=2):
    return [f"train-{kind}", "--config", f"configs/{kind}_demo.yaml", "--data", data,
            "--output", output, "--device", "cpu", "--epochs", epochs,
            "--max-steps", max_steps, "--batch-size", 2, "--workers", 0, "--threads", 1]


def _metrics(output):
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
    assert rows
    for row in rows:
        assert all(math.isfinite(value) for value in row.values() if isinstance(value, (int, float)))
    return rows


def _assert_training_output(output, kind, steps, *, world=1, samples=4):
    checkpoint = load_checkpoint(output / "last.pt")
    assert checkpoint["format"] == CHECKPOINT_FORMAT
    assert checkpoint["model_type"] == kind
    assert checkpoint["step"] == steps
    assert checkpoint["optimizer_state_dict"] is not None
    assert checkpoint["scheduler_state_dict"] is not None
    assert all(torch.isfinite(value).all() for value in checkpoint["state_dict"].values())
    run = json.loads((output / "run.json").read_text())
    assert run["device_type"] == "cpu"
    assert run["world_size"] == world
    assert run["training_samples"] == samples
    rows = _metrics(output)
    assert rows[-1]["step"] == steps
    assert len({row["step"] for row in rows}) == len(rows)
    return checkpoint


@pytest.fixture(scope="module")
def cli_pipeline(tmp_path_factory, geometry_factory, run_cli):
    root = tmp_path_factory.mktemp("cli_pipeline")
    data = geometry_factory(root / "geometry")
    tokenizer, cache, prior = (root / name for name in ("tokenizer", "cache", "prior"))
    run_cli(*_train_args("tokenizer", data, tokenizer))
    run_cli("cache-tokens", "--checkpoint", tokenizer / "last.pt", "--data", data,
            "--output", cache, "--batch-size", 2, "--device", "cpu")
    run_cli(*_train_args("prior", cache, prior))
    return {"data": data, "tokenizer": tokenizer, "cache": cache, "prior": prior}


@pytest.mark.integration
def test_cli_train_cache_prior_preserves_splits(cli_pipeline):
    paths = cli_pipeline
    _assert_training_output(paths["tokenizer"], "tokenizer", 2)
    prior = _assert_training_output(paths["prior"], "prior", 2)
    tokenizer, _ = load_model(paths["tokenizer"] / "last.pt", "cpu")
    cache = json.loads((paths["cache"] / "manifest.json").read_text())
    geometry = json.loads((paths["data"] / "manifest.json").read_text())
    assert cache["tokenizer_hash"] == tokenizer_hash(tokenizer)
    assert prior["extra"]["tokenizer_hash"] == cache["tokenizer_hash"]
    assert cache["num_samples"] == 6
    assert cache["split_counts"] == {"train": 4, "val": 1, "test": 1}
    assert {(row["id"], row["split"]) for row in cache["items"]} == {
        (row["id"], row["split"]) for row in geometry["samples"]}
    for split, expected in (("train", 4), ("val", 1), ("test", 1)):
        dataset = TokenDataset(paths["cache"], split)
        assert len(dataset) == expected
        for index in range(len(dataset)):
            bits = dataset[index]["bits"]
            assert bits.shape == (4, 8, 8)
            assert torch.all((bits == 0) | (bits == 1))
    rows = _metrics(paths["prior"])
    for row in rows:
        assert row["bit_count"] == 4 * row["token_count"]
        assert 0 <= row["bit_correct"] <= row["bit_count"]
        assert 0 <= row["token_correct"] <= row["token_count"]
        assert row["bit_accuracy"] == pytest.approx(row["bit_correct"] / row["bit_count"])


@pytest.mark.integration
def test_support_cli_counts_only_the_requested_cache_split(cli_pipeline, tmp_path, run_cli):
    from orbit_folding.token_support import cache_histogram

    root = cli_pipeline["cache"]
    counts = cache_histogram(root, "test")
    bits = TokenDataset(root, "test")[0]["bits"].numpy().astype(np.int64)
    indices = np.sum(bits * np.array([8, 4, 2, 1])[:, None, None], axis=0)
    assert counts["series"][0]["counts"] == np.bincount(indices.ravel(), minlength=16).tolist()
    assert counts["provenance"]["samples"] == 1
    run_cli("analyze-support", "--tokens", root, "--split", "test", "--output", tmp_path)
    report = json.loads((tmp_path / "token_support.json").read_text())
    assert report["series"][0]["total_tokens"] == 64
    assert report["provenance"]["split"] == "test"


@pytest.mark.integration
def test_repeated_training_refuses_to_overwrite_checkpoint(cli_pipeline, run_cli):
    output = cli_pipeline["prior"]
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    result = run_cli(*_train_args("prior", cli_pipeline["cache"], output), check=False)
    assert result.returncode and "not empty" in result.stdout + result.stderr
    assert before == {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}


@pytest.mark.integration
def test_validation_is_held_out_and_saves_best_checkpoint(cli_pipeline):
    for task in ("tokenizer", "prior"):
        output = cli_pipeline[task]
        assert (output / "best.pt").exists()
        rows = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
        assert rows and all(row["samples"] == 1 for row in rows)
        metric = "uniform_sdf_mae" if task == "tokenizer" else "hidden_bce"
        assert all(row["selection_metric"] == metric for row in rows)
        best = load_checkpoint(output / "best.pt")
        assert best["extra"]["engine"]["best_score"] == min(row["selection_score"] for row in rows)


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["tokenizer", "prior"])
@pytest.mark.parametrize("weights_only", [False, True])
def test_initialization_loads_weights_but_starts_a_fresh_run(cli_pipeline, tmp_path, run_cli, kind, weights_only):
    from orbit_folding.provenance import file_sha256

    source = cli_pipeline[kind] / "last.pt"
    if weights_only:
        payload = load_checkpoint(source)
        payload["optimizer_state_dict"] = payload["scheduler_state_dict"] = None
        payload["extra"].pop("engine")
        source = tmp_path / "weights.pt"
        torch.save(payload, source)
    before = file_sha256(source)
    output = tmp_path / "initialized"
    data = cli_pipeline["data" if kind == "tokenizer" else "cache"]
    run_cli(*_train_args(kind, data, output, epochs=2, max_steps=1),
            "--seed", 19, "--init-checkpoint", source)
    result = _assert_training_output(output, kind, 1)
    assert [row["step"] for row in _metrics(output)] == [1]
    assert all(int(state["step"]) == 1 for state in result["optimizer_state_dict"]["state"].values())
    assert result["extra"]["engine"]["seed"] == 19
    assert result["extra"]["initialization"]["source_sha256"] == before == file_sha256(source)
    assert json.loads((output / "run.json").read_text())["initialization"] == result["extra"]["initialization"]
    if kind == "prior":
        assert result["extra"]["tokenizer_hash"] == load_checkpoint(source)["extra"]["tokenizer_hash"]


@pytest.mark.integration
def test_initialized_run_resumes_with_initialization_provenance(cli_pipeline, tmp_path, run_cli):
    output = tmp_path / "initialized"
    source = cli_pipeline["tokenizer"] / "last.pt"
    run_cli(*_train_args("tokenizer", cli_pipeline["data"], output, max_steps=1), "--init-checkpoint", source)
    metadata = load_checkpoint(output / "last.pt")["extra"]["initialization"]
    run_cli(*_train_args("tokenizer", cli_pipeline["data"], output, max_steps=2), "--resume", output / "last.pt")
    assert load_checkpoint(output / "last.pt")["extra"]["initialization"] == metadata


@pytest.mark.integration
@pytest.mark.parametrize("binding", [None, "a" * 64])
def test_prior_initialization_rejects_missing_or_wrong_tokenizer_binding(cli_pipeline, tmp_path, run_cli, binding):
    payload = load_checkpoint(cli_pipeline["prior"] / "last.pt")
    payload["extra"]["tokenizer_hash"] = binding
    source = tmp_path / "wrong.pt"
    torch.save(payload, source)
    output = tmp_path / "rejected"
    result = run_cli(*_train_args("prior", cli_pipeline["cache"], output), "--init-checkpoint", source, check=False)
    assert result.returncode and "binding" in result.stderr
    assert not output.exists()


def test_resume_and_initialization_are_mutually_exclusive(run_cli):
    result = run_cli("train-tokenizer", "--resume", "last.pt", "--init-checkpoint", "weights.pt", check=False)
    assert result.returncode and "not allowed" in result.stderr


@pytest.mark.integration
@pytest.mark.parametrize("invalid", ["task", "shape", "nonfinite"])
def test_bad_initialization_is_rejected_before_writing(cli_pipeline, tmp_path, run_cli, invalid):
    payload = load_checkpoint(cli_pipeline["tokenizer"] / "last.pt")
    if invalid == "task":
        payload = load_checkpoint(cli_pipeline["prior"] / "last.pt")
    else:
        key = next(key for key, value in payload["state_dict"].items() if value.is_floating_point() and value.ndim > 0)
        payload["state_dict"][key] = (payload["state_dict"][key][0:0] if invalid == "shape" else
                                      torch.full_like(payload["state_dict"][key], float("nan")))
    source, output = tmp_path / "bad.pt", tmp_path / "rejected"
    torch.save(payload, source)
    result = run_cli(*_train_args("tokenizer", cli_pipeline["data"], output), "--init-checkpoint", source, check=False)
    assert result.returncode and not output.exists()


@pytest.mark.integration
def test_resume_rejects_modified_data_payload(cli_pipeline, tmp_path, run_cli, rewrite_npz):
    root = tmp_path / "changed"
    shutil.copytree(cli_pipeline["data"], root)
    with np.load(root / "generated_0.npz") as sample:
        target = sample["query_sdf"] + 0.01
    rewrite_npz(root / "generated_0.npz", query_sdf=target)
    result = run_cli(*_train_args("tokenizer", root, tmp_path / "output", max_steps=4),
                     "--resume", cli_pipeline["tokenizer"] / "last.pt", check=False)
    assert result.returncode and "fingerprint" in result.stdout + result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["tokenizer", "prior"])
def test_resume_matches_uninterrupted_run_with_same_epoch_target(cli_pipeline, tmp_path, run_cli, kind):
    data = cli_pipeline["data" if kind == "tokenizer" else "cache"]
    resumed, full = tmp_path / "resumed", tmp_path / "full"
    run_cli(*_train_args(kind, data, resumed, max_steps=4),
            "--resume", cli_pipeline[kind] / "last.pt")
    run_cli(*_train_args(kind, data, full, max_steps=4))
    resumed_state = _assert_training_output(resumed, kind, 4)
    full_state = _assert_training_output(full, kind, 4)
    assert [row["step"] for row in _metrics(resumed)] == [3, 4]
    for key in full_state["state_dict"]:
        torch.testing.assert_close(resumed_state["state_dict"][key], full_state["state_dict"][key], rtol=0, atol=0)
    assert resumed_state["scheduler_state_dict"] == full_state["scheduler_state_dict"]
    assert resumed_state["epoch"] == full_state["epoch"]


@pytest.mark.integration
@pytest.mark.parametrize("change", ["epochs", "seed", "batch_size"])
def test_resume_rejects_changed_training_contract(cli_pipeline, tmp_path, run_cli, change):
    output = tmp_path / "rejected"
    arguments = _train_args("prior", cli_pipeline["cache"], output, max_steps=4)
    if change == "epochs":
        arguments[arguments.index("--epochs") + 1] = 5
    elif change == "seed":
        arguments.extend(["--seed", 12345])
    else:
        arguments[arguments.index("--batch-size") + 1] = 1
    result = run_cli(*arguments, "--resume", cli_pipeline["prior"] / "last.pt", check=False)
    assert result.returncode != 0
    assert "resume" in (result.stdout + result.stderr).lower()
    assert not (output / "last.pt").exists()


@pytest.mark.integration
def test_prior_resume_rejects_a_different_tokenizer_cache(cli_pipeline, tmp_path, run_cli, rewrite_npz):
    cache = tmp_path / "other_cache"
    shutil.copytree(cli_pipeline["cache"], cache)
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_hash = manifest["tokenizer_hash"]
    other_hash = ("0" if source_hash[0] != "0" else "1") + source_hash[1:]
    manifest["tokenizer_hash"] = other_hash
    manifest["provenance"]["tokenizer_hash"] = other_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for entry in manifest["items"]:
        rewrite_npz(cache / entry["path"], tokenizer_hash=np.str_(other_hash),
                    provenance=np.str_(json.dumps(manifest["provenance"], sort_keys=True)))
    # This is internally valid; rejection must be based on resume provenance.
    assert TokenDataset(cache)[0]["bits"].shape == (4, 8, 8)
    output = tmp_path / "rejected"
    result = run_cli(*_train_args("prior", cache, output, max_steps=4),
                     "--resume", cli_pipeline["prior"] / "last.pt", check=False)
    assert result.returncode != 0
    assert any(word in (result.stdout + result.stderr).lower() for word in ("hash", "manifest", "tokenizer"))
    assert not (output / "last.pt").exists()


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["tokenizer", "prior"])
def test_convert_checkpoint_cli_uses_explicit_configuration(cli_pipeline, tmp_path, run_cli, kind):
    original = load_checkpoint(cli_pipeline[kind] / "last.pt")
    source, output = tmp_path / "original.pt", tmp_path / "converted.pt"
    torch.save({"state_dict": original["state_dict"], "epoch": 3, "global_step": 7}, source)
    run_cli("convert-checkpoint", "--source", source, "--config", f"configs/{kind}_demo.yaml",
            "--output", output, "--kind", kind, "--trust-original")
    converted = load_checkpoint(output)
    assert converted["model_type"] == kind
    assert (converted["epoch"], converted["step"]) == (3, 7)
    assert len(converted["extra"]["conversion"]["source_sha256"]) == 64
    assert str(source) not in json.dumps(converted["extra"])
    for key, value in original["state_dict"].items():
        torch.testing.assert_close(converted["state_dict"][key], value, rtol=0, atol=0)
    load_model(output, "cpu")


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["tokenizer", "prior"])
def test_train_command_rejects_wrong_config_task(tmp_path, run_cli, kind):
    other = "prior" if kind == "tokenizer" else "tokenizer"
    result = run_cli(f"train-{kind}", "--config", f"configs/{other}_demo.yaml",
                     "--data", tmp_path / "missing", "--output", tmp_path / "output",
                     "--device", "cpu", "--threads", 1, check=False)
    assert result.returncode != 0
    assert "Task" in result.stdout + result.stderr
    assert not (tmp_path / "output" / "last.pt").exists()


@pytest.mark.integration
def test_stability_cli_uses_only_requested_split(geometry_root, tmp_path, run_cli):
    output = tmp_path / "stability"
    result = run_cli("evaluate", "stability", "--data", geometry_root,
                     "--output", output, "--split", "test", "--device", "cpu")
    assert result.returncode == 0
    reports = list(output.glob("*.json")) + list(output.glob("*.jsonl"))
    assert reports, "Stability evaluation must write a machine-readable report"
    serialized = "\n".join(path.read_text() for path in reports)
    assert "generated_5" in serialized
    assert all(f"generated_{index}" not in serialized for index in range(5))
    for path in reports:
        if path.suffix == ".json":
            json.loads(path.read_text())
        else:
            for line in path.read_text().splitlines():
                if line.strip():
                    json.loads(line)


@pytest.mark.distributed
@pytest.mark.parametrize("kind", ["tokenizer", "prior"])
@pytest.mark.parametrize("initialize", [False, True])
def test_two_rank_cpu_gloo_training(cli_pipeline, tmp_path, run_python, kind, initialize):
    data = cli_pipeline["data" if kind == "tokenizer" else "cache"]
    output = tmp_path / f"ddp_{kind}"
    arguments = _train_args(kind, data, output, epochs=2)
    if initialize:
        arguments.extend(["--init-checkpoint", cli_pipeline[kind] / "last.pt"])
    run_python("-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
               "-m", "orbit_folding", *arguments, timeout=180)
    checkpoint = _assert_training_output(output, kind, 2, world=2)
    assert checkpoint["extra"]["engine"]["world_size"] == 2
    assert len(checkpoint["extra"]["engine"]["rng_by_rank"]) == 2
    assert [row["step"] for row in _metrics(output)] == [1, 2]
    assert ("initialization" in checkpoint["extra"]) == initialize
    if kind == "prior":
        manifest = json.loads((data / "manifest.json").read_text())
        assert checkpoint["extra"]["tokenizer_hash"] == manifest["tokenizer_hash"]
        for row in _metrics(output):
            assert row["bit_count"] == 4 * row["token_count"]
            assert row["bit_accuracy"] == pytest.approx(row["bit_correct"] / row["bit_count"])
