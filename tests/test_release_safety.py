# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Regression cases for run safety, source identity, normals and AMP accounting."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from orbit_folding.checkpoints import load_checkpoint, verify_prior_binding
from orbit_folding.preprocessing import import_preprocessed, preprocess
from orbit_folding.provenance import dataset_fingerprint, file_sha256
from orbit_folding.training import _check_output, _optimizer_update, _training_device


@pytest.mark.parametrize("requested,world,rank,expected", [("cuda", 1, 0, "cuda:2"),
    ("cuda:3", 1, 0, "cuda:3"), ("auto", 1, 0, "cuda:2"), ("cuda", 2, 1, "cuda:1")])
def test_cuda_training_resolves_a_concrete_device_index(monkeypatch, requested, world, rank, expected):
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    assert str(_training_device(requested, rank, world)) == expected
    assert [str(device) for device in selected] == [expected]


@pytest.mark.parametrize("configured,override,expected", [(17, None, 17), (17, 5, 5), (None, None, 2026)])
def test_training_seed_uses_config_unless_explicitly_overridden(monkeypatch, configured, override, expected):
    import orbit_folding.training as training

    config = {"seed": configured} if configured is not None else {}
    monkeypatch.setattr(training, "_train", lambda *args: args[-3])
    assert training.train(config, "unused", "unused", device="cpu", seed=override) == expected


def test_nonempty_output_is_rejected_before_any_write(tmp_path):
    existing = tmp_path / "last.pt"
    existing.write_bytes(b"precious checkpoint")
    with pytest.raises(FileExistsError, match="not empty"):
        _check_output(tmp_path, None, 0)
    assert existing.read_bytes() == b"precious checkpoint"


@pytest.mark.parametrize("interval", ["step","epoch"])
def test_all_skipped_updates_are_not_reported_as_success(geometry_root, tmp_path, tokenizer_config, monkeypatch, interval):
    import orbit_folding.training as training

    def skip(model, optimizer, scaler, scheduler, clip):
        optimizer.zero_grad(set_to_none=True)
        return False, float("inf")

    monkeypatch.setattr(training, "_optimizer_update", skip)
    tokenizer_config["Trainer"]["lr_scheduler_interval"] = interval
    output = tmp_path / "overflow"
    with pytest.raises(FloatingPointError, match="No optimizer updates completed"):
        training.train(tokenizer_config, geometry_root, output, device="cpu", epochs=1, max_steps=1)
    assert (output / "last.pt").exists() and (output / "skipped_updates.jsonl").exists()
    checkpoint = load_checkpoint(output / "last.pt")
    assert checkpoint["scheduler_state_dict"]["last_epoch"] == 0


def test_inconsistent_resume_log_requires_a_new_branch(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"step": 1}\n{"step": 2}\n')
    with pytest.raises(ValueError, match="logs"):
        _check_output(tmp_path, tmp_path / "last.pt", 1)
    with pytest.raises(ValueError, match="fresh"):
        _check_output(tmp_path, tmp_path / "best.pt", 2)
    _check_output(tmp_path, tmp_path / "last.pt", 2)


def test_content_fingerprint_changes_even_without_manifest_edits(geometry_root, rewrite_npz):
    before = dataset_fingerprint(geometry_root)
    with np.load(geometry_root / "generated_0.npz") as data:
        replacement = data["query_sdf"] + 0.001
    rewrite_npz(geometry_root / "generated_0.npz", query_sdf=replacement)
    assert dataset_fingerprint(geometry_root) != before


def test_declared_checksum_rejects_modified_payload(geometry_root, rewrite_npz):
    path = geometry_root / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["samples"][0]
    entry["sha256"] = file_sha256(geometry_root / entry["path"])
    path.write_text(json.dumps(manifest))
    rewrite_npz(geometry_root / entry["path"], center=np.array([99, 99, 99]))
    with pytest.raises(ValueError, match="checksum"):
        dataset_fingerprint(geometry_root)


@pytest.mark.parametrize("override", [False, True])
def test_known_prior_binding_mismatch_cannot_be_overridden(override):
    prior = SimpleNamespace(checkpoint_metadata={"tokenizer_hash": "a" * 64})
    with pytest.raises(ValueError, match="different tokenizer"):
        verify_prior_binding(prior, "b" * 64, allow_unverified=override)


def test_missing_prior_binding_requires_explicit_legacy_override():
    prior = SimpleNamespace()
    with pytest.raises(ValueError, match="no tokenizer binding"):
        verify_prior_binding(prior, "a" * 64)
    assert verify_prior_binding(prior, "a" * 64, allow_unverified=True) is False


def _legacy_file(destination, geometry_root):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with np.load(geometry_root / "generated_0.npz") as sample:
        np.savez_compressed(destination, encoder_points=sample["points"], encoder_normals=sample["normals"],
                            query_points=sample["query_points"], query_sdf=sample["query_sdf"],
                            uniform_points=sample["uniform_points"], uniform_sdf=sample["uniform_sdf"])


def test_import_rejects_resolved_source_aliases_before_writing(tmp_path, geometry_root):
    root = tmp_path / "raw"
    _legacy_file(root / "sample/data.npz", geometry_root)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["sample"], "test": ["sample/data.npz"]}))
    with pytest.raises(ValueError, match="(?i)duplicate source"):
        import_preprocessed(root, split, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_import_rejects_identical_content_across_splits(tmp_path, geometry_root):
    root = tmp_path / "raw"
    _legacy_file(root / "a.npz", geometry_root)
    (root / "b.npz").write_bytes((root / "a.npz").read_bytes())
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["a.npz"], "test": ["b.npz"]}))
    with pytest.raises(ValueError, match="Identical source"):
        import_preprocessed(root, split, tmp_path / "out")


@pytest.mark.parametrize("nested", [False, True])
def test_import_cannot_overwrite_source_data(tmp_path, geometry_root, nested):
    root = tmp_path / "raw"
    source = root / "a.npz"
    _legacy_file(source, geometry_root)
    before = source.read_bytes()
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["a.npz"]}))
    output = root / "imported" if nested else root
    with pytest.raises(ValueError, match="outside"):
        import_preprocessed(root, split, output)
    assert source.read_bytes() == before


@pytest.mark.parametrize("nested", [False, True])
def test_component_normals_match_material_sdf_including_cavities(tmp_path, nested):
    import open3d as o3d

    first = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=12)
    second = o3d.geometry.TriangleMesh.create_sphere(radius=0.25, resolution=12)
    if not nested:
        first.translate([-0.8, 0, 0])
        second.translate([0.8, 0, 0])
        second.triangles = o3d.utility.Vector3iVector(np.asarray(second.triangles)[:, [0, 2, 1]])
    root = tmp_path / "raw"
    root.mkdir()
    o3d.io.write_triangle_mesh(str(root / "assembly.ply"), first + second)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["assembly"]}))
    output = tmp_path / "processed"
    manifest = preprocess(root, output, surface_points=1024, query_points=256, uniform_points=64, split_file=split)
    assert manifest["samples"][0]["components"] == 2
    with np.load(output / "assembly.npz") as sample:
        probes = np.concatenate([sample["points"] - 1e-4 * sample["normals"],
                                 sample["points"] + 1e-4 * sample["normals"]])
    mesh = o3d.io.read_triangle_mesh(str(output / "assembly.ply"))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    signed = scene.compute_signed_distance(o3d.core.Tensor(probes.astype(np.float32)), nsamples=5).numpy().reshape(2, -1)
    assert (signed[0] < 0).mean() > 0.99
    assert (signed[1] > 0).mean() > 0.99
    assert manifest["samples"][0]["sha256"] == file_sha256(output / "assembly.npz")


def test_actual_cpu_scaler_skip_preserves_optimizer_schedule_and_parameters():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    scaler.scale(model(torch.ones(1, 1)).sum()).backward()
    next(model.parameters()).grad.fill_(float("inf"))
    initial_epoch = scheduler.last_epoch
    updated, _ = _optimizer_update(model, optimizer, scaler, scheduler, 1.0)
    assert not updated and scheduler.last_epoch == initial_epoch
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    scaler.scale(model(torch.ones(1, 1)).sum()).backward()
    updated, _ = _optimizer_update(model, optimizer, scaler, scheduler, 1.0)
    assert updated and scheduler.last_epoch == initial_epoch + 1
