# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Asset integration examples must share the main cache contract."""

import json

import numpy as np
import pytest
from torch.utils.data import DataLoader

from orbit_folding.checkpoints import load_model, save_checkpoint
from orbit_folding.data import geometry_dataset
from orbit_folding.token_cache import export_tokens
from orbit_folding.training import build_model


@pytest.mark.parametrize("kind", ["lfq", "standard_vq"])
@pytest.mark.parametrize("locked", [True, False])
def test_catalog_matches_configured_cli_cache(
        geometry_root, tokenizer_config, tmp_path, run_python, run_cli, kind, locked):
    tokenizer_config["Tokenizer"]["type"] = kind
    tokenizer_config["EnableAxisRepresentativeLock"] = locked
    checkpoint = tmp_path / "tokenizer.pt"
    save_checkpoint(checkpoint, build_model(tokenizer_config), tokenizer_config)
    catalog, official = tmp_path / "catalog", tmp_path / "official"
    result = run_python("applications/catalog.py", "--checkpoint", checkpoint,
        "--data-root", geometry_root, "--output", catalog, "--seed", "17")
    run_cli("cache-tokens", "--checkpoint", checkpoint, "--data", geometry_root,
        "--output", official, "--seed", "17", "--threads", "1")
    printed = json.loads(result.stdout)
    assert printed["num_samples"] == 6 and printed["quantizer_type"] == kind
    assert ("bit_shape" if kind == "lfq" else "token_shape") in printed
    a, b = (json.loads((directory / "manifest.json").read_text()) for directory in (catalog, official))
    assert a["provenance"]["encoder_sampling"] == b["provenance"]["encoder_sampling"]
    for left, right in zip(a["items"], b["items"]):
        assert (left["id"], left["split"]) == (right["id"], right["split"])
        with np.load(catalog / left["path"]) as actual, np.load(official / right["path"]) as expected:
            np.testing.assert_array_equal(actual["bits" if kind == "lfq" else "indices"],
                                          expected["bits" if kind == "lfq" else "indices"])


def test_catalog_records_explicit_point_budget(geometry_root, tokenizer_config, tmp_path, run_python):
    checkpoint = tmp_path / "tokenizer.pt"
    save_checkpoint(checkpoint, build_model(tokenizer_config), tokenizer_config)
    catalog, expected = tmp_path / "catalog", tmp_path / "expected"
    run_python("applications/catalog.py", "--checkpoint", checkpoint, "--data-root", geometry_root,
        "--output", catalog, "--pc-size", "32", "--seed", "29")
    model, config = load_model(checkpoint)
    export_tokens(model, DataLoader(geometry_dataset(geometry_root, config, "all", seed=29,
        pc_size=32, points_only=True)), expected, "cpu")
    manifest = json.loads((catalog / "manifest.json").read_text())
    settings = manifest["provenance"]["encoder_sampling"]
    assert settings["pc_size"] == 32 and settings["point_strategy"] == "spatial_mixed"
    for row in manifest["items"]:
        with np.load(catalog / row["path"]) as actual, np.load(expected / row["path"]) as reference:
            np.testing.assert_array_equal(actual["bits"], reference["bits"])
