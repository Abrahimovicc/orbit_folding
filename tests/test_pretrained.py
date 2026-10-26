# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Local weight preparation verifies the original cache lineage before binding."""

import json

import pytest
import torch
import yaml

from orbit_folding.checkpoints import load_checkpoint
from orbit_folding.training import build_model
from orbit_folding.weights import prepare_hf_release, verify_weights


@pytest.fixture
def source_pair(tmp_path, tokenizer_config, prior_config):
    root = tmp_path / "private_sources"
    root.mkdir()
    tokenizer_path = root / "tokenizer.ckpt"
    torch.save({"state_dict": build_model(tokenizer_config).state_dict()}, tokenizer_path)
    torch.save({"state_dict": build_model(prior_config).state_dict(),
                "hyper_parameters": {"token_path": "tokens"}}, root / "prior.ckpt")
    metadata = {"stage1_ckpt": "tokenizer.ckpt", "tokens_path": "tokens", "codebook_size": 16}
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "tokenizer.yaml").write_text(yaml.safe_dump({**tokenizer_config, "DataSource": str(root / "shapes")}))
    (root / "prior.yaml").write_text(yaml.safe_dump({**prior_config, "token_path": str(root / "tokens")}))
    return {"pairs": [{"id": "reference", "scope": "Unit-test weights, not trained results",
                       "source_root": str(root), "tokenizer_source": "tokenizer.ckpt",
                       "prior_source": "prior.ckpt", "tokenizer_config": "tokenizer.yaml",
                       "prior_config": "prior.yaml", "token_metadata": "metadata.json"}]}


def test_prepare_binds_verified_pairs_without_exporting_private_paths(source_pair, tmp_path):
    output = tmp_path / "release"
    manifest = prepare_hf_release(source_pair, output)
    assert manifest["published"] is False
    assert verify_weights(output) == manifest
    prior = load_checkpoint(output / "reference/prior.pt")
    assert prior["extra"]["tokenizer_hash"] == manifest["pairs"][0]["tokenizer_hash"]
    assert "token_path" not in prior["config"]
    private_root = source_pair["pairs"][0]["source_root"]
    for path in output.rglob("*"):
        if path.suffix in {".json", ".yaml", ".md"}:
            assert private_root not in path.read_text()
    with pytest.raises(FileExistsError):
        prepare_hf_release(source_pair, output)


@pytest.mark.parametrize("field", ["stage1_ckpt", "tokens_path"])
def test_prepare_rejects_a_mismatched_source_lineage(source_pair, tmp_path, field):
    from pathlib import Path

    root = Path(source_pair["pairs"][0]["source_root"])
    path = root / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = "another_source"
    path.write_text(json.dumps(metadata))
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="another"):
        prepare_hf_release(source_pair, output)
    assert not output.exists()
