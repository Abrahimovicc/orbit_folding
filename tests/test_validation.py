# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Mask-sweep validation, objective-aligned SDF selection and support diagnostics."""

import math

import pytest
import torch
from torch.utils.data import DataLoader
import yaml

from orbit_folding.data import geometry_dataset
from orbit_folding.token_cache import TokenDataset, export_tokens
from orbit_folding.training import build_model
from orbit_folding.validation import validate


@pytest.mark.parametrize("kind", ["bit", "categorical"])
def test_mask_sweep_is_batch_invariant_and_separates_primary_metric(
        kind,geometry_root,tokenizer_config,prior_config,project_root,tmp_path):
    tokenizer = build_model(tokenizer_config).eval()
    root = tmp_path/'cache'
    export_tokens(tokenizer,DataLoader(geometry_dataset(geometry_root,tokenizer_config,"all")),root,"cpu")
    config = (prior_config if kind=="bit" else yaml.safe_load((project_root/'configs/prior_categorical_demo.yaml').read_text()))
    model = build_model(config).train()
    data = TokenDataset(root,"train")
    name = "hidden_bce" if kind=="bit" else "hidden_ce"
    rng = torch.get_rng_state().clone()
    one = validate(model,data,task="prior",device="cpu",batch_size=1,mask_ratios=[.1,.5,.9],
                   selection_metric="mean_"+name)
    many = validate(model,data,task="prior",device="cpu",batch_size=3,mask_ratios=[.1,.5,.9],
                    selection_metric="mean_"+name)
    assert model.training and torch.equal(rng,torch.get_rng_state())
    assert one["selection_metric"]=="mean_"+name
    assert one["selection_score"]==pytest.approx(sum(row[name] for row in one["by_mask_ratio"])/3)
    assert one[name]==one["by_mask_ratio"][1][name]
    assert one["token_exact"]==one["by_mask_ratio"][1]["token_exact"]
    for a,b in zip(one["by_mask_ratio"],many["by_mask_ratio"]):
        assert a["token_count"]==b["token_count"]
        assert a["token_correct"]==b["token_correct"]
        assert a[name]==pytest.approx(b[name],rel=1e-6)
    assert len({row["token_count"] for row in one["by_mask_ratio"]})==3


def test_tokenizer_validation_reports_truncated_error_and_pooled_support(geometry_root,tokenizer_config):
    model = build_model(tokenizer_config)
    data = geometry_dataset(geometry_root,tokenizer_config,"val")
    old = validate(model,data,task="tokenizer",device="cpu")
    new = validate(model,data,task="tokenizer",device="cpu",selection_metric="truncated_uniform_sdf_mae")
    assert old["selection_metric"]=="uniform_sdf_mae"
    assert new["selection_metric"]=="truncated_uniform_sdf_mae"
    assert math.isfinite(new["selection_score"])
    assert old["uniform_sdf_mae"]==pytest.approx(new["uniform_sdf_mae"])
    assert 0<new["token_effective_support"]<=new["token_nominal_usage"]<=1
    assert new["token_codes_for_50"]>=1
    assert new["truncated_absolute_error_sum"]==pytest.approx(new["selection_score"]*new["query_count"])


@pytest.mark.parametrize("ratios", [[.1,.9],[.5,.5],[0,.5],[-.1,.5],[.5,float("nan")]])
def test_mask_sweep_rejects_invalid_ratios(ratios,prior_config):
    with pytest.raises(ValueError,match="ratios|0.5"):
        validate(build_model(prior_config),[],task="prior",device="cpu",mask_ratios=ratios)


def test_sparse_context_profile_changes_coverage_not_architecture(project_root):
    paper = yaml.safe_load((project_root/'configs/prior_paper.yaml').read_text())
    sparse = yaml.safe_load((project_root/'configs/prior_sparse_context.yaml').read_text())
    a,b = build_model(paper),build_model(sparse)
    assert a.ratio_max==.6 and b.ratio_max==.95
    assert {key:tuple(value.shape) for key,value in a.state_dict().items()} == {
        key:tuple(value.shape) for key,value in b.state_dict().items()}
    loss,_ = b({'bits':torch.randint(0,2,(2,12,8,8)).float()})
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in b.parameters())
