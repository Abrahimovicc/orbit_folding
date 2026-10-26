# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Validation-only probability calibration and checkpoint integrity."""

import json
import math

import pytest
import torch
from torch.utils.data import DataLoader

from orbit_folding.calibration import calibrate_field, calibrate_prior, fit_temperature
from orbit_folding.checkpoints import load_checkpoint, save_checkpoint
from orbit_folding.data import geometry_dataset
from orbit_folding.token_cache import export_tokens
from orbit_folding.training import build_model


def test_temperature_matches_constant_logit_optimum():
    logits=torch.full((400,),4.)
    targets=torch.tensor([1.,1.,1.,0.]).repeat(100)
    report=fit_temperature(logits,targets,bounds=(.5,6.))
    assert report['selected_temperature']==pytest.approx(4/math.log(3),rel=.002)
    assert report['validation_bce_after'] < report['validation_bce_before']
    torch.testing.assert_close(logits.sign(),(logits/report['selected_temperature']).sign())


@pytest.mark.parametrize('bounds',[(0,3),(2,3),(.5,.9),(1,1),(.5,float('nan'))])
def test_temperature_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError,match='bounds'):
        fit_temperature(torch.ones(4),torch.ones(4),bounds)


@pytest.mark.parametrize('logits,targets',[
    ([],[]),([1,2],[1]),([float('nan')],[1]),([1],[.5]),([1],[float('nan')])])
def test_temperature_rejects_invalid_observations(logits,targets):
    with pytest.raises(ValueError,match='finite logits|binary'):
        fit_temperature(torch.tensor(logits),torch.tensor(targets))


@pytest.fixture
def prior_with_cache(geometry_root,tokenizer_config,prior_config,tmp_path):
    tokenizer=build_model(tokenizer_config).eval()
    cache=tmp_path/'tokens'
    manifest=export_tokens(tokenizer,DataLoader(geometry_dataset(geometry_root,tokenizer_config,'all')),cache,'cpu')
    checkpoint=tmp_path/'prior.pt'
    save_checkpoint(checkpoint,build_model(prior_config),prior_config,extra={'tokenizer_hash':manifest['tokenizer_hash']})
    return checkpoint,cache


def test_prior_calibration_uses_only_validation_and_changes_only_output_head(prior_with_cache,tmp_path,monkeypatch):
    import orbit_folding.calibration as calibration
    from orbit_folding.token_cache import TokenDataset

    accessed=[]
    class Recorded(TokenDataset):
        def __getitem__(self,index):
            accessed.append(self.items[index]['split'])
            return super().__getitem__(index)
    monkeypatch.setattr(calibration,'TokenDataset',Recorded)
    checkpoint,cache=prior_with_cache
    before=load_checkpoint(checkpoint)
    rng=torch.get_rng_state().clone()
    output=tmp_path/'calibrated.pt'
    report=calibrate_prior(checkpoint,cache,output)
    assert torch.equal(rng,torch.get_rng_state())
    assert accessed==['val'] and report['samples']==1 and report['split']=='val'
    assert report['validation_bce_after'] <= report['validation_bce_before']
    after=load_checkpoint(output)
    assert before['config']==after['config']
    assert before['extra']['tokenizer_hash']==after['extra']['tokenizer_hash']
    assert after['optimizer_state_dict'] is None
    assert before['extra']==load_checkpoint(checkpoint)['extra']
    assert all(torch.equal(value,after['state_dict'][key]) for key,value in before['state_dict'].items()
               if key not in {'model.head.weight','model.head.bias'})
    assert after['extra']['calibration']['source_sha256']==report['source_sha256']
    with pytest.raises(FileExistsError):
        calibrate_prior(checkpoint,cache,output)


def test_prior_calibration_rejects_known_binding_mismatch(prior_with_cache,tmp_path):
    checkpoint,cache=prior_with_cache
    from orbit_folding.checkpoints import load_model
    model,config=load_model(checkpoint)
    wrong=tmp_path/'wrong.pt'
    save_checkpoint(wrong,model,config,extra={'tokenizer_hash':'wrong'})
    output=tmp_path/'result.pt'
    with pytest.raises(ValueError,match='different tokenizer'):
        calibrate_prior(wrong,cache,output)
    assert not output.exists()


def test_prior_calibration_cli(prior_with_cache,tmp_path,run_cli):
    checkpoint,cache=prior_with_cache
    result=run_cli('calibrate-prior','--checkpoint',checkpoint,'--tokens',cache,
        '--output',tmp_path/'posterior.pt','--device','cpu')
    assert json.loads(result.stdout)['split']=='val'
    assert (tmp_path/'posterior.json').exists()


@pytest.mark.parametrize('weighting',['uniform','balanced'])
def test_field_calibration_fits_existing_bias_without_changing_other_weights(
        geometry_root,tokenizer_config,tmp_path,monkeypatch,weighting):
    import orbit_folding.calibration as calibration
    from orbit_folding.checkpoints import load_model

    checkpoint=tmp_path/'tokenizer.pt'
    save_checkpoint(checkpoint,build_model(tokenizer_config),tokenizer_config)
    model,config=load_model(checkpoint)
    model.encode=lambda points,frame: {'quantized_planes':points.norm(dim=-1).mean(1)}
    model.decode=lambda planes,xyz: (xyz.norm(dim=-1)-planes[:,None]).clamp(-.1,.1)+model.field_head.sdf_head[-1].bias*model.field_head.output_scale
    monkeypatch.setattr(calibration,'load_model',lambda *args:(model,config))
    before=load_checkpoint(checkpoint)
    output=tmp_path/'calibrated.pt'
    report=calibrate_field(checkpoint,geometry_root,output,train_samples=4,
        observation_points=16,weighting=weighting)
    assert report['train_samples']==4 and report['validation_samples']==1
    assert report['validation_ids']==['generated_4']
    assert 'generated_5' not in report['train_ids']+report['validation_ids']
    assert report['fitted_sdf_offset']==pytest.approx(-.002,abs=3e-5)
    after=load_checkpoint(output)
    assert after['config']==before['config']
    assert all(torch.equal(value,after['state_dict'][key]) for key,value in before['state_dict'].items()
               if key!='field_head.sdf_head.8.bias')
    if report['accepted_by_validation_guards']:
        assert report['validation_fitted']['inside_gap_pp'] <= report['validation_before']['inside_gap_pp']
    else:
        assert report['selected_sdf_offset']==0
    with pytest.raises(FileExistsError):
        calibrate_field(checkpoint,geometry_root,output)


def test_field_calibration_cli(geometry_root,tokenizer_config,tmp_path,run_cli):
    checkpoint=tmp_path/'tokenizer.pt'
    save_checkpoint(checkpoint,build_model(tokenizer_config),tokenizer_config)
    result=run_cli('calibrate-field','--checkpoint',checkpoint,'--data',geometry_root,
        '--output',tmp_path/'distance.pt','--device','cpu','--train-samples','4','--observation-points','16')
    assert json.loads(result.stdout)['validation_samples']==1


def test_field_calibration_rejects_nonfinite_field(geometry_root,tokenizer_config,tmp_path,monkeypatch):
    import orbit_folding.calibration as calibration
    from orbit_folding.checkpoints import load_model

    checkpoint=tmp_path/'tokenizer.pt'
    save_checkpoint(checkpoint,build_model(tokenizer_config),tokenizer_config)
    model,config=load_model(checkpoint)
    model.decode=lambda planes,xyz:torch.full_like(xyz[...,0],float('nan'))
    monkeypatch.setattr(calibration,'load_model',lambda *args:(model,config))
    output=tmp_path/'invalid.pt'
    with pytest.raises(FloatingPointError,match='Nonfinite'):
        calibrate_field(checkpoint,geometry_root,output,train_samples=1,observation_points=16)
    assert not output.exists()
