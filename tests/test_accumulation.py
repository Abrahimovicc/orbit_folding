# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Resume inside an epoch with an incomplete gradient-accumulation group."""
import copy
import json

import torch
import pytest
import yaml

from orbit_folding.checkpoints import load_checkpoint


@pytest.mark.parametrize("interval", ["step", "epoch"])
def test_mid_epoch_accumulation_resume_is_exact(geometry_root, tokenizer_config, tmp_path, run_cli, interval):
    config = copy.deepcopy(tokenizer_config)
    config["Trainer"]["accumulate_grad_batches"] = 3
    config["Trainer"]["lr_scheduler_interval"] = interval
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    resumed, complete = tmp_path / "resumed", tmp_path / "complete"

    def arguments(output, steps):
        return ["train-tokenizer", "--config", config_path, "--data", geometry_root,
                "--output", output, "--device", "cpu", "--threads", 1,
                "--batch-size", 1, "--epochs", 1, "--max-steps", steps]

    run_cli(*arguments(resumed, 1))
    partial = load_checkpoint(resumed / "last.pt")
    assert partial["extra"]["engine"]["next_batch"] == 3
    run_cli(*arguments(resumed, 2), "--resume", resumed / "last.pt")
    run_cli(*arguments(complete, 2))
    a, b = load_checkpoint(resumed / "last.pt"), load_checkpoint(complete / "last.pt")
    assert a["step"] == b["step"] == 2
    assert a["scheduler_state_dict"]["last_epoch"] == (1 if interval=="epoch" else 2)
    for name in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][name], b["state_dict"][name], rtol=0, atol=0)


@pytest.mark.parametrize("pause", [1,2,3])
def test_epoch_scheduler_resumes_across_epoch_boundaries(geometry_root,tokenizer_config,tmp_path,run_cli,pause):
    config = copy.deepcopy(tokenizer_config)
    config["Trainer"].update(accumulate_grad_batches=3,lr_scheduler_interval="epoch")
    path = tmp_path/'epoch.yaml'
    path.write_text(yaml.safe_dump(config))
    resumed,complete = tmp_path/'resumed',tmp_path/'complete'

    def args(output,steps):
        return ["train-tokenizer","--config",path,"--data",geometry_root,"--output",output,
                "--device","cpu","--threads",1,"--batch-size",1,"--epochs",2,"--max-steps",steps]

    run_cli(*args(resumed,pause))
    run_cli(*args(resumed,4),"--resume",resumed/'last.pt')
    run_cli(*args(complete,4))
    a,b = load_checkpoint(resumed/'last.pt'),load_checkpoint(complete/'last.pt')
    assert a['scheduler_state_dict']==b['scheduler_state_dict']
    assert a['scheduler_state_dict']['last_epoch']==2
    for name in a['state_dict']:
        torch.testing.assert_close(a['state_dict'][name],b['state_dict'][name],rtol=0,atol=0)
    rows = [json.loads(line) for line in (complete/'metrics.jsonl').read_text().splitlines()]
    assert rows[0]['lr']==rows[1]['lr'] and rows[2]['lr']==rows[3]['lr']
    assert rows[0]['lr']>rows[2]['lr']
