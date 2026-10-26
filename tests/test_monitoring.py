# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Verify that diagnostics reflect real logs, splits and checkpoint fields."""

import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from orbit_folding.checkpoints import save_checkpoint
from orbit_folding.monitoring import format_progress, plot_training, preview_checkpoint
from orbit_folding.token_cache import export_tokens
from orbit_folding.training import build_model, geometry_dataset
from orbit_folding.visualization import COLORS, render_alignment


@pytest.mark.parametrize("task", ["tokenizer", "prior"])
def test_progress_contains_task_specific_metrics(task):
    row = {"epoch": 0, "step": 3, "objective": 0.25, "lr": 1e-4,
           "debug/token_mae": 0.01, "debug/token_sign_acc": 0.9, "debug/code_count": 12,
           "bit_accuracy": 0.75, "token_exact": 0.25, "mask_ratio": 0.5}
    line = format_progress(task, row)
    assert "epoch 001" in line and "update 00003" in line
    assert ("SDF MAE 0.01000" in line) if task == "tokenizer" else ("bit 75.00%" in line)
    assert ("batch codes 12" in line) if task == "tokenizer" else ("token 25.00%" in line)


@pytest.mark.parametrize("task", ["tokenizer", "prior"])
def test_plot_reads_recorded_updates_without_rewriting_log(tmp_path, task):
    log = "\n".join(json.dumps({"step": index, "objective": 2 - index * 0.1, "lr": 1e-4})
                    for index in (1, 2, 3))
    (tmp_path / "metrics.jsonl").write_text(log)
    (tmp_path / "run.json").write_text(json.dumps({"task": task}))
    plot_training(tmp_path)
    image = np.asarray(Image.open(tmp_path / "training.png").convert("RGB"))
    assert image.shape[1] >= 1800 and image.std() > 10
    assert (tmp_path / "metrics.jsonl").read_text() == log


def test_plot_rejects_duplicate_update_numbers(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"step": 1}\n{"step": 1}\n')
    with pytest.raises(ValueError, match="increase"):
        plot_training(tmp_path)


@pytest.mark.parametrize("task", ["tokenizer", "prior"])
def test_plot_includes_validation_and_marks_best_recorded_score(tmp_path, task, monkeypatch):
    from matplotlib.figure import Figure

    training = [{"step": step, "objective": 3 - step * 0.1} for step in (1, 2, 3)]
    metric = "uniform_sdf_mae" if task == "tokenizer" else "hidden_bce"
    validation = [{"step": step, "selection_metric": metric, "selection_score": score,
                   "bit_accuracy": 0.8, "token_exact": 0.4, "uniform_sign_accuracy": 0.75}
                  for step, score in ((1, 0.3), (2, 0.1), (3, 0.2))]
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, training)))
    before = "\n".join(map(json.dumps, validation))
    (tmp_path / "validation.jsonl").write_text(before)
    (tmp_path / "run.json").write_text(json.dumps({"task": task}))
    figures = []
    original = Figure.savefig

    def capture(figure, *args, **kwargs):
        figures.append([axis.get_title(loc="left") for axis in figure.axes])
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    plot_training(tmp_path)
    assert len(figures[0]) == 6 and any("Held-out validation" in title for title in figures[0])
    sidecar = json.loads((tmp_path / "training.plot.json").read_text())
    assert sidecar["validation_updates"] == [1, 2, 3]
    assert sidecar["best_recorded_validation"] == {"step": 2, "selection_metric": metric, "selection_score": 0.1}
    assert (tmp_path / "validation.jsonl").read_text() == before


def test_plot_rejects_nonfinite_validation_score(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"step": 1, "objective": 0.2}\n')
    (tmp_path / "run.json").write_text('{"task": "prior"}')
    (tmp_path / "validation.jsonl").write_text('{"step": 1, "selection_metric": "hidden_bce", "selection_score": NaN}\n')
    with pytest.raises(ValueError, match="finite"):
        plot_training(tmp_path)


@pytest.mark.parametrize("kind", ["tokenizer", "bit", "categorical"])
def test_training_plots_objective_aligned_and_mask_sweep_validation(
        kind, geometry_root, tokenizer_config, prior_config, project_root, tmp_path):
    import yaml
    from orbit_folding.training import train

    config = tokenizer_config if kind == "tokenizer" else prior_config
    metric = "truncated_uniform_sdf_mae"
    data = geometry_root
    if kind != "tokenizer":
        metric = "mean_hidden_bce" if kind == "bit" else "mean_hidden_ce"
        if kind == "categorical":
            config = yaml.safe_load((project_root / "configs/prior_categorical_demo.yaml").read_text())
        data = tmp_path / "tokens"
        export_tokens(build_model(tokenizer_config), DataLoader(
            geometry_dataset(geometry_root, tokenizer_config, "all"), batch_size=2), data, "cpu")
    config["Validation"] = {"selection_metric": metric}
    if kind != "tokenizer":
        config["Validation"]["mask_ratios"] = [.1, .5, .9]
    config.setdefault("Trainer", {}).update(lr_scheduler_interval="epoch", plot_training=True)
    output = tmp_path / "run"
    result = train(config, data, output, epochs=1, batch_size=2, device="cpu", threads=1)
    assert result["epoch"] == 1
    assert Image.open(output / "training.png").size[0] >= 1800
    sidecar = json.loads((output / "training.plot.json").read_text())
    assert sidecar["best_recorded_validation"]["selection_metric"] == metric
    assert sidecar["validation_updates"] == [result["step"]]


@pytest.mark.parametrize("task", ["tokenizer", "prior"])
def test_checkpoint_preview_is_split_labeled_and_preserves_cpu_rng(
        tmp_path, geometry_root, tokenizer_config, prior_config, task):
    config = tokenizer_config if task == "tokenizer" else prior_config
    run = tmp_path / "run"
    run.mkdir()
    save_checkpoint(run / "last.pt", build_model(config), config)
    data = geometry_root
    if task == "prior":
        data = tmp_path / "tokens"
        tokenizer = build_model(tokenizer_config)
        export_tokens(tokenizer, DataLoader(geometry_dataset(geometry_root, tokenizer_config, "all"), batch_size=2), data, "cpu")
    state = torch.get_rng_state().clone()
    preview_checkpoint(run, data, split="val")
    assert torch.equal(state, torch.get_rng_state())
    assert Image.open(run / "preview.png").size[0] >= 2000
    metadata = json.loads((run / "preview.json").read_text())
    assert metadata["sample_id"] == "generated_4" and metadata["split"] == "val"
    assert metadata["task"] == task
    if task == "prior":
        assert metadata["mask_ratio"] == 0.5
        assert metadata["hidden_accuracy"]["observed_bits_unchanged"]
    else:
        assert metadata["slice_z"] == 0


def test_alignment_uses_paper_palette_and_is_nonblank_and_moving(geometry_root, tmp_path):
    result = render_alignment(geometry_root, tmp_path, frames=4, points=96,
                              width=960, height=360, supersample=2)
    assert result["palette"] == COLORS.tolist()
    assert result["renderer"] == "software"
    image = Image.open(result["gif"])
    first = np.asarray(image.convert("RGB")).copy()
    image.seek(1)
    second = np.asarray(image.convert("RGB"))
    assert np.mean(np.abs(first.astype(float) - second)) > 0.5
    for column in range(4):
        panel = first[95:290, column * 240 + 30:column * 240 + 210]
        assert np.any(panel.max(axis=-1).astype(int) - panel.min(axis=-1).astype(int) > 60)
    assert result["folded_pointwise_max_abs_model_units"] < 1e-10


def test_alignment_rejects_unknown_renderer(geometry_root, tmp_path):
    with pytest.raises(ValueError, match="renderer"):
        render_alignment(geometry_root, tmp_path, renderer="missing")
