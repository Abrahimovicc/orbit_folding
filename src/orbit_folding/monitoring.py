# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Training-log plots and reproducible, explicitly split-labeled diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def format_progress(task, report):
    """One readable line; all objective terms remain in metrics.jsonl."""
    prefix = (f"{task} | epoch {report['epoch'] + 1:03d} | update {report['step']:05d}"
              f" | loss {report['objective']:.4f}")
    if task == "tokenizer":
        prefix += (f" | SDF MAE {report['debug/token_mae']:.5f}"
                   f" | sign {100 * report['debug/token_sign_acc']:.1f}%"
                   f" | batch codes {report['debug/code_count']:g}")
    else:
        if "bit_accuracy" in report:
            prefix += f" | bit {100 * report['bit_accuracy']:.2f}%"
        prefix += (f" | token {100 * report['token_exact']:.2f}%"
                   f" | hidden {100 * report['mask_ratio']:.1f}%")
    return prefix + f" | lr {report['lr']:.2e}"


def _figure(rows=2, cols=2, width=10.8, height=6.4):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), dpi=170, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, cols, squeeze=False)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8, color="#9da5ac")
        axis.title.set_fontsize(11)
    return figure, axes


def plot_training(run, output=None):
    """Plot recorded updates as-is: no smoothing or interpolated observations."""
    run = Path(run)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No recorded training updates")
    steps = [row["step"] for row in rows]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Training steps must increase; use a fresh run directory for a new run")
    run_info = json.loads((run / "run.json").read_text())
    task = run_info["task"]
    categorical = run_info.get("prior_type") == "categorical"
    validation_path = run / "validation.jsonl"
    validation = ([json.loads(line) for line in validation_path.read_text().splitlines() if line.strip()]
                  if validation_path.exists() else [])
    metric = "uniform_sdf_mae" if task == "tokenizer" else ("hidden_ce" if categorical else "hidden_bce")
    if validation:
        allowed = ({metric, "truncated_uniform_sdf_mae"} if task == "tokenizer" else {metric, "mean_" + metric})
        metric = validation[0].get("selection_metric")
        if (metric not in allowed
                or any(row.get("selection_metric") != metric or not np.isfinite(row.get("selection_score", np.nan))
                for row in validation)
                or any(b["step"] < a["step"] for a, b in zip(validation, validation[1:]))):
            raise ValueError("Validation records need finite task-matched scores and nondecreasing steps")
    figure, axes = _figure(rows=3 if validation else 2, height=9.4 if validation else 6.4)

    def curves(axis, title, series, percent=False):
        for key, label, color in series:
            values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
            axis.plot(steps, values * (100 if percent else 1), color=color, label=label, linewidth=1.7)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Optimizer update", fontsize=9)
        axis.grid(axis="y", color="#e7ebed", linewidth=0.6)
        if percent:
            axis.set_ylabel("Percent", fontsize=9)
            axis.set_ylim(0, 102)
        axis.legend(frameon=False, fontsize=8)

    curves(axes[0, 0], "Training objective", [("objective", "Objective", "#193f77")])
    if task == "tokenizer":
        curves(axes[0, 1], "SDF MAE on sampled training queries", [
            (f"debug/{name}_mae", label, color) for name, label, color in
            (("raw", "Encoder", "#168276"), ("continuous", "Continuous", "#c87932"),
             ("token", "Quantized", "#284f94"))])
        curves(axes[1, 0], "Quantized-field sign accuracy", [
            ("debug/token_sign_acc", "Sampled queries", "#168276")], percent=True)
        curves(axes[1, 1], "Distinct codes in the logged minibatch", [
            ("debug/code_count", "Unique codes (rank mean under DDP)", "#c87932")])
    elif task == "prior":
        accuracy_curves = [("token_exact", "Token exactness", "#c87932")]
        if not categorical:
            accuracy_curves.insert(0, ("bit_accuracy", "Bit accuracy", "#168276"))
        curves(axes[0, 1], "Masked-index accuracy" if categorical else "Masked-bit completion accuracy",
               accuracy_curves, percent=True)
        curves(axes[1, 0], "Sampled hidden-cell fraction", [
            ("mask_ratio", "Hidden cells", "#284f94")], percent=True)
        curves(axes[1, 1], "Learning rate", [("lr", "AdamW", "#8b546f")])
    else:
        raise ValueError("Unknown training task")
    best = None
    if validation:
        validation_steps = [row["step"] for row in validation]
        scores = [row["selection_score"] for row in validation]
        best = validation[int(np.argmin(scores))]
        selection_axis, accuracy_axis = axes[2]
        selection_axis.plot(validation_steps, scores, color="#284f94", marker=".", label=metric)
        selection_axis.scatter([best["step"]], [best["selection_score"]], marker="*", s=100,
                               color="#b24a32", zorder=4, label=f"Best recorded: update {best['step']}")
        titles = {"uniform_sdf_mae": "Uniform SDF MAE", "truncated_uniform_sdf_mae": "Truncated SDF MAE",
                  "hidden_ce": "Hidden-index CE", "hidden_bce": "Hidden-bit BCE",
                  "mean_hidden_ce": "Mean masked-index CE", "mean_hidden_bce": "Mean masked-bit BCE"}
        selection_axis.set_title("Held-out validation | " + titles[metric],
                                 loc="left", fontweight="bold")
        accuracy_series = ([("uniform_sign_accuracy", "Uniform sign", "#168276")] if task == "tokenizer" else
                           ([("token_exact", "Exact tokens, 50%", "#c87932")] if categorical else
                            [("bit_accuracy", "Hidden bits, 50%", "#168276"), ("token_exact", "Exact tokens, 50%", "#c87932")]))
        for key, label, color in accuracy_series:
            values = [row.get(key) if row.get(key) is not None else np.nan for row in validation]
            accuracy_axis.plot(validation_steps, np.asarray(values) * 100, color=color, marker=".", label=label)
        if task == "tokenizer":
            for key, label, color in [("token_effective_support", "Effective support", "#8b546f"),
                                      ("token_nominal_usage", "Nominal usage", "#284f94")]:
                if any(key in row for row in validation):
                    accuracy_axis.plot(validation_steps, [row.get(key, np.nan) * 100 for row in validation],
                                       color=color, linestyle="--", label=label)
        elif any(any(item["mask_ratio"] == .9 for item in row.get("by_mask_ratio", [])) for row in validation):
            for key, label, color in [("token_exact", "Exact tokens, 90%", "#b24a32"),
                                      *([] if categorical else [("bit_accuracy", "Hidden bits, 90%", "#775b99")])]:
                values = [next((item.get(key, np.nan) for item in row.get("by_mask_ratio", []) if item["mask_ratio"] == .9), np.nan)
                          for row in validation]
                accuracy_axis.plot(validation_steps, np.asarray(values) * 100, color=color, linestyle="--", label=label)
        accuracy_axis.set_title("Held-out validation accuracy", loc="left", fontweight="bold")
        accuracy_axis.set(ylim=(0, 102), ylabel="Percent")
        for axis in (selection_axis, accuracy_axis):
            axis.set_xlabel("Optimizer update", fontsize=9)
            axis.grid(axis="y", color="#e7ebed", linewidth=0.6)
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"{task.capitalize()} training | {len(rows)} recorded updates", fontsize=14, fontweight="bold", x=0.07, ha="left")
    note = ("Training minibatches and separate held-out validation; best marker uses this log. No smoothing."
            if validation else "Training minibatches only; not held-out benchmark scores. No smoothing.")
    figure.text(0.07, 0.025, note, fontsize=9, color="#5d666d")
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.87, hspace=0.53, wspace=0.25)
    path = Path(output) if output else run / "training.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    figure.clear()
    path.with_suffix(".plot.json").write_text(json.dumps({"task": task, "training_updates": steps,
        "validation_updates": [row["step"] for row in validation],
        "best_recorded_validation": ({key: best[key] for key in ("step", "selection_metric", "selection_score")}
                                     if best is not None else None)}, indent=2) + "\n")
    return str(path)


@torch.random.fork_rng(devices=[])
@torch.no_grad()
def preview_checkpoint(run, data, output=None, *, split="val", sample_id=None, device="cpu", seed=2026):
    """Inspect a saved checkpoint, without mutating the training model or RNG."""
    from .checkpoints import load_model
    from .training import geometry_dataset
    from .token_cache import TokenDataset

    run = Path(run)
    model, config = load_model(run / "last.pt", device)
    task = config["Task"]
    dataset = geometry_dataset(data, config, split, seed) if task == "tokenizer" else TokenDataset(data, split)
    index = 0
    if sample_id:
        indices = [index for index, item in enumerate(dataset.items) if item["id"] == sample_id]
        if not indices:
            raise ValueError(f"No sample {sample_id!r} in split {split!r}")
        index = indices[0]
    batch = dataset[index]
    identity = batch["sample_id"]
    metadata = {"task": task, "sample_id": identity, "split": split, "seed": seed,
                "checkpoint": "last.pt", "scope": "Checkpoint diagnostic, not a benchmark result"}
    if task == "tokenizer":
        points, frame = batch["points"][None].to(device), batch["frame"][None].to(device)
        encoded = model.encode(points, frame)
        coordinate = torch.linspace(-0.55, 0.55, 96, device=device)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        queries = torch.stack([xx, yy, torch.zeros_like(xx)], -1).reshape(1, -1, 3)
        figure, axes = _figure(rows=1, cols=4, width=13, height=3.8)
        cloud = encoded["representative"][0].cpu().numpy()
        axes[0, 0].scatter(cloud[:, 0], cloud[:, 1], s=4, c="#284f94", linewidths=0)
        axes[0, 0].set_title("Input, representative frame", loc="left", fontweight="bold")
        axes[0, 0].set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55), aspect="equal")
        missing = []
        for axis, name, title in zip(axes[0, 1:], ("raw_planes", "continuous_planes", "quantized_planes"),
                                     ("Encoder field", "Continuous field", "Quantized field")):
            values = torch.cat([model.decode(encoded[name], chunk) for chunk in queries.split(2048, dim=1)], dim=1)
            field = values.reshape(96, 96).cpu().numpy()
            axis.imshow(field, origin="lower", extent=(-0.55, 0.55, -0.55, 0.55),
                        cmap="RdBu_r", vmin=-0.05, vmax=0.05)
            if field.min() < 0 < field.max():
                axis.contour(xx.cpu(), yy.cpu(), field, levels=[0], colors="black", linewidths=1)
            else:
                axis.text(0.5, 0.05, "No zero crossing in slice", transform=axis.transAxes,
                          ha="center", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none"})
                missing.append(name)
            axis.set_title(title + " | z = 0", loc="left", fontweight="bold")
        metadata.update(slice_z=0, sdf_color_range_model_units=[-0.05, 0.05], no_zero_crossing=missing)
        note = "SDF range [-0.05, 0.05] model units: blue inside, red outside; black contour = 0. Not a rendered mesh."
    elif model.prior_type == "categorical":
        from matplotlib import colormaps
        from .models.categorical import batch_indices, categorical_completion

        indices = batch_indices({key: value[None].to(device) for key, value in batch.items()
                                 if isinstance(value, torch.Tensor)}, model.codebook_size)
        result = categorical_completion(model, indices, .5, seed)
        target, predicted = indices[0].cpu().numpy(), result["indices"][0].cpu().numpy()
        hidden = result["masked"][0, 0].cpu().numpy()
        metadata.update(prior_type="categorical", mask_ratio=.5, hidden_accuracy=result["accuracy"],
                        error_legend={"correct_hidden": "green", "wrong_hidden": "red", "observed": "gray"})
        figure, axes = _figure(rows=1, cols=4, width=13, height=3.8)
        maps = (target, np.ma.array(target, mask=hidden), predicted,
                np.where(hidden, (target != predicted).astype(float), np.nan))
        titles = ("Target indices", "Observed indices", "Completed indices", "Hidden-token errors")
        for index, (axis, values, title) in enumerate(zip(axes.flat, maps, titles)):
            axis.imshow(values, vmin=0, vmax=1 if index == 3 else model.codebook_size - 1,
                        cmap=colormaps["RdYlGn_r"].with_extremes(bad="#e9edef") if index == 3 else "tab20",
                        interpolation="nearest")
            axis.set_title(title, loc="left", fontweight="bold", fontsize=10)
            axis.set(xticks=[], yticks=[])
        note = (f"Fixed 50% whole-cell mask | exact tokens {100 * result['accuracy']['hidden_token_accuracy']:.2f}%"
                " | errors: green correct, red wrong, gray observed; index colors are IDs.")
    else:
        from .models.prior import masked_completion
        from .metrics import hidden_accuracy

        bits = batch["bits"][None].to(device)
        result = masked_completion(model, bits, ratio=0.5, seed=seed)
        target, hard, probability = (tensor[0].cpu().numpy() for tensor in
                                     (bits, result["hard_bits"], result["probabilities"]))
        hidden = result["masked"][0, 0].cpu().numpy()
        accuracy = hidden_accuracy(result["hard_bits"], bits, result["masked"])
        metadata.update(bit_plane=0, mask_ratio=0.5, hidden_accuracy=accuracy)
        figure, axes = _figure(rows=1, cols=5, width=13, height=3.8)
        maps = (target[0], np.where(hidden, 0.5, target[0]), probability[0], hard[0],
                np.where(hidden, (target != hard).mean(axis=0), np.nan))
        titles = ("Target\nBit 0", "Observed\nGray = hidden", "Relaxed probability\nBit 0",
                  "Hard prediction\nBit 0", "Wrong-bit fraction\nHidden cells")
        for index, (axis, values, title) in enumerate(zip(axes.flat, maps, titles)):
            axis.imshow(values, vmin=0, vmax=1, cmap="gray" if index in (0, 1, 3) else "cividis", interpolation="nearest")
            axis.set_title(title, loc="left", fontweight="bold", fontsize=10)
            axis.set(xticks=[], yticks=[])
        note = (f"Fixed 50% whole-cell mask | hidden bits {100 * accuracy['hidden_bit_accuracy']:.2f}%"
                f" | exact tokens {100 * accuracy['hidden_token_accuracy']:.2f}% | error uses all bit planes; white = observed.")
    figure.suptitle(f"{task.capitalize()} checkpoint | {identity} | split: {split}", fontsize=14, fontweight="bold", x=0.04, ha="left")
    figure.text(0.04, 0.035, note, fontsize=9, color="#4e5961")
    figure.subplots_adjust(left=0.04, right=0.99, bottom=0.19, top=0.79, wspace=0.33)
    path = Path(output) if output else run / "preview.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    figure.clear()
    metadata["image"] = path.name
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return str(path)
