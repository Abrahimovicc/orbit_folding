# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Count-pooled codebook utilization and source-backed distribution plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FORMAT = "orbit_folding.token_support.v1"


def summarize_counts(counts):
    """Compute entropy support from pooled integer counts, not batch averages."""
    counts = np.asarray(counts)
    if counts.ndim != 1 or counts.size == 0 or counts.dtype.kind not in "iu":
        raise ValueError("counts must be a nonempty one-dimensional integer histogram")
    if np.any(counts < 0):
        raise ValueError("counts must be nonnegative")
    # Sum as Python integers so overflow cannot masquerade as a small histogram.
    total = sum(map(int, counts))
    if not 0 < total <= 2**53:
        raise ValueError("histogram needs observations, with total count <= 2**53")
    ranked = np.sort(counts.astype(np.int64))[::-1]
    probabilities = ranked / total
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    perplexity = float(np.exp(entropy))
    cumulative = np.cumsum(ranked)
    return {"codebook_size": len(counts), "total_tokens": int(total),
            "active_codes": int(np.count_nonzero(counts)),
            "nominal_usage": float(np.count_nonzero(counts) / len(counts)),
            "entropy_nats": entropy, "perplexity": perplexity,
            "effective_support": perplexity / len(counts),
            "codes_for_50": int(np.searchsorted(cumulative, total * 0.5) + 1),
            "codes_for_90": int(np.searchsorted(cumulative, total * 0.9) + 1),
            "cumulative_mass": (cumulative / total).tolist()}


def cache_histogram(token_root, split="test"):
    """Pool MSB-first binary token cells from one explicit cache split."""
    from .token_cache import TokenDataset

    dataset = TokenDataset(token_root, split)
    if dataset.codebook_size > 2**20:
        raise ValueError("Dense support plots are limited to at most 2**20 codes")
    codebook_size = dataset.codebook_size
    counts = np.zeros(codebook_size, dtype=np.int64)
    for batch in dataset:
        if dataset.index_cache:
            indices = batch["indices"].numpy()
        else:
            bits = batch["bits"].numpy().astype(np.int64)
            weights = 1 << np.arange(dataset.num_bits - 1, -1, -1, dtype=np.int64)
            indices = np.sum(bits * weights[:, None, None], axis=0)
        counts += np.bincount(indices.ravel(), minlength=codebook_size)
    name = "Standard VQ" if dataset.index_cache else "LFQ"
    return {"format": FORMAT, "title": name + " token cache",
            "scope": f"Selected cache split: {split}; {len(dataset)} samples. Not a paper benchmark.",
            "provenance": {"split": split, "samples": len(dataset),
                           "tokenizer_hash": dataset.tokenizer_hash,
                           "counting": "All stored token cells, pooled across selected samples"},
            "series": [{"name": name, "counts": counts.tolist()}]}


def compare_caches(vq_root, lfq_root, output, *, split="test", vocabulary=8192):
    """Compare freshly extracted matched cohorts, not archived reference counts."""
    from .token_cache import TokenDataset

    vq, lfq = TokenDataset(vq_root, split), TokenDataset(lfq_root, split)
    if vq.quantizer_type != "standard_vq" or lfq.quantizer_type != "lfq":
        raise ValueError("Expected standard VQ and LFQ caches in that order")
    identities = lambda dataset: {(r["id"], r["split"], r["category"]) for r in dataset.items}
    if (identities(vq) != identities(lfq) or vq.token_shape != lfq.token_shape
            or vq.codebook_size != vocabulary or lfq.codebook_size != vocabulary):
        raise ValueError("Support control requires identical cohorts, token grids and vocabulary sizes")
    fingerprints = [d.manifest["provenance"].get("geometry_fingerprint") for d in (vq, lfq)]
    if any(value is None for value in fingerprints) or len(set(fingerprints)) != 1:
        raise ValueError("Matched support control requires verified identical source geometry")
    series = cache_histogram(vq_root, split)["series"] + cache_histogram(lfq_root, split)["series"]
    return plot_support({"format": FORMAT, "title": "Matched VQ/LFQ support control",
        "scope": "Fresh cache statistics. Matched cohorts/grid/vocabulary do not alone prove matched training budgets.",
        "provenance": {"samples": len(vq), "split": split, "geometry_fingerprint": fingerprints[0],
                       "tokenizer_hashes": [vq.tokenizer_hash, lfq.tokenizer_hash]}, "series": series}, output)


def plot_support(stats, output="outputs/token_support"):
    """Export a PNG/PDF and machine-readable metrics from actual histograms.

    Multi-series plots require matched vocabulary sizes and token counts.
    This does not by itself prove matched input identities or training budgets.
    Histograms need zero entries for unused codes; never trim the vocabulary.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if isinstance(stats, (str, Path)):
        stats = json.loads(Path(stats).read_text())
    if stats.get("format") != FORMAT:
        raise ValueError("Unsupported token-support format")
    series = stats.get("series", [])
    if not 1 <= len(series) <= 4:
        raise ValueError("Provide between one and four count histograms")
    names = [row.get("name") for row in series]
    if not all(isinstance(name, str) and 0 < len(name) <= 28 for name in names) or len(set(names)) != len(names):
        raise ValueError("Series names must be unique nonempty strings of at most 28 characters")
    metrics = [summarize_counts(row["counts"]) for row in series]
    if len({row["codebook_size"] for row in metrics}) > 1 or len({row["total_tokens"] for row in metrics}) > 1:
        raise ValueError("Comparison requires matched vocabulary sizes and token counts")
    colors = ["#c43c2f", "#1f5fa8", "#008577", "#905e91"] if len(series) > 1 else ["#1f5fa8"]
    figure = Figure(figsize=(11.4, 3.8), dpi=200, facecolor="white")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.38})
    left.plot([0, 100], [0, 100], color="#8c959d", linestyle=":", linewidth=1.2, label="Uniform")
    for index, (name, row, color) in enumerate(zip(names, metrics, colors)):
        size = row["codebook_size"]
        x = np.arange(1, size + 1) * 100 / size
        y = np.asarray(row["cumulative_mass"]) * 100
        left.plot(np.r_[0, x], np.r_[0, y], color=color, linewidth=2,
                  linestyle="--" if index % 2 == 0 and len(series) > 1 else "-",
                  label=f"{name}: {row['codes_for_50']:,} codes for 50% mass")
        marker_x = row["codes_for_50"] * 100 / size
        left.scatter([marker_x], [y[row["codes_for_50"] - 1]], color=color, s=24, zorder=4)
        center = len(series) - 1 - index
        for offset, key, shade in ((0.17, "nominal_usage", "#bcc5cd"),
                                   (-0.17, "effective_support", color)):
            value = row[key] * 100
            right.barh(center + offset, value, height=0.27, color=shade)
            right.text(value + 1.5, center + offset, f"{value:.1f}%", va="center", fontsize=9,
                       color="#29333c")
    left.set(title="Token-mass concentration", xlabel="Top-ranked codes (% of vocabulary)",
             ylabel="Cumulative token mass (%)", xlim=(0, 100), ylim=(0, 102))
    left.legend(loc="lower right", frameon=False, fontsize=8)
    right.set(title="Usage versus effective support", xlabel="Fraction of vocabulary (%)",
              yticks=np.arange(len(series)), yticklabels=names[::-1],
              xlim=(0, 113), ylim=(-0.62, len(series) - 0.38))
    from matplotlib.patches import Patch

    right.legend(handles=[Patch(facecolor="#bcc5cd", label="Nominal usage"),
                          Patch(facecolor="#404a54", label="Effective support (series color)")],
                 loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=8)
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.title.set(fontsize=11, fontweight="bold")
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
        axis.set_axisbelow(True)
        axis.grid(axis="x" if axis is right else "both", color="#e5e9ec", linewidth=0.6)
    title = str(stats.get("title", "Codebook utilization"))
    if len(title) > 90:
        raise ValueError("Plot title must have at most 90 characters")
    figure.suptitle(title, x=0.065, ha="left", fontsize=13, fontweight="bold")
    vocabulary, total = metrics[0]["codebook_size"], metrics[0]["total_tokens"]
    figure.text(0.065, 0.89, f"K = {vocabulary:,} | {total:,} token cells per series | effective support = exp(H) / K",
                fontsize=9, color="#56616b")
    figure.text(0.065, 0.025, str(stats.get("scope", "User-supplied counts")), fontsize=8, color="#56616b")
    figure.subplots_adjust(left=0.065, right=0.96, bottom=0.24, top=0.77)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(root / f"token_support.{suffix}", dpi=200)
    figure.clear()
    report = {"format": FORMAT, "title": title, "scope": stats.get("scope", "User-supplied counts"),
              "provenance": stats.get("provenance", {}),
              "series": [{"name": name, **row} for name, row in zip(names, metrics)],
              "definitions": {"nominal_usage": "nonzero histogram bins / vocabulary size",
                  "effective_support": "exp(-sum(p*log(p))) / vocabulary size",
                  "codes_for_50": "smallest ranked prefix carrying at least 50% of tokens",
                  "aggregation": "Count-pool all selected cells before computing entropy"}}
    (root / "token_support.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
