# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Histogram definitions, split-bound cache counts and honest paper redraws."""

import json

import numpy as np
from PIL import Image
import pytest

from orbit_folding.token_support import FORMAT, plot_support, summarize_counts


def test_uniform_histogram_has_full_support():
    row = summarize_counts([4, 4, 4, 4])
    assert row["total_tokens"] == 16
    assert row["nominal_usage"] == row["effective_support"] == 1
    assert row["codes_for_50"] == 2 and row["codes_for_90"] == 4


def test_touched_codes_can_still_have_concentrated_mass():
    row = summarize_counts([97, 1, 1, 1])
    assert row["nominal_usage"] == 1 and row["effective_support"] < 0.3
    assert row["codes_for_50"] == 1


def test_unused_bins_and_count_pooling_keep_the_denominator():
    pooled = summarize_counts(np.array([8, 0, 0, 0]) + [0, 8, 0, 0])
    assert pooled["codebook_size"] == 4
    assert pooled["effective_support"] == 0.5
    assert summarize_counts([8, 0, 0, 0])["effective_support"] == 0.25


@pytest.mark.parametrize("counts", [[], [0, 0], [1, -1], [1.5, 2], [[1, 2]],
                                    [True, False], [np.nan], [2**53 + 1],
                                    [2**53, 1], [2**62, 2**62, 2**62, 2**62]])
def test_invalid_histograms_are_rejected(counts):
    with pytest.raises(ValueError):
        summarize_counts(counts)


@pytest.mark.parametrize("second", [[1, 1, 2], [2, 2]])
def test_comparison_rejects_unmatched_histograms(tmp_path, second):
    source = {"format": FORMAT, "series": [{"name": "A", "counts": [1, 1]},
                                             {"name": "B", "counts": second}]}
    with pytest.raises(ValueError, match="matched"):
        plot_support(source, tmp_path)


def test_paper_counts_reproduce_published_support_and_rank_counts(project_root, tmp_path):
    source = project_root / "data/paper_token_support.json"
    report = plot_support(source, tmp_path)
    vq, lfq = report["series"]
    assert vq["total_tokens"] == lfq["total_tokens"] == 5341184
    assert vq["codebook_size"] == lfq["codebook_size"] == 8192
    assert 100 * vq["nominal_usage"] == pytest.approx(96.3, abs=0.05)
    assert 100 * vq["effective_support"] == pytest.approx(4.5, abs=0.05)
    assert 100 * lfq["effective_support"] == pytest.approx(96.2, abs=0.05)
    assert vq["codes_for_50"] == 33 and lfq["codes_for_50"] == 3218
    image = np.asarray(Image.open(tmp_path / "token_support.png"))
    assert image.shape[1] >= 2000 and image.std() > 10
    assert (tmp_path / "token_support.pdf").stat().st_size > 1000
    assert json.loads((tmp_path / "token_support.json").read_text())["provenance"]["shapes"] == 5216


def test_standalone_support_script(project_root, tmp_path, run_python):
    run_python("tools/plot_token_support.py", "--stats", project_root / "data/paper_token_support.json",
               "--output", tmp_path)
    assert (tmp_path / "token_support.png").exists()
