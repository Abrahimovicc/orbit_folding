<!-- Geometric Orbit Representatives for Unaligned 3D Tokenization -->
<!-- Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig -->
<!-- Contact: Xiang Chen <xiang.chen@uni-leipzig.de> | SPDX-License-Identifier: MIT -->

# Codebook Support

[Back to Main README](../README.md)

Nominal usage measures how many codes occur at least once. It does not measure
how evenly they are used. Given pooled integer counts `c` over **all K codes**:

- Nominal usage: `count_nonzero(c) / K`.
- Entropy: `H = -sum(p * log(p))`, with `p = c / sum(c)` and zero terms omitted.
- Perplexity: `exp(H)`; effective support: `exp(H) / K`.
- Codes for 50% mass: the smallest frequency-ranked prefix reaching half the tokens.

Pool counts before computing entropy. Averaging per-batch support gives a
different statistic. Retain zero-count bins; trimming them changes K and can
artificially inflate nominal and effective support. Empty observations are an
error, not evidence of either collapse or broad support.

## Paper Reference

[paper_token_support.json](../data/paper_token_support.json) contains actual
frequency-ranked histograms behind paper Figure 5. They cover 5,216 ShapeNet
test shapes, 55 categories, matched `K=8192`, and 5,341,184 token cells per
quantizer. Exact integer counts were recovered from the saved probabilities;
both integrality and totals were checked. No missing curve was synthesized.
Code identities are omitted because rank statistics do not require them.

```bash
orbit analyze-support --stats data/paper_token_support.json --output outputs/paper_support
```

This redraws published measurements; it does **not** train VQ or rerun the
benchmark. The public training path implements LFQ, not a newly adapted VQ
baseline. See the paper citation in [README](../README.md#citation).

## Your Token Cache

```bash
orbit analyze-support --tokens outputs/tokens_demo --split test --output outputs/demo_support
```

The tool validates the cache, converts MSB-first bit planes into integer
indices, and vectorizes cell counting with `bincount`. It preserves the
explicit selected split and records its sample count and tokenizer hash.
Each sample is loaded separately, bounding memory by one token map plus the
histogram. Dense histograms are limited to at most `2**20` codes.

The four-object demo is a pipeline check. A high usage fraction from one tiny
token map is not evidence that a full trained tokenizer avoids collapse.

## Custom Comparison

Provide one to four histograms, including unused codes, in this JSON schema:

```json
{
  "format": "orbit_folding.token_support.v1",
  "title": "Matched tokenizer comparison",
  "scope": "Describe the actual dataset, split and measurement protocol",
  "provenance": {},
  "series": [
    {"name": "Tokenizer A", "counts": [8, 0, 0, 0]},
    {"name": "Tokenizer B", "counts": [2, 2, 2, 2]}
  ]
}
```

These small arrays only illustrate the file format, not experimental results.
`counts` must be nonnegative integers, with a positive total no greater than
`2**53`. The comparison rejects different K or token counts; equal totals
alone do not establish equal data, sampling, representation or training
budgets. Those controls must be documented in the supplied provenance.

Outputs are `token_support.png`, `token_support.pdf` and
`token_support.json`, including full cumulative curves and metric definitions.
The independent wrapper is [tools/plot_token_support.py](../tools/plot_token_support.py).

[Back to Main README](../README.md#token-support)
