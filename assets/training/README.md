<!-- Geometric Orbit Representatives for Unaligned 3D Tokenization -->
<!-- Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig -->
<!-- Contact: Xiang Chen <xiang.chen@uni-leipzig.de> | SPDX-License-Identifier: MIT -->

# Recorded Example Run

[Back to Main README](../../README.md)

Actual CPU runs, seed 2026, 64 optimizer updates per component, two training
objects and one validation object. The separate test object is not used in
these previews. Demo presets and reproduction commands are documented in the
[training guide](../../docs/TRAINING.md#example-outputs).
Preprocessing uses centroid normalization, 4,096 surface points, 8,192 query
points and 2,048 uniform queries per object. Each completed epoch evaluates the
same held-out sample; validation does not alter the training RNG state.

| Artifact | Source |
| --- | --- |
| [Tokenizer curves](tokenizer_curves.png) | [All 64 logged updates](tokenizer_metrics.jsonl) |
| [Prior curves](prior_curves.png) | [All 64 logged updates](prior_metrics.jsonl) |
| Tokenizer held-out scores | [64 validation records](tokenizer_validation.jsonl), [plot metadata](tokenizer_curves.plot.json); best at update 1 |
| Prior held-out scores | [64 validation records](prior_validation.jsonl), [plot metadata](prior_curves.plot.json); best at update 64 |
| [Tokenizer field slices](tokenizer_preview.png) | Final checkpoint, `mandarin_orange`, validation split, `z=0` |
| [Prior bit maps](prior_preview.png) | Final checkpoint, `mandarin_orange`, validation split, fixed 50% mask |

Representative console output, with the same values retained in the logs:

```text
tokenizer | epoch 001 | update 00001 | loss 301.3458 | SDF MAE 0.03845 | sign 60.2% | batch codes 15 | lr 9.99e-05
tokenizer | epoch 064 | update 00064 | loss 328.4883 | SDF MAE 0.03684 | sign 63.3% | batch codes 16 | lr 1.00e-06
prior | epoch 001 | update 00001 | loss 0.7383 | bit 49.04% | token 3.85% | hidden 20.3% | lr 2.00e-04
prior | epoch 064 | update 00064 | loss 0.6740 | bit 62.80% | token 12.20% | hidden 32.0% | lr 1.00e-05
```

These are pipeline diagnostics, not benchmark results. The tokenizer is
poorly trained at this budget; its validation slices have no zero crossing.
Minibatch prior accuracy varies with the sampled masks and is not a held-out
completion score. The figures show held-out validation in separate panels and
mark its best recorded score; no smoothing, fitted trends, or substitute
geometry is used. Validation sign accuracy on a mostly exterior uniform grid
does not establish surface reconstruction quality.

[Back to Main README](../../README.md#training)
