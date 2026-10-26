<!-- Geometric Orbit Representatives for Unaligned 3D Tokenization -->
<!-- Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig -->
<!-- Contact: Xiang Chen <xiang.chen@uni-leipzig.de> | SPDX-License-Identifier: MIT -->

# VQ And Categorical Baselines

[Back to Main README](../README.md)

Two controls are available through the same training engine:

| Control | Configuration | Changed component |
| --- | --- | --- |
| Learned-centroid VQ | `tokenizer_vq_demo.yaml` / `tokenizer_vq_matched.yaml` | `Tokenizer.type: standard_vq`; same geometric representative, encoder, plane decoder and SDF objectives as the corresponding LFQ preset |
| Categorical index prior | `prior_categorical_demo.yaml` / `prior_categorical_paper.yaml` | `Prior.type: categorical`; masked-index cross entropy instead of independent-bit BCE |

The VQ implementation uses the installed `VectorQuantize` library, with the
original learned-centroid/EMA configuration: commitment weight 0.25, Euclidean
matching and default EMA decay 0.8. LFQ uses its own hypercube quantizer and
auxiliary objective. These quantizer-specific terms differ by design; the
surrounding reconstruction settings are matched.

Categorical priors work with **either** an LFQ cache or a VQ cache. They predict
hard code indices and decode through the originating frozen tokenizer. VQ
indices do not have LFQ bit semantics: there is no VQ bit accuracy or relaxed
bit decoder. No category labels enter the categorical network.

## Run On The Examples

First complete the [Quick Start preprocessing](../README.md#quick-start).
The following performs a small CPU functionality check, not a quality benchmark:

```bash
orbit train-tokenizer --config configs/tokenizer_vq_demo.yaml \
  --data data/samples/processed --output outputs/vq_demo \
  --device cpu --max-steps 2

orbit cache-tokens --checkpoint outputs/vq_demo/last.pt \
  --data data/samples/processed --output outputs/vq_tokens --device cpu

orbit train-prior --config configs/prior_categorical_demo.yaml \
  --data outputs/vq_tokens --output outputs/categorical_demo \
  --device cpu --max-steps 2

orbit evaluate tokens --tokens outputs/vq_tokens \
  --prior outputs/categorical_demo/last.pt --split test \
  --output outputs/categorical_metrics

orbit evaluate completion --data data/samples/processed \
  --tokens outputs/vq_tokens --tokenizer outputs/vq_demo/last.pt \
  --prior outputs/categorical_demo/last.pt --resolution 32 \
  --mask-ratio 0.75 --output outputs/vq_completion

orbit plot-training --run outputs/categorical_demo --data outputs/vq_tokens
orbit analyze-support --tokens outputs/vq_tokens --split test \
  --output outputs/vq_support
```

`last.pt` supports exact resume; `best.pt` tracks the configured held-out score:
50%-mask CE in the demo, mean CE across five mask ratios in the paper preset.
Fixed-budget reproduction uses `last.pt`.
Logs/plots report CE, exact-token accuracy and the hidden-cell fraction, not
invented bit metrics. The checkpoint preview shows target, observed and
completed index maps plus errors on hidden cells. Index colors identify codes;
their numeric order has no geometric meaning. Untrained decoders can have no
zero-level surface; evaluation reports this failure explicitly.

## Reference-Scale Training

```bash
orbit train-tokenizer --config configs/tokenizer_vq_matched.yaml \
  --data data/imported --output outputs/vq --device cuda --workers 4

orbit cache-tokens --checkpoint outputs/vq/last.pt --data data/imported \
  --output data/vq_tokens --device cuda

orbit train-prior --config configs/prior_categorical_paper.yaml \
  --data data/vq_tokens --output outputs/vq_categorical \
  --device cuda --batch-size 1 --workers 4
```

Use `configs/tokenizer_matched.yaml` for the matched LFQ reconstruction control.
Both tokenizer presets use K=4096. For a new K=8192 comparison, set
`Tokenizer.codebook_size: 8192` in **both** tokenizer configs, and set
`vocab_size` and `Tokenizer.codebook_size` to 8192 in the categorical prior.
LFQ bit priors additionally need `num_bits: 13`. Rebuild caches and priors for
each tokenizer. Changing K does not reproduce the archived 8192-code support
experiment without also matching its data, training and extraction protocol.

Both training commands accept `torchrun`, for example:

```bash
torchrun --standalone --nproc-per-node=2 -m orbit_folding train-prior \
  --config configs/prior_categorical_paper.yaml --data data/vq_tokens \
  --output outputs/vq_categorical_ddp --device cuda --batch-size 1
```

Batch size is per process. Categorical logits have B x H x W x K elements;
start with batch size 1 at 128 x 128, K=4096. Training uses hidden-cell-count
weighted gradients across ranks. The VQ library synchronizes codebook EMA
statistics across distributed workers. CPU/Gloo execution is tested for both
baselines; CUDA/NCCL multi-GPU hardware remains unverified.

## Shared-Interface Prior Comparison

Train the bit prior and categorical prior on the **same LFQ cache**, keeping
its split and tokenizer checkpoint fixed:

```bash
orbit train-prior --config configs/prior_paper.yaml --data data/lfq_tokens \
  --output outputs/lfq_bits --device cuda
orbit train-prior --config configs/prior_categorical_paper.yaml \
  --data data/lfq_tokens --output outputs/lfq_indices --device cuda --batch-size 1
```

Evaluate each with the same seed, mask ratio, split and frozen decoder using
`orbit evaluate completion`. Bit priors expose oracle/hard/relaxed fields;
categorical priors expose oracle/categorical-index fields. For token-only
evaluation, `orbit evaluate tokens` gives both priors identical CPU Bernoulli
masks at the same seed and batch size. Categorical rows report pooled hidden
CE, exact-token accuracy and maximum class probability. They do not report
LFQ-state MSE reductions.

The paper categorical preset preserves the original learned token/row/column
embeddings, 160-channel backbone, ten dilated residual blocks and dropout 0.06.
It matches the bit prior's backbone width/depth, **not** parameter count or
compute: its vocabulary embedding and K-way head are larger. CE and BCE have
different denominators and should not be compared as if they were the same loss.

VQ raw-input inference also uses `orbit encode` / `orbit decode`. Encoded VQ
files contain integer HW arrays, not bit planes. A paired categorical prior
defaults to hard decoding; requesting `--state relaxed` is an error.

## Verification And Scope

The original VQ checkpoint loads with all active tensor shapes intact; an
original-versus-release check found identical indices, quantized planes and
hard-index decoding on its audited feature input. The original categorical
checkpoint likewise preserves all active weights and produces identical logits
and predictions on an audited 128 x 128 map. These local comparisons used
PyTorch 2.7.0+cu128, not a promise of cross-platform bitwise determinism.

Single-GPU checks ran the complete small VQ/index-prior workflow, plus one
effective update of the full 4096-way, 128 x 128 categorical preset. Unit/CLI
tests cover arbitrary VQ vocabulary sizes, observed-token preservation,
content-bound caches, held-out validation, previews, exact CPU resume and
two-process training. These are implementation checks. Neither baseline has
been independently validated as a reproduction of every paper baseline result.
The categorical LFQ-cache control subsequently completed **80 epochs** on the
primary matched split, reaching **84.27%** exact tokens at 50% masking. Its
[full training audit](REPRODUCTION.md#from-scratch-validation) also reports
geometry and support, rather than treating token accuracy as a complete quality
measure. Fresh VQ training to convergence and the K=8192 support control remain
unverified.

[Back to Main README](../README.md#training)
