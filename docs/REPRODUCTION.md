<!-- Geometric Orbit Representatives for Unaligned 3D Tokenization -->
<!-- Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig -->
<!-- Contact: Xiang Chen <xiang.chen@uni-leipzig.de> | SPDX-License-Identifier: MIT -->

# Paper Reproduction

[Back to Main README](../README.md)

The release separates reference-checkpoint evaluation, functional training
tests and training a benchmark model from scratch. These are different checks.
The [machine-readable audit](../data/reproduction_audit.json) records the local
verification on 25 October 2026, including artifact hashes and unverified items.
Weights are prepared locally but **not published**; full benchmark geometry and
original token arrays are not bundled. A source checkout with four example
meshes alone cannot reproduce the paper's quantitative results.

## Verified Results

| Check | Observed result | What was rerun |
| --- | --- | --- |
| Fixed-mask prior, 512 held-out fields | At 50% masking: **97.23%** bit accuracy, **78.01%** token exactness | New inference using the migrated `matched_completion` pair and original token arrays. All five masking ratios and the LFQ-state table agree at the reported precision; 48 protocol/metric checks pass. |
| Original versus release model | All 153 active tensor values match for both pairs; frames agree on 16 real-object cases | Forward passes on the same full-size inputs, plus a full-objective backward pass per pair. CUDA token indices agree in 15/16 cases; maximum quantized SDF difference is 7.18e-5. This is numerical compatibility, not bitwise CUDA determinism. |
| CUDA training and decoding | Tokenizer update, eight-field export, prior update and decoded completion succeed | Bounded real-data workflow with reduced point/query budgets, not converged training. |
| Reference-weight geometry | Reconstruction and completion succeed on two held-out objects each, at resolution 64 | Newly sampled/exported inputs; raw/continuous/quantized and oracle/hard/relaxed paths all yield nonempty meshes. Functional checks, not a quality estimate. |
| VQ/LFQ support | Effective support **4.5% / 96.2%**; 50%-mass codes **33 / 3,218** | Recompute entropy and ranked mass from archived integer histograms. No new VQ/LFQ training or extraction. |
| Independent-jitter stability | Fresh point-only evaluation on **2,461 ShapeNet / 8,192 Objaverse-LVIS / 2,882 ScanObjectNN** observations | New raw-data inference under the fully specified release protocol. The older archive used by the previous audit is deprecated; see [protocol history](BENCHMARKS.md#historical-stress-results). |
| Decoded drift at 75% masking | Published **5 / 6 / 8**; fresh seeded rerun **5 / 8 / 8**, all 512 meshes valid | New prior inference, regeneration on the archived 56-cubed grid and mesh resampling. Small threshold-borderline differences remain; no exact-value claim. |

## Mesh Implementation And Fidelity

The 26 October [mesh audit](../data/mesh_implementation_audit.json) traces the
original quantitative decoder, reconstruction evaluator and presentation preset
separately. On six identical real observations, 153 active tokenizer tensors
match, and all 36 oracle/soft/hard comparisons across archived and fresh caches
produce identical tri-plane and SDF values in the reference PyTorch 2.7 runtime.
Original and release meshes have matching validity and vertex/triangle counts;
maximum same-field nearest-vertex discrepancy is 1.49e-8 model units.
The fresh hard-bit field with no isosurface also has no isosurface in the
original implementation. This is a bounded implementation comparison, not
a claim of deterministic GPU encoding across every input or backend.

The general evaluator and raw inference export previously used `[-0.6, 0.6]`;
they now default to the original `[-0.5, 0.5]` cube. An explicit `--bounds`
option preserves replay of earlier reports. The quantitative boundary decoder
already used the correct cube and is unchanged. The original gallery's
extent-preserving component rule is now available as a vectorized, optional
[export preset](INFERENCE.md#paper-mesh-export), without smoothing. The first
two full-shape test objects both export closed, nonempty meshes at resolution
160; this checks the public CLI, not population-level quality.

The paper's Table 2 uses the **matched** tokenizer, not the long-running
full-shape branch. Its evaluator encodes a 2,048-point observation, estimates
the frame from that subset, and uses the same representative observation as
the mesh reference. The general evaluator's full-cloud frame and independent
reference surface do not implement that contract. The new
[`benchmark fidelity`](BENCHMARKS.md#decoded-fidelity) command evaluates it
explicitly. On all 512 original test IDs, with resolution 64 and recorded draws:

| Quantized Reconstruction | Paper | Original Matched Weights, Rerun | Retrained, 12 Epochs |
| --- | ---: | ---: | ---: |
| Valid meshes | 512 / 512 | 512 / 512 | 512 / 512 |
| CD x 1e3, mean | 0.5343 | 0.5292 | 0.5617 |
| EMD x 100, mean | 5.2240 | 5.2824 | 5.3262 |
| Signed-query accuracy | 80.00% | 81.72% | 79.97% |
| Inside-fraction gap, pp | 6.76 | 5.95 | 9.42 |

The retrained quantized CD and EMD are respectively 5.1% and 2.0% above the
paper values, without additional training. Signed-query accuracy is close;
the inside-fraction gap remains larger. Raw and continuous branch results,
artifact hashes and exact metric definitions are retained in the audit. These
are deterministic new draws under the source-compatible measurement contract,
not replay of the original unrecorded worker/global RNG stream. They do not
replace the older, differently defined general-evaluator results below or
establish the fresh-token relaxed-boundary advantage.

The separate full-shape continuation reaches checkpoint `epoch=105`; the
paper-facing full-shape tokenizer remains `epoch=07`. In the continuation
directory, `last.ckpt` is stale at epoch 11, while `last-v1.ckpt` is epoch 105.
Checkpoint metadata, rather than filenames alone, determines lineage. A
historical matched-prior diagnostic found that the longer continuation improved
reconstruction and support but reduced token predictability. The audit records
that comparison separately; it is not a reason to require 100-plus epochs for
the matched 12-epoch recipe.

## Distance Refinement Results

The subsequent [quality audit](../data/quality_refinement.json) keeps the original
twelve-epoch tokenizer and changes only its existing shared field-head bias.
The offset is fit with balanced Huber SDF supervision on 128 seeded training
objects, bounded to +/-0.003 model units. All 205 disjoint validation objects
check signed-field guards and mesh fidelity before the single 512-object test
evaluation. The balanced candidate improves validation CD, EMD, sign accuracy
and inside-gap over both the original weights and unweighted calibration;
all three have 205 valid validation meshes. No test metrics select the offset.

| Quantized Reconstruction, 512 Objects | Paper | Before Calibration | Distance-Calibrated |
| --- | ---: | ---: | ---: |
| CD x 1e3 | 0.5343 | 0.5617 | 0.5400 |
| EMD x 100 | 5.2240 | 5.3262 | 5.2291 |
| Signed-query accuracy | 80.00% | 79.97% | 80.48% |
| Inside-fraction gap, pp | 6.76 | 9.42 | 7.64 |
| Valid meshes | 512 / 512 | 512 / 512 | 512 / 512 |

Quantized CD and EMD are respectively 1.1% and 0.1% above the paper values;
inside-gap remains 0.87 pp larger. The raw branch reaches CD 0.5230 and EMD
5.3054, compared with paper values 0.5347 and 5.3128. All branches and tails
are retained in the audit. These are new deterministic measurements under the
source-compatible protocol, not the original unrecorded RNG stream.

This is an optional post-training calibration, not unchanged-protocol
reproduction. Encoder and quantizer tensors remain identical, no smoothing is
applied, and the original source checkpoint is preserved. The full identity
hash changes; the original completion pair is not silently rebound to the
calibrated field. The [commands](TRAINING.md#distance-calibration) expose the
fit and validation guards.

A separate two-epoch, decoder-only optimization took 189 seconds but worsened
validation geometry, including after explicitly matching observation/query
budgets. It was rejected before test evaluation; its weights, config and source
snapshot remain local, not in the default public workflow. The training engine
and primary model architecture are unchanged by this experiment.

Posterior temperature calibration of the full-shape adapted prior lowers
validation BCE slightly. Its fresh 512-object test still has relaxed edge
distance 0.01688 versus 0.01637 for hard bits, with 511/512 valid meshes per
mode. It does not recover the relaxed-versus-hard edge advantage. The original
prior remains the default; probability calibration is optional and is not
presented as a geometry improvement. No test-driven temperature search followed.

## From-Scratch Validation

The initial primary-completion workflow completed on 25 October 2026: **12 tokenizer
epochs and 80 epochs per prior**, with no pretrained initialization or skipped
updates. The original 4,096 training IDs were divided into 3,891 train and 205
validation observations; all 512 test IDs were preserved. This single-GPU run
used bf16, effective tokenizer batch 16, bit-prior batch 8 and categorical batch 2.
The seven workflow steps took about **2 h 39 min** on an RTX PRO 6000 Blackwell;
this is not a dedicated-hardware throughput measurement. Its downstream cache
used the validation-selected tokenizer after seven completed epochs, not the
final twelve-epoch tokenizer. The measurements below retain that initial run;
the corrected reproduction workflow uses fixed-budget `last.pt` checkpoints.

Both new and reference tokenizers were evaluated on the same source observations,
with newly extracted caches, identical test ordering and identical masks. The
reference pair below uses the original matched-completion weights, not the
full-shape reconstruction pair.

| Held-Out Measurement | Initial Training | Reference Pair |
| --- | ---: | ---: |
| Bit accuracy, 50% masking | 98.14% | 97.23% |
| Exact tokens, 50% masking | 83.83% | 78.03% |
| Quantized mesh validity | 512 / 512 | 512 / 512 |
| Reconstruction CD x 1e4, mean | 5.347 | 5.237 |
| Reconstruction CD x 1e4, median / p95 | 3.516 / 14.443 | 2.815 / 12.695 |
| LFQ effective support, matched K=4096 | 4.75% | 9.37% |

The reconstruction rows compare predicted meshes with stored reference surface
points in radius-0.5 model coordinates at resolution 56. Their CD is the general
evaluator's symmetric mean squared distance, not the bbox-normalized decoded-oracle
CD used by the boundary table. Mean reconstruction CD differs by about 2.1%;
the median and tail differences are larger. The reference token-exact value
differs slightly from the archived 78.01% because tokens were re-extracted.

The initial completion accuracies are retained as measured, including improvements
over the paper reference. The tokenizers have different distributions: the fresh
tokenizer's support is more concentrated. Its categorical prior reaches 84.27%
exact tokens at 50% masking, slightly above its bit prior. At 75%, all three
decoded paths yield 512 valid meshes and zero drift failures on this matched
cohort; relaxed edge distance is 0.01646 versus 0.01626 / 0.01631 for hard bits /
categorical indices. This run therefore does not establish the relaxed-boundary
advantage measured with the separate full-shape pair. Nor is its K=4096 support
the paper's K=8192, 55-category comparison.

The [path-free training audit](../data/full_training_validation.json) records
budgets, selected checkpoints, source/data hashes, all five masking ratios,
support, reconstruction and decoded metrics. Full logs, weights and meshes
remain local under `outputs/paper_training_20261025/`; they are not bundled or
published. These checks establish an executable from-scratch primary recipe
and comparable geometry, not independent reproduction of every paper claim.

## Reproduction Recovery

The initial tokenizer's early checkpoint was selected using untruncated SDF
error, although its training objective uses truncated targets. On the separate
205-object validation split, the complete twelve-epoch checkpoint lowers
truncated uniform SDF MAE from 0.00756 to 0.00627 and increases effective support
from 4.33% to 9.75%. Selecting the fixed-budget checkpoint corrects the workflow;
it does not change or retrain the tokenizer. On all 512 test observations,
effective support rises from **4.75% to 10.00%**, compared with **9.37%** for
the original matched checkpoint. Nominal usage remains lower, 72.4% versus
95.0%; the distributions are not identical. This is the K=4096 matched control,
not the K=8192 full-test support experiment.
Support increases over the early checkpoint in all eight categories. Relative
to the reference, category-level support still ranges from 0.69x to 1.20x;
the aggregate recovery does not imply uniform category-level agreement.

We then trained priors on the complete tokenizer's newly exported tokens.
An 80-epoch bf16/per-update-schedule control and an 80-epoch FP32/epoch-schedule
control both retain good 50%-mask performance but reach only 24.85% and 24.60%
exact tokens at 90% masking. Restoring the original precision and schedule
therefore does not resolve sparse-context degradation. The reference and
release LFQ implementation files are byte-identical; the package version alone
does not explain these results.

The paper profile samples masks in `[0.05, 0.60]`, with a separate 5%
unconditional case. A prespecified 20-epoch refinement extends that range to
`[0.05, 0.95]`, without changing architecture or loss. It starts from the
FP32 control's final checkpoint, resets AdamW, uses learning rate `2e-5`
decaying to `1e-6`, and retains the fixed-budget final checkpoint. On the same
512 test fields and identical masks it reaches **95.98% bit accuracy / 72.86%
exact tokens at 90% masking**, while retaining **97.97% / 82.40% at 50%**.
This is added training coverage and budget, not unchanged-protocol reproduction
or evidence that the underlying representation improves over the paper.
The [refinement commands](TRAINING.md#sparse-context-refinement) and separate
original-mask continuation control make these distinctions explicit.

| Prior On The Same Final Tokens | Total Epochs | 50% Exact Tokens | 90% Bit Accuracy | 90% Exact Tokens |
| --- | ---: | ---: | ---: | ---: |
| FP32, epoch schedule, masks 5%-60% | 80 | 82.59% | 86.03% | 24.60% |
| Continue with masks 5%-60% | 80 + 20 | 82.65% | 86.16% | 24.68% |
| Refine with masks 5%-95% | 80 + 20 | 82.40% | 95.98% | 72.86% |

Both continuations start from the same checkpoint with identical optimizer,
learning-rate schedule, seed, batch size and budget. Only mask coverage differs.
Additional original-mask optimization alone does not recover 90%-mask accuracy
in this run. Checkpoint selection, precision/schedule and mask coverage were
tested separately; no single control explains every difference from the paper.

| Quantized Reconstruction, Same 512 Objects | Initial Checkpoint | Complete Tokenizer | Reference Pair |
| --- | ---: | ---: | ---: |
| Valid meshes | 512 / 512 | 512 / 512 | 512 / 512 |
| CD x 1e4, mean | 5.347 | 5.307 | 5.237 |
| CD x 1e4, median | 3.516 | 3.062 | 2.815 |
| CD x 1e4, p95 | 14.443 | 14.965 | 12.695 |

These are the general reconstruction evaluator's model-space distances at
resolution 56, not decoded-oracle drift metrics. Mean CD is within 1.3% of
the reference; median error improves over the initial selection, but the tail
remains worse. Recovering support does not establish identical geometric
quality or reproduce every paper claim.

All runs and unfavorable controls are retained in the
[recovery audit](../data/reproduction_recovery.json), including configuration,
checkpoint and data hashes, validation histories, category-level support and
test metrics. Decisions use training/validation evidence and fixed budgets,
not test-set checkpoint selection.

## Full-Shape Boundary Validation

The full-shape checkpoint pair is distinct from the new primary-completion run.
We repeated prior inference and mesh decoding on the original 512 token maps
with the original 75% mask, then evaluated the fixed meshes with five
prespecified surface-sampling seeds (20260518-20260522), 1,024 points per surface
and the same edge definition. All 512 meshes remain valid in every path.

| Decode Path | Paper Edge Distance | Rerun Edge, Mean Over Sampling Seeds |
| --- | ---: | ---: |
| Relaxed LFQ | 0.01814 | **0.01931** |
| Thresholded bits | 0.01835 | 0.01950 |
| Categorical indices | 0.01979 | 0.02098 |

Relaxed mean edge error is lower in **all five sampling repeats**, by about
1.0% relative to hard bits and 7.9% relative to categorical indices after
averaging repeats. Drift counts range from 4-6 / 6-8 / 7-8 for relaxed / hard /
categorical, versus the paper's 5 / 6 / 8. These are surface-sampling repeats,
not independent training seeds. Paired object-bootstrap 95% intervals include
zero; the result recovers the average ordering, not a new significance claim.

We also tested geometry-to-token extraction on the same 512 object IDs, with
unchanged checkpoints and the same saved mask. Neither portable sampling nor
the optional reference-algorithm adapter retained the mean-edge advantage in
these runs. Edge distances were 0.01848 / 0.01818 / 0.01646 for portable
extraction and 0.03239 / 0.03217 / 0.01628 for reference-algorithm extraction
(relaxed / hard / categorical). One categorical mesh is invalid in each run;
its absence remains in failure-inclusive counts, and its valid-only edge mean
does not cover the full cohort. The adapter matches the original per-cell
selection algorithm under a recorded local seed; it does not recover the
unrecorded original global/worker RNG stream.

Thus the fixed-cache full-shape protocol reproduces the relaxed-edge trend,
while re-extraction sensitivity remains a limitation of this frozen interface.
No objects or unfavorable runs were removed. The
[boundary audit](../data/boundary_reproduction.json) includes every run,
paired intervals, binding/mask checks and artifact hashes. Repeat the analysis
with [the benchmark commands](BENCHMARKS.md#boundary-repetition).

### Paired Fresh-Token Adaptation

A further run re-extracted tokens for 4,096 training, 205 validation and all
512 test objects. Training covers all 55 categories. Validation objects come
from the original held-out subset of both priors, not from their prior training
objects. With the tokenizer frozen, both priors receive ten adaptation epochs,
batch size 8, masks 5%-95% and learning rate `2e-5` decaying to `1e-6`.
Final checkpoints, original test masks and resolution 56 are fixed before
evaluating geometry. All 512 oracle mesh hashes agree before and after adaptation.
This is warm-start adaptation, not independent full-shape retraining.

| Fresh-Token Run | Decode | Valid / 512 | Failures / 512 | Edge, Mean Over Five Seeds |
| --- | --- | ---: | ---: | ---: |
| Frozen reference priors | Relaxed | 512 | 4 | 0.01846 |
| Frozen reference priors | Hard | 511 | 3 | 0.01644 |
| Frozen reference priors | Categorical | 511 | 3 | 0.01649 |
| Paired adaptation | Relaxed | 511 | 3 | 0.01686 |
| Paired adaptation | Hard | 511 | 2 | 0.01637 |
| Paired adaptation | Categorical | 511 | 4 | 0.01650 |

Failures include invalid meshes as well as drift-threshold violations; edge
means exclude invalid meshes and therefore need the validity column. The
five seeds repeat surface sampling on fixed meshes, not model training.
The lower relaxed mean after adaptation is partly a selection effect: a
previously valid outlier becomes invalid. On the same 511 valid objects,
relaxed edge is **0.01670 before / 0.01686 after**. Adaptation does not improve
this paired comparison, and relaxed edge remains above hard and categorical
edge in all five repeats. Its paired bootstrap intervals for these differences
remain positive. **The fresh-token relaxed-boundary advantage is not reproduced.**

The new and archived caches agree on 94.2% of bits but only 64.8% of complete
tokens for the same objects. Repeated exports with the new sampling protocol
agree on 99.98% of bits; spot-checked point inputs and frames match exactly.
Thus extraction protocol differences are substantial, not interchangeable
with exact replay of the archived token arrays. The original export's RNG
stream was not preserved. The [recovery audit](../data/reproduction_recovery.json)
retains fresh-token controls, five-seed metrics, common-valid comparisons,
failure-inclusive counts and paired object-bootstrap intervals. Higher prior
accuracy alone does not establish a decoded-boundary improvement.

## Fixed-Mask Token And State Evaluation

Use the **matched_completion** tokenizer/prior pair, not the full-shape
`reconstructive` pair. The paper used 512 token maps of shape 128 x 128 with
12 bits per cell. Evaluation shuffles the explicit test array with NumPy seed
20260511, then samples independent Bernoulli cell masks with a local **CPU**
Torch generator seeded by `20260511 + round(mask_ratio * 10000)`. The reference
batch size is 16; all bits of a hidden cell are replaced with 0.5.

For an original `tokens.npy` and its `metadata.json`:

```bash
orbit evaluate tokens \
  --prior checkpoints/official/matched_completion/prior.pt \
  --tokenizer checkpoints/official/matched_completion/tokenizer.pt \
  --indices data/reference_tokens/tokens.npy \
  --indices-metadata data/reference_tokens/metadata.json \
  --allow-unverified-indices --split test --device cuda --mask-device cpu \
  --seed 20260511 --batch-size 16 --max-samples 512 \
  --output outputs/paper_tokens

python tools/check_paper_reproduction.py \
  --report outputs/paper_tokens/token_prior.json \
  --output outputs/paper_tokens/precision_check.json
```

The legacy metadata lacks a tokenizer hash, so its use requires the explicit
`--allow-unverified-indices` flag. The report records this unverified binding
alongside hashes of the array, metadata and prior; the flag cannot override a
known hash mismatch. It does not establish training-set independence. A newly
exported release cache has a verified binding and needs no override:

```bash
orbit evaluate tokens --tokens data/tokens --split test \
  --prior outputs/prior/last.pt --device cuda --output outputs/token_metrics
```

The checker compares measurements against the
[reported reference values](../data/paper_reference.json); the evaluator never
reads those values when computing predictions. Fresh token extraction can
change thresholded bits through sampling or floating-point differences, even
with the same checkpoint. Matching the paper therefore also requires its
observations, frame caches and extraction protocol.

Bit accuracy pools correct hidden bits; token exactness requires all 12 bits of
a hidden cell to match. BCE and confidence are averaged over hidden bits.
Uncertain states have `abs(2p-1) <= 0.9`; wrong-bit subsets use thresholded
predictions. MSE reductions compare relaxed `2p-1` with hard +/-1 states against
the oracle. Thin/edge diagnostics concern adjacent **LFQ states**, not mesh edge
distance. The report includes numerators, denominators and mask/selection hashes.

`orbit evaluate completion` remains the general geometry diagnostic: it uses
exact-count masks and model-space field/mesh metrics. It is not a replacement
for the paper's fixed-Bernoulli token protocol or its bbox-normalized decoded
boundary table. Use [`orbit benchmark decode` and `geometry`](BENCHMARKS.md#decoded-boundary-metrics)
for the paper boundary definitions and common-mask mesh cohort.

## Geometry And Training

The [import adapter](DATA.md#existing-sdf-data) supports the original SDF schema,
explicit legacy uniform-query offsets and full-observation frame caches.
Input coordinates **and** signed distances are divided by `input_scale` once.
The original archive's uniform suffix starts at query 70000; this is an explicit
opt-in convention, not a guess to apply to unrelated NPZs.

For reference-architecture training, use
[tokenizer_paper.yaml](../configs/tokenizer_paper.yaml) or
[tokenizer_matched.yaml](../configs/tokenizer_matched.yaml), followed by a newly
exported token cache and [prior_paper.yaml](../configs/prior_paper.yaml). The
training seed defaults to the config's `seed`, otherwise 2026; `--seed` overrides
it. The prior preset uses 20260511. Preserve data IDs, normalization, frame
policy, vocabulary and tokenizer/prior pairing when comparing results.

The trainer supports explicit step- or epoch-wise cosine schedules; paper
presets use the original epoch-wise interval. The initial run above used
step-wise scheduling, BF16 and a validation-selected tokenizer. Deterministic
portable sampling and a validation holdout still differ from the original
trainer's sampler RNG and full training cohort. Compatible weights/objectives
do not establish identical optimization trajectories.

## Remaining Limits

- The primary matched-completion recipe has completed full from-scratch training
  and evaluation. One seed is not a convergence guarantee or a full-paper reproduction.
- [VQ and categorical-prior training presets](BASELINES.md) are available and
  functionally tested, including strict original-weight compatibility. Their
  categorical LFQ-cache control also completed 80 epochs. Fresh VQ training to
  convergence and the 8,192-code, 55-category support comparison remain unverified.
- Full raw-data stress and fresh 512-object decoded mesh comparisons have now
  been rerun. Their reported protocol/sampling differences remain explicit;
  they are not silently forced to match historical values.
- CUDA training was checked on one GPU; CPU/Gloo DDP is tested, but CUDA/NCCL
  multi-GPU hardware execution and throughput remain unmeasured.
- Small CUDA scatter/rounding differences can flip near-zero LFQ bits. Do not
  promise bitwise equality across GPU runs or library versions.

See [validation](VALIDATION.md) for the executable test coverage and
[pretrained pairs](PRETRAINED.md) for artifact identity and publication status.

[Back to Main README](../README.md#evaluation)
