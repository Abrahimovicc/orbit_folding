<!-- Geometric Orbit Representatives for Unaligned 3D Tokenization -->
<!-- Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig -->
<!-- Contact: Xiang Chen <xiang.chen@uni-leipzig.de> | SPDX-License-Identifier: MIT -->

# Release Validation

[Back to Main README](../README.md)

Local verification on 25-26 October 2026 used Python 3.11, PyTorch 2.9.1,
Open3D 0.19.0 and vector-quantize-pytorch 1.25.2 for CPU tests. Additional
single-GPU runs used PyTorch 2.11.0+cu128 on an RTX PRO 6000 Blackwell. The
original/release parity audit used the reference's PyTorch 2.7.0+cu128.

## Automated Checks

The complete suite passed **467 tests**, including:

- SO(3), proper-frame and full-observation representative contracts;
- hard LFQ versus binary-probability decoding and relaxed-state gradients;
- finite gradients for every learned parameter, including zero curriculum weights;
- spatial masks, observed-bit preservation and exact hidden-bit/token counts;
- split leakage, malformed/nonfinite arrays and token-cache provenance;
- restricted checkpoint loading, strict original-weight conversion and atomic saving;
- single-process reconstruction and prior training through independent CLI subprocesses;
- bitwise CPU resume equivalence, including a mid-epoch partial accumulation group;
- nonempty-run protection, actual data-content fingerprints and stale-cache rejection;
- source-path aliases, cross-split byte duplicates and source overwrite protection;
- disconnected-shell and nested-cavity normals checked against signed distances;
- actual CPU GradScaler overflow, with no skipped optimizer/scheduler advancement;
- deterministic held-out validation and `best.pt` checkpoint selection;
- pooled support and truncated-SDF validation; batch/rank-independent
  five-mask validation; training-to-plot integration for both prior types;
- epoch-wise and per-update cosine schedules, exact mid-epoch resume and
  no schedule advancement in an all-skipped epoch;
- raw-input rotation consistency through normalization, spatial subsampling and encoding;
- explicit legacy input compatibility, full-observation PCA-only/locked frame policies,
  matched training/evaluation sampling, and cached-frame mismatch rejection;
- invalid token-index and bit-probability rejection, without wrapping or clipping;
- fresh weights-only initialization, preserved provenance on resume and strict prior binding;
- two-process CPU/Gloo fresh and initialized training for the tokenizer and completion prior;
- field, mesh and failure-report semantics, model-space bounds, extent-aware
  filtering, unchanged vertex locations and exported topology counts;
- independent LFQ/VQ fidelity CLI paths, observed-subset frame estimation,
  exact KD-tree CD/Hungarian EMD, zero-target signs and failed-sample denominators;
- fixed evaluation sampling budgets independent of training config;
- train-only field-bias fitting, validation guards, unchanged non-bias tensors,
  validation-only posterior calibration, strict prior binding and fresh outputs;
- asset-catalog parity with the main cache CLI for LFQ/VQ, both axis-lock
  policies, and explicitly overridden point budgets;
- all four real example meshes, normalized SDF recomputation and data-import units.
- task-specific console output, log plots, split-labeled checkpoint previews,
  preserved preview RNG state and nonblank, moving point-cloud animations.
- pooled token-count entropy, exact 50%-mass ranks, unused vocabulary bins,
  split-bound LFQ cache histograms and redraws of the paper's saved VQ/LFQ counts.
- SDF-label-free inference, coordinate recovery, invalid-field CLI exit status,
  strict prior/tokenizer binding and checksum-verified pretrained preparation.
- fixed-Bernoulli masks, pooled BCE/accuracy, relaxed-state reductions, CPU RNG
  preservation, legacy-array opt-in and paper-precision checking;
- legacy uniform-query suffixes and original full-observation frame import;
- concrete CUDA device selection, config/CLI seed precedence and explicit
  failure when every mixed-precision update overflows.
- learned-centroid VQ with power-of-two and arbitrary vocabularies; EMA and
  hard-index restoration; categorical CE/counts and observed-index preservation;
- VQ/index-prior train/cache/evaluate/preview workflows, exact CPU resume,
  LFQ-cache categorical controls, and two-rank CPU/Gloo baseline training.

Run `pytest -q` to repeat these checks. The GitHub Actions workflow is supplied
but was not executed on the remote GitHub service during local preparation.
README links, local image references and the source-folder audit were also
checked. The preceding 462-test suite also passed in a separate temporary copy of
only the public whitelist files, with research-checkout paths removed from
the import search path. Release modules came from that copy.
No new upload ZIP was generated for this revision.
The CUDA/NCCL multi-GPU path is implemented, but multi-GPU hardware execution
and throughput were not measured in this validation.

The [26 October mesh audit](../data/mesh_implementation_audit.json) additionally
compares original and release decoding on 36 identical field states and records
512-object observed-cloud fidelity for original matched weights and the
complete twelve-epoch retrained tokenizer. It also verifies the original
160-cubed presentation export on two real test observations. No training was
added for these checks; numerical fidelity and export compatibility do not
establish a fresh-token relaxed-edge improvement.

The later [quality audit](../data/quality_refinement.json) records the selected
single-bias distance calibration, matched-budget validation and its complete
512-object test evaluation. It retains rejected two-epoch decoder refinement
and posterior-calibration comparisons separately. The accepted field changes
one existing bias tensor; the original model architecture, training engine and
tokenizer/prior binding checks remain unchanged.

## End-To-End Sample Workflow

The four licensed meshes were preprocessed on CPU, split into 2 training,
1 validation and 1 test object, and passed through tokenizer training,
split-preserving token export, prior training and all three evaluation modes.
The training smoke runs performed two optimizer updates per model. No-isosurface
outputs were retained as explicit failures rather than replaced with geometry.
These checks validate the implementation and interfaces, not learned quality.

The documented Python observation interface also ran on the bundled licensed
teacup with the locally verified reconstructive pair. Encoding needed no SDF
labels; oracle, hard-bit and relaxed completion each exported a valid mesh at
resolution 96, with strict prior bindings for completion. The asset-catalog
example is additionally checked against the main cache CLI for both LFQ/VQ
and both axis-lock policies, including an explicit point-budget override.

The training guide additionally shows actual 64-update CPU runs on the sample
split, with their raw training/validation logs and best-score markers. The
figures were inspected as PNGs for clipping and overlapping labels. The tiny tokenizer is not converged; field slices
without a zero crossing are labeled, and no paper-quality result is implied.

The README GIF is a 96-frame, looping 1920 x 640 real-geometry visualization,
using 8,192 displayed points, Open3D and 3x supersampling. It uses Figure 4's
blue/copper/jade point and halo colors and foreground axes. Both observation
frames are estimated independently; the folded panel combines complementary
spatial halves. Metadata records the source attribution and point discrepancy
around numerical precision, without fitted registration or learned geometry.

## Compatibility And Reproduction

The separate compatibility audit described in [CHECKPOINTS.md](CHECKPOINTS.md)
checked active research weights and objective parity. It does not make the
four-object examples a reproduction of the paper's benchmark. Two original
tokenizer/prior pairs were separately migrated, bound using their original token
metadata and prior configuration, and verified by SHA256 and strict state loading.
The local Hugging Face folder is prepared but has not been uploaded; see
[PRETRAINED.md](PRETRAINED.md). A licensed teacup was encoded and decoded on CPU
with the full-size reconstructive pair at resolution 48; oracle and 50%-masked
relaxed decoding both produced finite, watertight meshes. This is an inference
smoke test, not a held-out quality benchmark.
The explicit `legacy-bbox` path was separately checked against the earlier
full-size teacup encoding: bits, frame, center, scale and tokenizer hash were
exactly unchanged. Centroid normalization is the new raw-input default, not a
retroactive change to the paper's benchmark input protocol or learned weights.

Full benchmark geometry is not redistributed. Original split identifiers and
support histograms are included. A new reference-prior evaluation on the full
512-field test array passes 48 protocol/metric comparisons across five masking
ratios. At 50% masking it recovers 97.23% bit accuracy and 78.01% exact tokens.
An additional real-data CUDA smoke imported eight objects, performed a bf16
tokenizer update, exported matching tokens, updated the prior and decoded one
held-out completion. The initial FP16 tokenizer trial overflowed all eight
attempts and is explicitly not counted as a successful training run.
Unmodified reference weights were additionally tested on two actual held-out
objects per task at resolution 64. Reconstruction's raw/continuous/quantized
paths and completion's oracle/hard/relaxed paths all produced valid nonempty
meshes. These two-object checks are functional, not quality estimates.

Full-size original/release forward and backward comparisons used 16 real-object
cases across both pretrained pairs. All active weights match; frame differences
were zero on these cases. CUDA token indices agreed in 15/16 cases; the maximum
quantized SDF difference was 7.18e-5. Small scatter differences at LFQ thresholds
prevent a bitwise-GPU claim.

The [reproduction report](REPRODUCTION.md) and
[machine-readable audit](../data/reproduction_audit.json) distinguish fresh
inference and archived-row reaggregation from the later primary training audit.
Neither the four-object demo nor archived support/stress/mesh statistics are
presented as a fresh full-paper training reproduction.

## Baseline Extension

The [baseline guide](BASELINES.md) documents VQ and categorical-prior presets.
The complete small GPU workflow ran on eight real source objects: two VQ
optimizer updates, an eight-object integer-token export, two categorical-prior
updates, held-out CE validation, token evaluation, decoded-field evaluation and
a rendered checkpoint preview. Two held-out decoded-field evaluations completed;
any missing isosurface remains an explicit failure, not a fabricated mesh.
Full-size K=4096, 128 x 128 VQ and categorical presets each completed an effective
GPU update. These are bounded training checks, not convergence results.

Original checkpoint comparisons additionally observed exact VQ hard indices,
quantized planes and hard decoding on the audited feature input, and exact
categorical logits/predictions on one 128 x 128 map. All active original weights
load strictly. The original/release comparison used the same PyTorch 2.7 runtime;
the public GPU training checks used PyTorch 2.11. CUDA/NCCL multi-GPU execution
is still unverified. See [baseline evidence](../data/baseline_validation.json).

## Benchmark And Efficiency Revision

The [benchmark recipes](BENCHMARKS.md) provide dataset adapters, a train-to-metrics
workflow, matched cache support, common-mask mesh regeneration and paper boundary
metrics. Tests additionally cover token-only decoder bypass, point-only sampling
equivalence, uncompressed import, mesh scale normalization, vectorized Laplacians,
missing-mesh denominators, saved-mask replay and HDF5 scans without SDF labels.
The workflow smoke is explicitly excluded from quality claims. The manual CI
distributed job now includes all distributed tests, not just the original CLI file.

Fresh runs evaluated all 2,461 ShapeNet, 8,192 Objaverse-LVIS and 2,882 ScanObjectNN
observations under the explicitly recorded release stress protocol. Frozen
reference weights regenerated 512 complete mesh cohorts at resolution 56;
the new seeded geometry evaluation counted 5/8/8 relaxed/hard/categorical drift
failures, versus the published 5/6/8. Source protocols and threshold-borderline
differences are retained, not adjusted to obtain the published values.

The historical stress-source discrepancy is documented in
[the benchmark guide](BENCHMARKS.md#historical-stress-results); the previous
archive audit is no longer presented as frame-isolated verification. The
manuscript itself was not changed during code-release work.
Measured cohorts, protocols and report hashes are retained in
[benchmark validation](../data/benchmark_validation.json).

## Full Primary Training

The primary matched-completion workflow subsequently completed 12 tokenizer
epochs and 80 epochs for each prior, without pretrained initialization or
skipped updates. All seven workflow steps succeeded on the original matched
IDs, with a validation holdout taken only from train. Fixed-mask evaluation
covered all 512 test fields and five masking ratios; all three decoded completion
paths produced 512 valid meshes at 75% masking. New and reference tokenizer
reconstruction were additionally compared on all 512 test objects.

The [training audit](../data/full_training_validation.json) retains checkpoint,
config, log and source hashes, actual budgets and measured results. It does not
turn budget completion into a convergence claim. The [interpretation](REPRODUCTION.md#from-scratch-validation)
includes the fresh tokenizer's lower effective support and the absence of a
relaxed-edge advantage in this matched run. Full K=8192 VQ/LFQ support training,
full-shape decoder retraining and CUDA/NCCL multi-GPU execution remain unverified.

The initial workflow trained the tokenizer for twelve epochs but selected its
seven-epoch `best.pt` using untruncated uniform SDF error. That checkpoint
selection did not match the paper's fixed-budget protocol. The reproduction
workflow now selects `last.pt` explicitly; `best.pt` remains an optional,
separately labeled validation-selected experiment. Paper presets also restore
epoch-wise cosine scheduling and monitor all five mask ratios. These workflow
corrections do not, by themselves, establish scientific reproduction.

## Boundary Follow-Up

The full-shape fixed-cache decode was evaluated with five prespecified mesh
sampling seeds. Relaxed mean edge error was lower than hard and categorical
decoding in every repeat. Paired object-bootstrap intervals include zero;
these repeats are not evidence of training-seed robustness. Two new point
extractions did not retain the mean-edge advantage. All results, including a
missing categorical mesh, are retained in the
[boundary audit](../data/boundary_reproduction.json).

The subsequent [recovery study](REPRODUCTION.md#reproduction-recovery) separates
fixed-budget tokenizer selection, precision/schedule controls and equal-budget
mask-coverage refinement. It also repeats full-shape geometry after paired
prior adaptation to fresh tokens. The latter does **not** recover the relaxed
edge advantage; invalid meshes and common-valid comparisons remain explicit.
See the [path-free recovery audit](../data/reproduction_recovery.json) for
measured outcomes, validation histories and all-object failure counts.

Additional tests verify the reference per-cell sampling algorithm and RNG draws,
explicit cache metadata, the unchanged portable sampling order, paired-object
bootstrap aggregation and missing-mesh denominators. A direct comparison against
the previous public data loader also found identical returned tensors on four
full-size real test objects. The reference adapter is opt-in and cache-only;
the original default training and export sampling behavior is unchanged.

[Back to Main README](../README.md#evaluation)
