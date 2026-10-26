<!-- Geometric Orbit Representatives for Unaligned 3D Tokenization -->
<!-- Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig -->
<!-- Contact: Xiang Chen <xiang.chen@uni-leipzig.de> | SPDX-License-Identifier: MIT -->

# Paper Benchmark Recipes

[Back to Main README](../README.md)

`orbit evaluate` provides general diagnostics. `orbit benchmark` implements
explicit paper-metric protocols. A successful command is not by itself evidence
of converged quality; reports retain missing geometry, sample counts, source
hashes, normalization and masks. Pretrained weights remain unpublished.

## Train To Metrics

The primary completion experiment uses the matched **K=4096**, eight-category
cohort, not the full-shape reconstructive checkpoint. Its original 4,096 train
and 512 test IDs are in [matched_completion.json](../data/splits/matched_completion.json).
Create a deterministic validation subset from train, without altering test:

```bash
python tools/make_validation_split.py \
  --source data/splits/matched_completion.json --output data/matched_with_val.json

orbit import-data --input-dir data/reference_npzs \
  --split-file data/matched_with_val.json --input-scale 2 \
  --legacy-uniform-start 70000 --storage uncompressed \
  --frame-cache data/reference_frames/train.npz data/reference_frames/test.npz \
  --output data/imported_matched

python tools/reproduce.py --data data/imported_matched \
  --output outputs/primary --device cuda --workers 4
```

The importer is for the known legacy archive, not arbitrary NPZs. Points and
distances are scaled once. Frame caches preserve original full-observation
frames; omit them only when deliberately recomputing frames. Uncompressed NPZs
use more disk but avoid compression/decompression overhead. Both storage modes
preserve the same arrays.

The workflow uses public commands to train the matched tokenizer, export tokens,
train bit and categorical priors, evaluate all five masking ratios, regenerate
meshes, and compute decoded-boundary metrics. It uses fixed-budget `last.pt`
checkpoints, matching the original experiment, and writes `workflow.json`, resolved YAMLs and per-command
logs. The full mode verifies both train/validation and test ID cohorts. It never
overwrites a prior run. `--checkpoint-selection best` explicitly selects a
different, validation-selected experiment. Resume an interrupted individual training command with
its saved config and `last.pt`; a new workflow needs a new output directory.

Multi-GPU and precision options are explicit, for example:

```bash
python tools/reproduce.py --data data/imported_matched \
  --output outputs/primary_bf16 --device cuda --gpus 2 --workers 4 \
  --precision bf16-mixed --tokenizer-batch-size 1 --tokenizer-accumulation 8
```

The effective tokenizer batch is per-rank batch times accumulation times ranks
(16 in this example). Precision, validation holdout and schedule interval
are part of the new run's recorded protocol; they are not bitwise replay of
the original optimization trajectory. The current presets execute 12 tokenizer
epochs and 80 epochs per prior. Completion of this budget is **not** an automatic
convergence or benchmark-agreement claim. Paper presets use epoch-wise cosine
scheduling; demos retain step-wise scheduling. Compare measured values with the
reported 97.23% bit / 78.01% token accuracy, using a stated tolerance.
The first [complete training validation](REPRODUCTION.md#from-scratch-validation)
reports 98.14% / 83.83% on the unchanged 512-object test cohort, alongside
reconstruction and support differences; it is not labeled a full-paper reproduction.

For the four examples, run the same command with `--smoke --device cpu` after
Quick Start preprocessing. Smoke uses small models and two updates, and is
explicitly labeled in the report. It has no paper-quality pass/fail gate.

## Decoded Fidelity

The paper's reconstruction table uses the matched tokenizer, not the separate
full-shape completion pair. Its evaluator subsamples the encoder observation
to 2,048 points, estimates a frame from that observation, and compares each
decoded mesh with the same representative-frame observation. The release
provides this contract independently of the general full-observation evaluator:

```bash
orbit benchmark fidelity --checkpoint outputs/primary/tokenizer/last.pt \
  --data data/imported_matched --split test --device cuda \
  --resolution 64 --metric-points 2048 --emd-points 512 \
  --encoder-points 8192 --query-points 8192 --uniform-points 2048 \
  --output outputs/primary_fidelity
```

`reference` point sampling is the default; use `--sampling-protocol portable`
for the release sampler. Both use recorded local RNGs. This replays the original
selection algorithm and measurement definitions, not its unrecorded global or
worker RNG stream. The initial encoder draw follows `PCsize`; the subsequent
`--metric-points` subset is the actual observation used for frame estimation,
encoding and the mesh reference. No cached full-cloud frame is substituted.
The three explicit sampling budgets keep observations and query draws matched
when comparing checkpoints trained with different point/query budgets. Omitting
them uses each checkpoint's own configuration; reports record the actual draws.

The JSON reports all three decoder branches: symmetric half-sum squared CD
times 1,000, Hungarian EMD times 100 on the first 512 points, signed-query
accuracy including exact-zero targets as outside, and per-observation inside
gap. KD-tree nearest neighbours replace the dense CD distance matrix without
changing the metric. Invalid meshes retain their sample denominator and null
CD/EMD. Histogram support here belongs to these subsampled observations, not
the full-resolution token cache. No smoothing or component filtering is applied.

The general `evaluate reconstruction` and `evaluate completion` commands also
default to model-space `[-0.5, 0.5]`. Their explicit `--bounds -0.6 0.6` option
replays earlier release reports, whose coordinates and protocol remain recorded.

## Decoded Boundary Metrics

Use one frozen tokenizer, its own token cache, and priors bound to that exact
tokenizer. For example, after training both priors on `data/full_shape_tokens`:

```bash
orbit benchmark decode --tokens data/full_shape_tokens \
  --tokenizer outputs/full_shape/last.pt \
  --bit-prior outputs/full_shape_bits/last.pt \
  --categorical-prior outputs/full_shape_indices/last.pt \
  --mask-ratio .75 --resolution 56 --device cuda --output outputs/boundary
```

All paths share one mask and oracle. The command writes a decoded cohort
manifest and `masks.npy`, then metrics under `outputs/boundary/metrics`.
`--mask-npy` can replay original NHW masks; `--sample-indices` accepts a JSON
list of source-cache row numbers. Masks are in **selected row order**, not an
implicit shuffled order. Legacy external arrays also accept `--indices`,
`--indices-metadata` and an explicit unverified-binding flag, as documented in
[reproduction](REPRODUCTION.md). Known tokenizer mismatches always fail.

The default 56-cubed grid and radius-0.5 bounds match the archived boundary meshes.
The grid resolution was recovered independently from their vertex lattice;
the old result JSON did not store it. Reports record resolution and the original
component cleanup rule (2% of the largest component, at least 64 triangles).
Changing grid resolution can change thin-part failures and is not just a
rendering preference.

Re-evaluate saved meshes, including the original directory layout:

```bash
orbit benchmark geometry --decode-dir data/saved_decodes \
  --sample-points 1024 --expected-samples 512 --output outputs/mesh_metrics
```

Paper metrics are **CD_n = sum of the two mean squared nearest-neighbor distances
divided by oracle bbox diagonal squared**, with no factor 0.5; Drift is the
fraction exceeding 0.005; F-score uses 1% of the same diagonal. Edge distance is
mean NN distance from the oracle's highest-Laplacian 20% vertex subset, divided
by the diagonal. Normal consistency uses absolute matched normal dot products.
These are not field-gradient drift or SDF MAE.

The Open3D surface sampler is now seeded per pair. The original run seeded NumPy
but not Open3D, so sampling values and threshold-borderline counts can differ.
Reports provide both valid-pair drift and failure-inclusive rates. Missing
meshes remain in the cohort, never becoming zero errors or silently disappearing.

### Boundary Repetition

The exact frozen-boundary cohort is listed in
[decoded_boundary.json](../data/splits/decoded_boundary.json): the original
first 512 test IDs from the full-shape split, not the balanced primary-completion
split. Its tokenizer/prior pair is `reconstructive`. Reuse the saved original
mask for comparison to the paper; a newly generated mask is a different run.

To prepare this exact geometry cohort from the legacy archive:

```bash
orbit import-data --input-dir data/reference_npzs \
  --split-file data/splits/decoded_boundary.json --input-scale 2 \
  --legacy-uniform-start 70000 --storage uncompressed \
  --frame-cache data/reference_frames/test.npz --output data/imported_boundary
```

After mesh decoding, repeat surface sampling without changing meshes or masks:

```bash
python tools/check_boundary_reproduction.py --decode-dir outputs/boundary \
  --output outputs/boundary_repeats --expected-samples 512
```

The default seeds are 20260518-20260522; all are reported, not ranked. The tool
checks unchanged mesh hashes and computes paired intervals by resampling objects
after averaging sampling repeats within each object. It records excluded edge
pairs and retains missing meshes in the per-run failure-inclusive summaries.
These repeats do not measure training-seed robustness.

For newly extracted tokens, default `cache-tokens` sampling remains unchanged.
An optional compatibility mode preserves the original per-cell selection
algorithm with an explicit, worker-independent local RNG:

```bash
orbit cache-tokens --checkpoint checkpoints/official/reconstructive/tokenizer.pt \
  --data data/imported_boundary --sampling-protocol reference --seed 20260511 \
  --device cuda --workers 4 --output data/boundary_reference_tokens
```

It is encoder-only and records the protocol in the token manifest. It does not
reconstruct the old unrecorded worker/global RNG stream or promise better quality.
The [full-shape validation](REPRODUCTION.md#full-shape-boundary-validation)
separates original-cache edge reproduction from fresh-extraction sensitivity.

## Fresh Token Support

After training matched VQ/LFQ K=8192 models and exporting their test caches:

```bash
orbit benchmark support --vq-tokens data/vq_tokens --lfq-tokens data/lfq_tokens \
  --vocabulary 8192 --split test --output outputs/fresh_support
```

This checks input IDs, token-grid dimensions, vocabulary and geometry fingerprint
before pooling counts. It does not substitute saved paper histograms. The saved
`paper_token_support.json` remains an explicitly archival reference. The
[baseline guide](BASELINES.md) describes how to change K in both models and
rebuild all caches. Matching cohorts alone does not prove matching train budgets.

## Dataset Stress

Install `python -m pip install -e ".[benchmark]"` for ScanObjectNN's HDF5 reader.
No SDF or watertight mesh is required for these point-only tests.

```bash
orbit benchmark stability --format scanobjectnn --data data/scans/test.h5 \
  --limit 2882 --expected-samples 2882 --device cuda --output outputs/scan_stress

orbit benchmark stability --format meshes --data data/objaverse_meshes \
  --limit 8192 --expected-samples 8192 --device cpu --output outputs/asset_stress

orbit benchmark stability --format shapenet --data data/reference_npzs \
  --split-file data/splits/reference_shapes.json \
  --categories 02691156 02933112 02958343 03001627 03636649 04256520 04379243 04530566 \
  --max-per-category 400 --expected-samples 2461 --output outputs/shape_stress
```

Each selected observation is sampled to 2,048 points, centered and normalized
to radius 0.5. Independent Gaussian jitter has **unit-radius sigma 0.005**,
therefore model sigma 0.0025, clipped at three sigma. Both estimated frames act
on the same unperturbed reference. Matched PCA-only and locked frames use
identical points/noise. Reports include median, p95, tails, identities and explicit
failures; invalid assets are never replaced by another asset. Labels only choose
an optional evaluation slice. No yaw minimization or learned category input is used.

## Historical Stress Results

Fresh release results under the explicit unit-radius sigma 0.005 protocol are:

| Source | N | Orbit Median | Orbit p95 | PCA p95 |
| --- | ---: | ---: | ---: | ---: |
| ShapeNet | 2,461 | 0.000764 | 1.124 | 96.35 |
| Objaverse-LVIS | 8,192 | 0.001577 | 1.724 | 86.63 |
| ScanObjectNN | 2,882 | 0.000757 | 0.02233 | 91.84 |

All numbers are frame-isolated PAE x 1e4, not distance between independently
noisy point sets. These fresh results are not substituted into the manuscript.

The earlier camera-ready figures 14.7x/65.2x came from the `matched_independent`
archive. A later `frameisolated` archive has an explicit metric-contract column
and different results (Objaverse 87.73/5.761; ScanObjectNN 89.54/0.1517, PCA/Orbit
p95). The prior release audit selected the older archive. Its stress section
is now marked deprecated; it is **not** treated as verified frame-isolated evidence.

New release runs state their normalization, noise units, point budget, sampler
and float64 KD-tree precision explicitly. They are not expected to recover the
old float32 CUDA numerical floor or an incompletely recorded historical RNG
stream. Do not mix these protocols or describe their discrepancy as a successful
exact reproduction. Manuscripts have not been silently changed by this code update.

[Back to Main README](../README.md#evaluation)
