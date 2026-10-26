<!-- Geometric Orbit Representatives for Unaligned 3D Tokenization -->
<!-- Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig -->
<!-- Contact: Xiang Chen <xiang.chen@uni-leipzig.de> | SPDX-License-Identifier: MIT -->

# Data Preparation

[Back to Main README](../README.md)

## Meshes

`orbit preprocess` accepts PLY, OBJ, STL and OFF meshes. It validates
watertightness, centers vertices at their centroid, normalizes maximum radius
to 0.5, orients each connected
shell against material occupancy (including cavity walls),
samples surface points and normals, and computes SDF supervision with Open3D
ray casting. Non-watertight meshes are rejected for this training path. Raw
point observations can use [direct encoding](INFERENCE.md) and the independent
frame-stability evaluator without SDF labels. This does not establish raw-scan
reconstruction quality.

```bash
orbit preprocess --input-dir data/my_meshes --output data/processed \
  --surface-points 8192 --query-points 32768 --uniform-points 8192 \
  --split-file data/split.json
```

The split JSON lists disjoint `train`, `val` and `test` mesh stems. Every input
must occur exactly once. Without a split file, the four bundled models use
two train, one validation and one test object. Their [attributions and
licenses](../data/attribution.md) apply to all derived samples and renders.
Full benchmark datasets are not redistributed.

Centroid normalization is shared with raw-input inference and stability
evaluation and commutes with rotation. `--normalization legacy-bbox` retains
the earlier bounding-box convention for compatibility; apply it once before
benchmark rotations. Imported research NPZs keep their existing centering and
only apply the explicit input scale.

Place output outside the input tree. Split validation rejects aliases of the
same source file and byte-identical source files assigned to different splits
before writing. This detects file-level duplicates, not geometrically equivalent
assets stored in different formats.

## Stored Arrays

Each processed NPZ has the following fields, indexed by `manifest.json`:

| Field | Shape | Convention |
| --- | --- | --- |
| `points`, `normals` | N x 3 | Model-space surface and unit normals |
| `frame` | 3 x 3 | Row-vector rotation: `points @ frame` |
| `query_points`, `query_sdf` | Q x 3, Q | Negative inside; same units as points |
| `uniform_points`, `uniform_sdf` | U x 3, U | Independent uniform-field supervision |
| `center`, `scale` | 3, scalar | Raw = model coordinates * scale + center |

The frame is estimated from the full stored cloud **before** encoder
subsampling. The loader does not rescale processed points or SDF targets.
Token export loads only points and the frame, with identical encoder sampling;
it does not decompress or validate unused SDF supervision. Training and field
evaluation still validate every supervision array. `cache-tokens --workers N`
enables parallel loading without changing the per-sample RNG contract.
The default encoder sampler is `portable`. The opt-in
`cache-tokens --sampling-protocol reference --seed N` preserves the original
per-cell selection algorithm using a local RandomState, independent of worker
assignment. It is a cache-only compatibility check, not a different training
default or recovery of an unrecorded historical RNG stream. Both choices are
identified in the cache manifest; see [boundary repetition](BENCHMARKS.md#boundary-repetition).
Sampling bands, shell offsets and query locations all use model-space units.
The manifest records `frame_policy` with its PCA method and axis-lock flag.
`--no-axis-lock` creates a PCA-only frame cache. Training, caching and evaluation
honor `EnableAxisRepresentativeLock`; when the stored policy differs or is
unknown, they estimate the requested frame from the full stored observation
before selecting encoder points, without overwriting the data. The effective
policy and sampling settings are written into run/evaluation/cache metadata.
The low-level `GeometryDataset(axis_lock=None)` explicitly retains stored frames.
The manifest records preprocessing counts, normalization, source checksums and
each processed NPZ's SHA256. Training and token export fingerprint the actual
referenced file contents, not just manifest text; declared checksum mismatches
are errors. Legacy manifests without checksums still receive content-based
fingerprints. Changing preprocessing requires a new training run or an explicit
[`--init-checkpoint`](TRAINING.md#initialize-from-weights), not exact resume.

For independent noise tests, model sigma `0.0025` corresponds to unit-radius
sigma `0.005`. Gaussian jitter is clipped per coordinate at three sigma.
Both estimated frames act on a common unperturbed reference, isolating frame
error from differences in the noisy observations.

## Existing SDF Data

```bash
orbit import-data --input-dir data/validated_npzs --split-file data/split.json \
  --input-scale 2.0 --output data/imported
```

The adapter reads `encoder_points`/`encoder_normals` or
`point_cloud`/`normals`, plus `query_points`/`query_sdf` and explicit uniform
arrays or an identified uniform query segment. Points and signed distances
are scaled together, once. Missing uniform supervision is an error.

For the paper's legacy archive, which stores uniform queries as a suffix but
has no segment metadata, explicitly select its known offset. Optional original
frame-cache NPZs preserve frames computed before input scaling:

```bash
orbit import-data --input-dir data/reference_npzs \
  --split-file data/splits/matched_completion.json --input-scale 2.0 \
  --legacy-uniform-start 70000 \
  --frame-cache data/reference_frames/train.npz data/reference_frames/test.npz \
  --output data/imported
```

Frame caches require `rel_paths` and proper `rotations` arrays, must cover every
imported identity, and must correspond to the requested axis-lock policy.
Hashes are retained in the manifest. Missing or conflicting frames are errors.
Do not apply this offset or frame cache to an unrelated preprocessing layout.
See `orbit import-data --help` and [preprocessing.py](../src/orbit_folding/preprocessing.py).

## Token Cache Types

LFQ exports retain binary uint8 CHW fields (`orbit_folding.token_cache.v1`).
VQ exports use integer int64 HW code indices and an explicit vocabulary size
(`orbit_folding.index_cache.v1`), without fabricated bit planes. Both preserve
sample identity, split, category metadata, geometry provenance and tokenizer
hash. Categorical priors accept either cache; bit priors accept only LFQ.
See [baseline training and evaluation](BASELINES.md).

[Back to Main README](../README.md#quick-start)
